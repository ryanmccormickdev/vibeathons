#!/usr/bin/env python3
"""
TreeHacks .env File Analysis
=============================
Reads treehacks_projects_enriched.json and checks whether projects committed
environment files (files with ".env" in the name) to their GitHub repos.

Produces graphs in ./analysis_graphs/ showing:
  - Number and percentage of projects with committed .env files per year
  - Breakdown of .env file variants found
  - .env files appearing in commit diffs

Usage:
    python env_files.py
"""

import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_FILE = "treehacks_projects_enriched.json"
OUTPUT_DIR = Path("env_analysis_graphs")
DPI = 180
FIG_W, FIG_H = 12, 7
sns.set_theme(style="whitegrid", font_scale=1.05)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

YEAR_ORDER = [
    "treehacks_2015", "treehacks_2016", "treehacks_2017", "treehacks_2018",
    "treehacks_2019", "treehacks_2020", "treehacks_2021", "treehacks_2022",
    "treehacks_2023", "treehacks_2024", "treehacks_2025", "treehacks_2026",
]

# Pattern to match .env files — basename must contain ".env"
# Matches: .env, .env.local, .env.production, .env.example, config.env, etc.
ENV_FILE_RE = re.compile(r"(^|/)\.env(\b|$|\.)", re.IGNORECASE)
# Stricter: only actual secret-carrying env files (exclude .env.example, .env.sample, .env.template)
ENV_SECRET_RE = re.compile(
    r"(^|/)\.env(\.local|\.production|\.development|\.staging|\.test)?$",
    re.IGNORECASE,
)
ENV_EXAMPLE_RE = re.compile(
    r"\.env\.(example|sample|template|dist)",
    re.IGNORECASE,
)

# Pattern to match .DS_Store files (macOS Finder metadata — should never be committed)
DS_STORE_RE = re.compile(r"(^|/)\.DS_Store$")


def load_data():
    print(f"Loading {INPUT_FILE} …")
    with open(INPUT_FILE) as f:
        data = json.load(f)
    print("Loaded.")
    return data


def year_int(year_key: str) -> int:
    return int(year_key.replace("treehacks_", ""))


def save(fig, num: int, name: str):
    fname = OUTPUT_DIR / f"env_{num:02d}_{name}.png"
    fig.savefig(fname, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [env_{num:02d}] {fname.name}")


# ── Data extraction ───────────────────────────────────────────────────────────

def _make_entry(path, repo_full_name, default_branch, last_sha=None):
    return {
        "path": path,
        "repo_full_name": repo_full_name,
        "default_branch": default_branch,
        "last_sha": last_sha,  # most recent commit SHA where file was present
    }


def _build_last_sha_map(repo):
    """For every file touched in commits_detailed, find the most recent commit
    SHA where the file was present (status != 'removed').  Commits are assumed
    to be returned newest-first by the GitHub API."""
    last_sha = {}          # path -> sha (first non-removal hit = newest)
    all_seen_paths = set() # every path that appeared in any commit
    for commit in repo.get("commits_detailed", []):
        sha = commit.get("sha")
        if not sha:
            continue
        for cf in commit.get("files", []):
            fname = cf.get("filename", "")
            if not fname:
                continue
            all_seen_paths.add(fname)
            status = (cf.get("status") or "").lower()
            if fname not in last_sha and status != "removed":
                last_sha[fname] = sha
    return last_sha


def find_env_files_in_repo(repo):
    """
    Given a repo metadata dict, return lists of .env and .DS_Store file paths
    found in files_sample and in commit diffs.  Each entry is a dict with
    'path', 'repo_full_name', 'default_branch', and 'last_sha' (the most
    recent commit SHA where the file was present) for GitHub link construction.
    """
    repo_full_name = repo.get("repo_full_name", "")
    default_branch = (repo.get("basic") or {}).get("default_branch", "main")

    # Pre-compute the latest SHA map for all files in commit history
    last_sha_map = _build_last_sha_map(repo)

    env_in_tree = []
    env_in_commits = []
    ds_in_tree = []
    ds_in_commits = []

    # 1) Check files_sample (full repo tree)
    for f in repo.get("files_sample", []):
        path = f.get("path", "")
        sha = last_sha_map.get(path)  # may be None if file wasn't in fetched commits
        if ENV_FILE_RE.search(path):
            env_in_tree.append(_make_entry(path, repo_full_name, default_branch, sha))
        if DS_STORE_RE.search(path):
            ds_in_tree.append(_make_entry(path, repo_full_name, default_branch, sha))

    # 2) Check committed files in commits_detailed
    for commit in repo.get("commits_detailed", []):
        for cf in commit.get("files", []):
            fname = cf.get("filename", "")
            sha = last_sha_map.get(fname)
            if ENV_FILE_RE.search(fname):
                env_in_commits.append(_make_entry(fname, repo_full_name, default_branch, sha))
            if DS_STORE_RE.search(fname):
                ds_in_commits.append(_make_entry(fname, repo_full_name, default_branch, sha))

    return env_in_tree, env_in_commits, ds_in_tree, ds_in_commits


def build_env_rows(data):
    """
    For each project, determine whether .env files exist in the repo tree or commits.
    Returns a list of dicts (one per project).
    """
    rows = []
    for year_key in YEAR_ORDER:
        if year_key not in data:
            continue
        projects = data[year_key]
        if not isinstance(projects, list):
            continue
        year = year_int(year_key)

        for pidx, proj in enumerate(projects):
            gm_list = proj.get("github_metadata", [])

            # Collect across all valid repos for this project
            all_env_tree = []
            all_env_commits = []
            all_ds_tree = []
            all_ds_commits = []
            has_valid_repo = False

            for repo in gm_list:
                if repo.get("error"):
                    continue
                has_valid_repo = True
                tree_envs, commit_envs, tree_ds, commit_ds = find_env_files_in_repo(repo)
                all_env_tree.extend(tree_envs)
                all_env_commits.extend(commit_envs)
                all_ds_tree.extend(tree_ds)
                all_ds_commits.extend(commit_ds)

            # Helper to extract path strings for counting/classification
            tree_paths = [e["path"] for e in all_env_tree]
            commit_paths = [e["path"] for e in all_env_commits]

            # Classify env files
            secret_tree = [p for p in tree_paths if ENV_SECRET_RE.search(p)]
            example_tree = [p for p in tree_paths if ENV_EXAMPLE_RE.search(p)]
            other_tree = [p for p in tree_paths
                          if not ENV_SECRET_RE.search(p) and not ENV_EXAMPLE_RE.search(p)]

            secret_commits = [p for p in commit_paths if ENV_SECRET_RE.search(p)]

            row = {
                "year": year,
                "year_key": year_key,
                "project_idx": pidx,
                "project_title": proj.get("project_title", ""),
                "project_url": proj.get("project_url", ""),
                "has_github": has_valid_repo,
                # Tree-level (current state of repo) — full dicts with repo info
                "env_files_in_tree": all_env_tree,
                "env_paths_in_tree": tree_paths,
                "num_env_in_tree": len(all_env_tree),
                "has_env_in_tree": len(all_env_tree) > 0,
                "has_secret_env_in_tree": len(secret_tree) > 0,
                "has_example_env_in_tree": len(example_tree) > 0,
                "num_secret_env_in_tree": len(secret_tree),
                "num_example_env_in_tree": len(example_tree),
                # Commit-level (env files that appeared in commit diffs)
                "env_files_in_commits": all_env_commits,
                "env_paths_in_commits": list(set(commit_paths)),
                "num_env_in_commits": len(set(commit_paths)),
                "has_env_in_commits": len(all_env_commits) > 0,
                "has_secret_env_in_commits": len(secret_commits) > 0,
                # .DS_Store tracking
                "ds_files_in_tree": all_ds_tree,
                "num_ds_in_tree": len(all_ds_tree),
                "has_ds_in_tree": len(all_ds_tree) > 0,
                "ds_files_in_commits": all_ds_commits,
                "has_ds_in_commits": len(all_ds_commits) > 0,
            }
            rows.append(row)

    return rows


# ── Graph functions ────────────────────────────────────────────────────────────

def plot_env_analysis(df):
    num = 1
    gh = df[df["has_github"]].copy()
    yrs = sorted(gh["year"].unique())
    yr_labels = [str(y) for y in yrs]

    # ── 1. Projects with any .env file in repo tree ──────────────────────────
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    env_count = gh[gh["has_env_in_tree"]].groupby("year").size()
    total_count = gh.groupby("year").size()
    x = np.arange(len(yrs))
    w = 0.35
    t_vals = [total_count.get(y, 0) for y in yrs]
    e_vals = [env_count.get(y, 0) for y in yrs]
    ax.bar(x - w / 2, t_vals, w, label="Total w/ GitHub", color="#7bafd4")
    ax.bar(x + w / 2, e_vals, w, label="Has .env file(s)", color="#e41a1c")
    ax.set_xticks(x)
    ax.set_xticklabels(yr_labels)
    for i, v in enumerate(e_vals):
        if v > 0:
            ax.text(i + w / 2, v + 0.5, str(v), ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Projects With .env Files in Repo Tree Per Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Projects")
    ax.legend()
    save(fig, num, "env_files_count_per_year")
    num += 1

    # ── 2. Percentage of projects with .env files ────────────────────────────
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    rate = [e_vals[i] / t_vals[i] * 100 if t_vals[i] else 0 for i in range(len(yrs))]
    ax.plot(yr_labels, rate, marker="o", linewidth=2.5, color="#e41a1c")
    ax.fill_between(yr_labels, rate, alpha=0.15, color="#e41a1c")
    ax.set_title("% of Projects With Committed .env Files Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of Projects (with GitHub)")
    ax.set_ylim(bottom=0)
    for i, (r, e) in enumerate(zip(rate, e_vals)):
        if e > 0:
            ax.annotate(f"{r:.1f}%\n({e})", (yr_labels[i], r),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, fontweight="bold")
    save(fig, num, "env_files_rate_over_time")
    num += 1

    # ── 3. Secret vs example .env files ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    secret_counts = gh[gh["has_secret_env_in_tree"]].groupby("year").size()
    example_counts = gh[gh["has_example_env_in_tree"]].groupby("year").size()
    s_vals = [secret_counts.get(y, 0) for y in yrs]
    ex_vals = [example_counts.get(y, 0) for y in yrs]
    x = np.arange(len(yrs))
    w = 0.3
    ax.bar(x - w / 2, s_vals, w, label="Likely secrets (.env, .env.local, …)", color="#e41a1c")
    ax.bar(x + w / 2, ex_vals, w, label="Example/template (.env.example, …)", color="#4daf4a")
    ax.set_xticks(x)
    ax.set_xticklabels(yr_labels)
    for i, (sv, ev) in enumerate(zip(s_vals, ex_vals)):
        if sv > 0:
            ax.text(i - w / 2, sv + 0.3, str(sv), ha="center", fontsize=8, color="#e41a1c")
        if ev > 0:
            ax.text(i + w / 2, ev + 0.3, str(ev), ha="center", fontsize=8, color="#2d7d2d")
    ax.set_title("Secret vs Example .env Files in Repo Tree", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Projects")
    ax.legend()
    save(fig, num, "secret_vs_example_env")
    num += 1

    # ── 4. .env files appearing in commit diffs ──────────────────────────────
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    commit_env = gh[gh["has_env_in_commits"]].groupby("year").size()
    c_vals = [commit_env.get(y, 0) for y in yrs]
    c_rate = [c_vals[i] / t_vals[i] * 100 if t_vals[i] else 0 for i in range(len(yrs))]
    ax.bar(yr_labels, c_vals, color="#ff7f00", alpha=0.7, label="Count")
    ax2 = ax.twinx()
    ax2.plot(yr_labels, c_rate, marker="s", color="#984ea3", linewidth=2.5, label="% Rate")
    ax2.set_ylabel("% of Projects")
    ax.set_title(".env Files Appearing in Commit Diffs Per Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Projects")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    save(fig, num, "env_in_commit_diffs")
    num += 1

    # ── 5. Most common .env file variants ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(FIG_W, 8))
    all_env_paths = []
    for _, row in gh.iterrows():
        for e in row["env_files_in_tree"]:
            p = e["path"]
            # Extract just the filename (basename)
            basename = p.rsplit("/", 1)[-1] if "/" in p else p
            all_env_paths.append(basename)
    if all_env_paths:
        counts = Counter(all_env_paths).most_common(20)
        names, vals = zip(*counts)
        y_pos = range(len(names))
        ax.barh(y_pos, vals, color=sns.color_palette("Reds_r", len(names)))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_title("Most Common .env File Names Across All Years", fontsize=15, fontweight="bold")
        ax.set_xlabel("Occurrences")
    else:
        ax.text(0.5, 0.5, "No .env files found in any repo", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.set_title("Most Common .env File Names", fontsize=15, fontweight="bold")
    save(fig, num, "common_env_filenames")
    num += 1

    # ── 6. Heatmap: env file variants by year ────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))
    env_by_year = defaultdict(lambda: Counter())
    for _, row in gh.iterrows():
        for e in row["env_files_in_tree"]:
            p = e["path"]
            basename = p.rsplit("/", 1)[-1] if "/" in p else p
            env_by_year[row["year"]][basename] += 1
    # Get top variants
    all_variants = Counter()
    for y_counts in env_by_year.values():
        all_variants.update(y_counts)
    top_variants = [v for v, _ in all_variants.most_common(15)]
    if top_variants and env_by_year:
        matrix = []
        active_yrs = sorted(env_by_year.keys())
        for y in active_yrs:
            matrix.append([env_by_year[y].get(v, 0) for v in top_variants])
        hm_df = pd.DataFrame(matrix, index=[str(y) for y in active_yrs], columns=top_variants)
        sns.heatmap(hm_df.T, annot=True, fmt="d", cmap="OrRd", ax=ax, linewidths=0.5)
        ax.set_title(".env File Variants by Year (Count)", fontsize=15, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Filename")
    else:
        ax.text(0.5, 0.5, "No .env files found", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.set_title(".env File Variants by Year", fontsize=15, fontweight="bold")
    save(fig, num, "env_variants_heatmap")
    num += 1

    # ── 7. Number of .env files per project (among those that have any) ──────
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    env_projects = gh[gh["has_env_in_tree"]].copy()
    if len(env_projects) > 0:
        med = env_projects.groupby("year")["num_env_in_tree"].median()
        mean = env_projects.groupby("year")["num_env_in_tree"].mean()
        active_yrs = sorted(env_projects["year"].unique())
        active_labels = [str(y) for y in active_yrs]
        ax.bar(active_labels, [med.get(y, 0) for y in active_yrs],
               alpha=0.6, label="Median", color="#66c2a5")
        ax.plot(active_labels, [mean.get(y, 0) for y in active_yrs],
                marker="s", label="Mean", color="#e41a1c", linewidth=2)
        ax.set_title("Number of .env Files Per Project (Among Projects With Any)", fontsize=15, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Count of .env Files")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No projects with .env files", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.set_title("Number of .env Files Per Project", fontsize=15, fontweight="bold")
    save(fig, num, "env_count_per_project")
    num += 1

    # ── 8. .DS_Store files per year ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ds_count = gh[gh["has_ds_in_tree"]].groupby("year").size()
    ds_vals = [ds_count.get(y, 0) for y in yrs]
    ds_rate = [ds_vals[i] / t_vals[i] * 100 if t_vals[i] else 0 for i in range(len(yrs))]
    ax.bar(yr_labels, ds_vals, color="#756bb1", alpha=0.7, label="Count")
    ax2 = ax.twinx()
    ax2.plot(yr_labels, ds_rate, marker="s", color="#e6550d", linewidth=2.5, label="% Rate")
    ax2.set_ylabel("% of Projects")
    for i, (v, r) in enumerate(zip(ds_vals, ds_rate)):
        if v > 0:
            ax.text(i, v + 0.3, str(v), ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Projects With Committed .DS_Store Files Per Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Projects")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    save(fig, num, "ds_store_per_year")
    num += 1

    # ── 9. Summary table printed to console ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  .env FILE ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"{'Year':<8} {'Total':>8} {'w/ GitHub':>10} {'w/ .env':>8} {'Rate':>8} {'Secret':>8} {'Example':>8}")
    print("-" * 70)
    for y in yrs:
        sub = df[df["year"] == y]
        sub_gh = gh[gh["year"] == y]
        total = len(sub)
        with_gh = len(sub_gh)
        with_env = sub_gh["has_env_in_tree"].sum()
        with_secret = sub_gh["has_secret_env_in_tree"].sum()
        with_example = sub_gh["has_example_env_in_tree"].sum()
        rate = with_env / with_gh * 100 if with_gh else 0
        print(f"{y:<8} {total:>8} {with_gh:>10} {with_env:>8} {rate:>7.1f}% {with_secret:>8} {with_example:>8}")
    print("=" * 70)

    # Also list all projects with secret .env files
    secret_projects = gh[gh["has_secret_env_in_tree"]].copy()
    if len(secret_projects) > 0:
        print(f"\n  Projects with likely-secret .env files ({len(secret_projects)} total):")
        print("-" * 70)
        for _, row in secret_projects.sort_values("year").iterrows():
            envs = [e["path"] for e in row["env_files_in_tree"] if ENV_SECRET_RE.search(e["path"])]
            print(f"  {row['year']} | {row['project_title'][:40]:<40} | {', '.join(envs[:5])}")
        print()

    return num


# ── Write GitHub links to text file ───────────────────────────────────────────

ENV_LINKS_FILE = "env_file_links.txt"


def write_env_links(df):
    """Write a text file listing GitHub links for every .env file found."""
    gh = df[df["has_github"]].copy()
    lines = []

    lines.append("=" * 90)
    lines.append("  GitHub Links to .env Files Found in TreeHacks Projects")
    lines.append("=" * 90)
    lines.append("")

    # ── Section 1: .env files in repo tree (current state) ──
    tree_projects = gh[gh["has_env_in_tree"]].sort_values("year")
    lines.append(f"SECTION 1: .env files in repo tree ({len(tree_projects)} projects)")
    lines.append("-" * 90)
    total_links = 0
    for _, row in tree_projects.iterrows():
        lines.append(f"")
        lines.append(f"  [{row['year']}] {row['project_title']}")
        if row["project_url"]:
            lines.append(f"         Devpost: {row['project_url']}")
        for env_file in row["env_files_in_tree"]:
            repo = env_file["repo_full_name"]
            ref = env_file["last_sha"] or env_file["default_branch"]
            path = env_file["path"]
            github_url = f"https://github.com/{repo}/blob/{ref}/{path}"
            is_secret = "[SECRET] " if ENV_SECRET_RE.search(path) else ""
            lines.append(f"         {is_secret}{github_url}")
            total_links += 1
    lines.append("")
    lines.append(f"  Total tree links: {total_links}")
    lines.append("")

    # ── Section 2: .env files in commit diffs ──
    commit_projects = gh[gh["has_env_in_commits"]].sort_values("year")
    lines.append(f"SECTION 2: .env files appearing in commit diffs ({len(commit_projects)} projects)")
    lines.append("-" * 90)
    total_commit_links = 0
    for _, row in commit_projects.iterrows():
        # Deduplicate by (repo, path)
        seen = set()
        lines.append(f"")
        lines.append(f"  [{row['year']}] {row['project_title']}")
        if row["project_url"]:
            lines.append(f"         Devpost: {row['project_url']}")
        for env_file in row["env_files_in_commits"]:
            repo = env_file["repo_full_name"]
            ref = env_file["last_sha"] or env_file["default_branch"]
            path = env_file["path"]
            key = (repo, path)
            if key in seen:
                continue
            seen.add(key)
            github_url = f"https://github.com/{repo}/blob/{ref}/{path}"
            is_secret = "[SECRET] " if ENV_SECRET_RE.search(path) else ""
            lines.append(f"         {is_secret}{github_url}")
            total_commit_links += 1
    lines.append("")
    lines.append(f"  Total commit-diff links: {total_commit_links}")
    lines.append("")

    # ── Section 3: .DS_Store files in repo tree ──
    ds_projects = gh[gh["has_ds_in_tree"]].sort_values("year")
    lines.append(f"SECTION 3: .DS_Store files in repo tree ({len(ds_projects)} projects)")
    lines.append("-" * 90)
    total_ds_links = 0
    for _, row in ds_projects.iterrows():
        lines.append(f"")
        lines.append(f"  [{row['year']}] {row['project_title']}")
        if row["project_url"]:
            lines.append(f"         Devpost: {row['project_url']}")
        for ds_file in row["ds_files_in_tree"]:
            repo = ds_file["repo_full_name"]
            ref = ds_file["last_sha"] or ds_file["default_branch"]
            path = ds_file["path"]
            github_url = f"https://github.com/{repo}/blob/{ref}/{path}"
            lines.append(f"         {github_url}")
            total_ds_links += 1
    lines.append("")
    lines.append(f"  Total .DS_Store links: {total_ds_links}")
    lines.append("")

    grand_total = total_links + total_commit_links + total_ds_links
    output = "\n".join(lines)
    with open(ENV_LINKS_FILE, "w") as f:
        f.write(output)
    print(f"\n  GitHub links written to {ENV_LINKS_FILE} ({grand_total} links total)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_data()

    print("Scanning repos for .env files …")
    env_rows = build_env_rows(data)
    df = pd.DataFrame(env_rows)
    print(f"  {len(df)} projects scanned across {df['year'].nunique()} years")
    print(f"  {df['has_env_in_tree'].sum()} projects have .env files in tree")
    print(f"  {df['has_env_in_commits'].sum()} projects have .env files in commit diffs")
    print(f"  {df['has_ds_in_tree'].sum()} projects have .DS_Store files in tree")

    del data

    print(f"\nGenerating .env analysis graphs into {OUTPUT_DIR}/ …\n")
    plot_env_analysis(df)

    # Write GitHub links for all .env files to a text file
    write_env_links(df)

    print(f"\nDone — graphs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
