#!/usr/bin/env python3
"""
TreeHacks Vibe-Coding Analysis — Comprehensive Graph Generation
================================================================
Reads treehacks_projects_enriched.json and produces a folder of annotated PNG
graphs covering every angle of the data, with special focus on detecting
vibe-coding patterns (AI-assisted bulk code generation) in hackathon projects.

Usage:
    python generate_all_graphs.py

Output:
    ./analysis_graphs/  — folder of numbered PNG files
"""

import json
import os
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_FILE = "treehacks_projects_enriched.json"
OUTPUT_DIR = Path("analysis_graphs")
DPI = 180
FIG_W, FIG_H = 12, 7  # default figure size
sns.set_theme(style="whitegrid", font_scale=1.05)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

YEAR_ORDER = [
    "treehacks_2015", "treehacks_2016", "treehacks_2017", "treehacks_2018",
    "treehacks_2019", "treehacks_2020", "treehacks_2021", "treehacks_2022",
    "treehacks_2023", "treehacks_2024", "treehacks_2025", "treehacks_2026",
]
YEAR_LABELS = [k.replace("treehacks_", "") for k in YEAR_ORDER]

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data():
    print(f"Loading {INPUT_FILE} …")
    with open(INPUT_FILE) as f:
        data = json.load(f)
    print("Loaded.")
    return data


def year_int(year_key: str) -> int:
    return int(year_key.replace("treehacks_", ""))


def save(fig, num: int, name: str):
    fname = OUTPUT_DIR / f"{num:03d}_{name}.png"
    fig.savefig(fname, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [{num:03d}] {fname.name}")


def add_description(ax, text: str, fontsize=8):
    """Put a description below the axes (inside the figure)."""
    return
    ax.annotate(
        text, xy=(0.5, -0.01), xycoords="figure fraction",
        ha="center", va="top", fontsize=fontsize,
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#cccccc", alpha=0.95),
    )


# ── Data extraction ───────────────────────────────────────────────────────────

def build_project_rows(data):
    """Flatten per-project summary stats into a list of dicts (one per project)."""
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
            # Take the first valid repo (primary)
            repo = None
            for r in gm_list:
                if not r.get("error"):
                    repo = r
                    break

            row = {
                "year_key": year_key,
                "year": year,
                "project_idx": pidx,
                "project_title": proj.get("project_title", ""),
                "project_url": proj.get("project_url", ""),
                "num_repos": len(gm_list),
                "num_valid_repos": sum(1 for r in gm_list if not r.get("error")),
                "has_github": repo is not None,
            }

            if repo:
                basic = repo.get("basic", {})
                row["repo_full_name"] = repo.get("repo_full_name", "")
                row["stars"] = basic.get("stargazers_count")
                row["forks"] = basic.get("forks_count")
                row["size_kb"] = basic.get("size_kb")
                row["open_issues"] = basic.get("open_issues_count")
                row["primary_language"] = basic.get("primary_language")
                row["license_spdx"] = (basic.get("license") or {}).get("spdx_id")
                row["archived"] = basic.get("archived")
                row["fork_flag"] = basic.get("fork")
                row["created_at"] = basic.get("created_at")
                row["pushed_at"] = basic.get("pushed_at")

                misc = repo.get("misc", {})
                row["has_wiki"] = misc.get("has_wiki")
                row["has_pages"] = misc.get("has_pages")
                row["has_discussions"] = misc.get("has_discussions")

                langs = repo.get("languages", {})
                row["num_languages"] = len(langs)
                row["total_lang_bytes"] = sum(langs.values()) if langs else 0
                row["languages_dict"] = langs

                contribs = repo.get("contributors", [])
                row["num_contributors"] = len(contribs)
                if contribs:
                    row["top_contributor_pct"] = (
                        max(c.get("contributions", 0) for c in contribs)
                        / max(sum(c.get("contributions", 0) for c in contribs), 1)
                    )
                else:
                    row["top_contributor_pct"] = None

                cs = repo.get("commits_summary", {})
                row["total_commits"] = cs.get("total_commits_fetched")
                row["total_additions"] = cs.get("total_additions")
                row["total_deletions"] = cs.get("total_deletions")
                row["total_files_changed"] = cs.get("total_files_changed")
                row["first_commit_date"] = cs.get("first_commit_date")
                row["last_commit_date"] = cs.get("last_commit_date")
                by_author = cs.get("by_author", {})
                row["num_commit_authors"] = len(by_author)

                fs = repo.get("files_summary", {})
                row["total_files"] = fs.get("total_files")
                row["total_file_size_bytes"] = fs.get("total_size_bytes")
                row["by_extension"] = fs.get("by_extension", {})
                row["by_top_level_dir"] = fs.get("by_top_level_dir", {})

                ad = repo.get("authors_detailed", [])
                row["num_authors_detailed"] = len(ad)
            else:
                for k in ["repo_full_name", "stars", "forks", "size_kb",
                           "open_issues", "primary_language", "license_spdx",
                           "archived", "fork_flag", "created_at", "pushed_at",
                           "has_wiki", "has_pages", "has_discussions",
                           "num_languages", "total_lang_bytes", "languages_dict",
                           "num_contributors", "top_contributor_pct",
                           "total_commits", "total_additions", "total_deletions",
                           "total_files_changed", "first_commit_date",
                           "last_commit_date", "num_commit_authors",
                           "total_files", "total_file_size_bytes",
                           "by_extension", "by_top_level_dir",
                           "num_authors_detailed"]:
                    row[k] = None

            rows.append(row)
    return rows


def build_commit_rows(data):
    """Flatten individual commits into rows for commit-level analysis."""
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
            repo = None
            for r in gm_list:
                if not r.get("error"):
                    repo = r
                    break
            if not repo:
                continue
            commits = repo.get("commits_detailed", [])
            for c in commits:
                msg = c.get("message") or ""
                stats = c.get("stats") or {}
                date_str = c.get("date")
                rows.append({
                    "year": year,
                    "year_key": year_key,
                    "project_idx": pidx,
                    "repo_full_name": repo.get("repo_full_name", ""),
                    "sha": c.get("sha"),
                    "author_login": c.get("author_login"),
                    "author_name": c.get("author_name"),
                    "author_email": c.get("author_email"),
                    "date": date_str,
                    "message": msg,
                    "msg_len": len(msg),
                    "msg_word_count": len(msg.split()),
                    "msg_first_line": msg.split("\n")[0] if msg else "",
                    "msg_first_line_len": len(msg.split("\n")[0]) if msg else 0,
                    "additions": stats.get("additions", 0),
                    "deletions": stats.get("deletions", 0),
                    "total_churn": stats.get("total", 0),
                    "num_files": len(c.get("files", [])),
                })
    return rows


# ── Graph functions ────────────────────────────────────────────────────────────
# Each function draws one or more figures and returns the next figure number.

def section_header(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────

def plot_overview(df, num):
    section_header("SECTION 1: Overview & Coverage")

    # 1 — Total projects per year
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    counts = df.groupby("year").size()
    ax.bar(counts.index.astype(str), counts.values, color=sns.color_palette("Blues_d", len(counts)))
    ax.set_title("Total Projects Submitted Per Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Projects")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontsize=9)
    save(fig, num, "projects_per_year"); num += 1

    # 2 — Projects with valid GitHub repos
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    total = df.groupby("year").size()
    valid = df[df["has_github"]].groupby("year").size()
    x = np.arange(len(YEAR_LABELS))
    w = 0.35
    present_years = sorted(df["year"].unique())
    labels = [str(y) for y in present_years]
    t_vals = [total.get(y, 0) for y in present_years]
    v_vals = [valid.get(y, 0) for y in present_years]
    ax.bar(x - w/2, t_vals, w, label="Total Projects", color="#7bafd4")
    ax.bar(x + w/2, v_vals, w, label="With Valid GitHub Repo", color="#2c7fb8")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Projects With Valid GitHub Repos Per Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Count")
    ax.legend()
    save(fig, num, "github_coverage_per_year"); num += 1

    # 3 — GitHub coverage rate
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    rate = [(v_vals[i] / t_vals[i] * 100) if t_vals[i] else 0 for i in range(len(present_years))]
    ax.plot(labels, rate, marker="o", linewidth=2.5, color="#d95f02")
    ax.fill_between(labels, rate, alpha=0.15, color="#d95f02")
    ax.set_title("GitHub Repo Coverage Rate Over Time (%)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% Projects with Valid GitHub Repo")
    ax.set_ylim(0, 105)
    save(fig, num, "github_coverage_rate"); num += 1

    return num


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: REPOSITORY BASICS
# ──────────────────────────────────────────────────────────────────────────────

def plot_repo_basics(df, num):
    section_header("SECTION 2: Repository Basics")
    gh = df[df["has_github"]].copy()

    # 4 — Median repo size by year
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med_size = gh.groupby("year")["size_kb"].median()
    ax.bar(med_size.index.astype(str), med_size.values, color=sns.color_palette("YlOrRd", len(med_size)))
    ax.set_title("Median Repository Size (KB) by Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Size (KB)")
    save(fig, num, "median_repo_size_kb"); num += 1

    # 5 — Repo size distribution (box)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    box_data = [gh[gh["year"] == y]["size_kb"].dropna().values for y in sorted(gh["year"].unique())]
    bp = ax.boxplot(box_data, labels=[str(y) for y in sorted(gh["year"].unique())],
                    showfliers=False, patch_artist=True)
    colors = sns.color_palette("Set2", len(box_data))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_title("Repository Size Distribution by Year (Outliers Hidden)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Size (KB)")
    save(fig, num, "repo_size_distribution_box"); num += 1

    # 6 — Stars distribution (violin)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    star_data = gh[gh["stars"].notna()].copy()
    star_data["stars_clipped"] = star_data["stars"].clip(upper=star_data["stars"].quantile(0.95))
    years_present = sorted(star_data["year"].unique())
    vd = [star_data[star_data["year"] == y]["stars_clipped"].values for y in years_present]
    parts = ax.violinplot(vd, positions=range(len(years_present)), showmedians=True)
    ax.set_xticks(range(len(years_present)))
    ax.set_xticklabels([str(y) for y in years_present])
    ax.set_title("Stars Distribution by Year (Top 5% Clipped)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Stars")
    save(fig, num, "stars_distribution_violin"); num += 1

    # 7 — Forks distribution
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fork_med = gh.groupby("year")["forks"].median()
    fork_mean = gh.groupby("year")["forks"].mean()
    yrs = sorted(gh["year"].unique())
    ax.bar([str(y) for y in yrs], [fork_med.get(y, 0) for y in yrs], alpha=0.7, label="Median", color="#66c2a5")
    ax.plot([str(y) for y in yrs], [fork_mean.get(y, 0) for y in yrs], marker="s", color="#e7298a", label="Mean", linewidth=2)
    ax.set_title("Forks Per Repo — Mean vs Median by Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Forks")
    ax.legend()
    save(fig, num, "forks_mean_median"); num += 1

    # 8 — Contributors per project
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med_c = gh.groupby("year")["num_contributors"].median()
    mean_c = gh.groupby("year")["num_contributors"].mean()
    ax.plot([str(y) for y in yrs], [med_c.get(y, 0) for y in yrs], marker="o", label="Median", linewidth=2)
    ax.plot([str(y) for y in yrs], [mean_c.get(y, 0) for y in yrs], marker="s", label="Mean", linewidth=2)
    ax.set_title("Contributors Per Project Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Contributors")
    ax.legend()
    save(fig, num, "contributors_per_project"); num += 1

    # 9 — License adoption
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    gh_lic = gh.copy()
    gh_lic["has_license"] = gh_lic["license_spdx"].notna() & (gh_lic["license_spdx"] != "NOASSERTION")
    lic_rate = gh_lic.groupby("year")["has_license"].mean() * 100
    ax.bar([str(y) for y in sorted(lic_rate.index)], lic_rate.values, color="#8da0cb")
    ax.set_title("License Adoption Rate Over Time (%)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% Repos with a License")
    save(fig, num, "license_adoption_rate"); num += 1

    # 10 — Top 20 most starred projects
    fig, ax = plt.subplots(figsize=(FIG_W, 9))
    top = gh.nlargest(20, "stars")[["repo_full_name", "year", "stars"]].reset_index(drop=True)
    labels = [f"{r['repo_full_name']} ({r['year']})" for _, r in top.iterrows()]
    ax.barh(range(len(top)), top["stars"].values, color=sns.color_palette("magma", len(top)))
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Top 20 Most Starred TreeHacks Projects (All Time)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Stars")
    save(fig, num, "top20_starred_projects"); num += 1

    # 11 — Fork flag (is-fork) rate
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    gh_fork = gh[gh["fork_flag"].notna()].copy()
    fork_rate = gh_fork.groupby("year")["fork_flag"].mean() * 100
    ax.plot([str(y) for y in sorted(fork_rate.index)], fork_rate.values, marker="o", linewidth=2.5, color="#e41a1c")
    ax.set_title("Percentage of Repos That Are Forks, by Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% Repos That Are Forks")
    save(fig, num, "fork_rate_over_time"); num += 1

    return num


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: LANGUAGE TRENDS
# ──────────────────────────────────────────────────────────────────────────────

def plot_language_trends(df, num):
    section_header("SECTION 3: Language Trends")
    gh = df[df["has_github"]].copy()

    # 12 — Top primary languages over time (heatmap)
    fig, ax = plt.subplots(figsize=(14, 8))
    lang_counts = gh.groupby(["year", "primary_language"]).size().unstack(fill_value=0)
    # Keep top 12 languages overall
    top_langs = lang_counts.sum().nlargest(12).index.tolist()
    lang_filt = lang_counts[top_langs]
    sns.heatmap(lang_filt.T, annot=True, fmt="d", cmap="YlGnBu", ax=ax, linewidths=0.5)
    ax.set_title("Primary Language Counts — Top 12 Languages by Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Language")
    save(fig, num, "primary_language_heatmap"); num += 1

    # 13 — Language diversity per project
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    div = gh.groupby("year")["num_languages"].agg(["mean", "median"])
    yrs = sorted(div.index)
    ax.plot([str(y) for y in yrs], [div.loc[y, "mean"] for y in yrs], marker="o", label="Mean", linewidth=2)
    ax.plot([str(y) for y in yrs], [div.loc[y, "median"] for y in yrs], marker="s", label="Median", linewidth=2)
    ax.set_title("Language Diversity: Avg Languages Per Repo Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Number of Languages")
    ax.legend()
    save(fig, num, "language_diversity"); num += 1

    # 14 — Python vs JS vs TS share over time
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for lang, color in [("Python", "#3572A5"), ("JavaScript", "#f1e05a"), ("TypeScript", "#2b7489")]:
        rates = []
        for y in sorted(gh["year"].unique()):
            sub = gh[gh["year"] == y]
            count = (sub["primary_language"] == lang).sum()
            rates.append(count / len(sub) * 100 if len(sub) else 0)
        ax.plot([str(y) for y in sorted(gh["year"].unique())], rates, marker="o", label=lang, linewidth=2.5, color=color)
    ax.set_title("Python vs JavaScript vs TypeScript — Primary Language Share", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% of Projects")
    ax.legend()
    save(fig, num, "python_js_ts_share"); num += 1

    # 15 — Stacked area chart of top languages (bytes-based)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    lang_bytes = defaultdict(lambda: defaultdict(int))
    for _, row in gh.iterrows():
        ld = row.get("languages_dict")
        if not isinstance(ld, dict):
            continue
        for lang, byt in ld.items():
            lang_bytes[row["year"]][lang] += byt
    years_s = sorted(lang_bytes.keys())
    all_langs_total = Counter()
    for y in years_s:
        for l, b in lang_bytes[y].items():
            all_langs_total[l] += b
    top8 = [l for l, _ in all_langs_total.most_common(8)]
    stack = {l: [] for l in top8}
    stack["Other"] = []
    for y in years_s:
        total_y = sum(lang_bytes[y].values())
        other = 0
        for l in top8:
            stack[l].append(lang_bytes[y].get(l, 0) / max(total_y, 1) * 100)
        for l, b in lang_bytes[y].items():
            if l not in top8:
                other += b
        stack["Other"].append(other / max(total_y, 1) * 100)
    bottom = np.zeros(len(years_s))
    colors = sns.color_palette("tab10", len(top8) + 1)
    for i, l in enumerate(top8 + ["Other"]):
        ax.bar([str(y) for y in years_s], stack[l], bottom=bottom, label=l, color=colors[i])
        bottom += np.array(stack[l])
    ax.set_title("Language Composition by Bytes (Top 8 + Other)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% of Total Bytes")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    save(fig, num, "language_bytes_stacked"); num += 1

    # 16 — Language count distribution (violin per year)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    yrs_p = sorted(gh["year"].unique())
    vdata = [gh[gh["year"] == y]["num_languages"].dropna().values for y in yrs_p]
    parts = ax.violinplot(vdata, positions=range(len(yrs_p)), showmedians=True)
    ax.set_xticks(range(len(yrs_p)))
    ax.set_xticklabels([str(y) for y in yrs_p])
    ax.set_title("Number of Languages Per Repo — Distribution by Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Language Count")
    save(fig, num, "language_count_violin"); num += 1

    return num


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: CODEBASE COMPOSITION
# ──────────────────────────────────────────────────────────────────────────────

def plot_codebase_composition(df, num):
    section_header("SECTION 4: Codebase Composition")
    gh = df[(df["has_github"]) & (df["total_files"].notna())].copy()

    # 17 — Avg total files per repo
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med = gh.groupby("year")["total_files"].median()
    mean = gh.groupby("year")["total_files"].mean()
    yrs = sorted(med.index)
    ax.bar([str(y) for y in yrs], [med.get(y, 0) for y in yrs], alpha=0.6, label="Median", color="#66c2a5")
    ax.plot([str(y) for y in yrs], [mean.get(y, 0) for y in yrs], marker="s", color="#e7298a", label="Mean", linewidth=2)
    ax.set_title("Files Per Repository Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("File Count")
    ax.legend()
    save(fig, num, "files_per_repo"); num += 1

    # 18 — Total code size per repo
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    gh["total_file_size_mb"] = gh["total_file_size_bytes"] / 1e6
    med_s = gh.groupby("year")["total_file_size_mb"].median()
    ax.bar([str(y) for y in sorted(med_s.index)], med_s.values, color=sns.color_palette("OrRd", len(med_s)))
    ax.set_title("Median Total Codebase Size (MB) by Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Size (MB)")
    save(fig, num, "median_codebase_size_mb"); num += 1

    # 19 — Top file extensions heatmap
    fig, ax = plt.subplots(figsize=(14, 9))
    ext_counts = defaultdict(lambda: defaultdict(int))
    for _, row in gh.iterrows():
        be = row.get("by_extension")
        if not isinstance(be, dict):
            continue
        for ext, info in be.items():
            if isinstance(info, dict):
                ext_counts[row["year"]][ext if ext else "(no ext)"] += info.get("count", 0)
    years_s = sorted(ext_counts.keys())
    all_ext_total = Counter()
    for y in years_s:
        for e, c in ext_counts[y].items():
            all_ext_total[e] += c
    top_ext = [e for e, _ in all_ext_total.most_common(20)]
    matrix = []
    for y in years_s:
        total_y = sum(ext_counts[y].values())
        matrix.append([(ext_counts[y].get(e, 0) / max(total_y, 1) * 100) for e in top_ext])
    df_ext = pd.DataFrame(matrix, index=[str(y) for y in years_s], columns=top_ext)
    sns.heatmap(df_ext.T, annot=True, fmt=".1f", cmap="YlOrBr", ax=ax, linewidths=0.5)
    ax.set_title("Top 20 File Extensions — % of Files by Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Extension")
    save(fig, num, "file_extension_heatmap"); num += 1

    # 20 — Top-level directories diversity
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    dir_counts = []
    for _, row in gh.iterrows():
        btld = row.get("by_top_level_dir")
        if isinstance(btld, dict):
            dir_counts.append({"year": row["year"], "num_dirs": len(btld)})
    if dir_counts:
        ddf = pd.DataFrame(dir_counts)
        med_d = ddf.groupby("year")["num_dirs"].median()
        yrs = sorted(med_d.index)
        ax.plot([str(y) for y in yrs], [med_d.get(y, 0) for y in yrs], marker="o", linewidth=2.5, color="#7570b3")
        ax.set_title("Median Number of Top-Level Directories by Year", fontsize=15, fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("Directory Count")
    save(fig, num, "top_level_dirs_count"); num += 1

    # 21 — Presence of key framework files over time
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    framework_markers = {
        "package.json": [],      # Node.js
        "requirements.txt": [],  # Python
        "Dockerfile": [],        # Docker
        "README.md": [],         # Documentation
    }
    for y in sorted(gh["year"].unique()):
        sub = gh[gh["year"] == y]
        total = len(sub)
        for marker in framework_markers:
            count = 0
            for _, row in sub.iterrows():
                btld = row.get("by_top_level_dir")
                be = row.get("by_extension")
                # Check top-level dir names or extension-based heuristic
                # Since we have files_sample in the raw data but not in our df,
                # we'll approximate by checking by_top_level_dir keys
                # This is imperfect but directional
                if isinstance(btld, dict):
                    # Rough proxy: check for common dir names
                    pass
            # Better approach: check by_extension for the file type
            if marker == "package.json":
                count = sum(1 for _, r in sub.iterrows()
                           if isinstance(r.get("by_extension"), dict) and ".json" in r.get("by_extension", {}))
            elif marker == "requirements.txt":
                count = sum(1 for _, r in sub.iterrows()
                           if isinstance(r.get("by_extension"), dict) and ".txt" in r.get("by_extension", {}))
            elif marker == "Dockerfile":
                count = sum(1 for _, r in sub.iterrows()
                           if isinstance(r.get("by_extension"), dict) and "" in r.get("by_extension", {}))
            elif marker == "README.md":
                count = sum(1 for _, r in sub.iterrows()
                           if isinstance(r.get("by_extension"), dict) and ".md" in r.get("by_extension", {}))
            framework_markers[marker].append(count / max(total, 1) * 100)
    yrs = sorted(gh["year"].unique())
    for marker, vals in framework_markers.items():
        ax.plot([str(y) for y in yrs], vals, marker="o", label=marker, linewidth=2)
    ax.set_title("Repos with Common File Types Over Time (%)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% of Repos")
    ax.legend()
    save(fig, num, "framework_file_presence"); num += 1

    # ── Project Scope: lines added, lines deleted, estimated LOC ─────────────
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    scope = gh.dropna(subset=["total_additions", "total_deletions"]).copy()
    # scope["est_loc"] = scope["total_file_size_bytes"].fillna(0) / 40  # rough LOC proxy
    yrs_s = sorted(scope["year"].unique())
    yr_labels_s = [str(y) for y in yrs_s]
    x = np.arange(len(yrs_s))
    w = 0.25

    med_add = scope.groupby("year")["total_additions"].median()
    med_del = scope.groupby("year")["total_deletions"].median()
    # med_loc = scope.groupby("year")["est_loc"].median()

    ax.bar(x - w, [med_add.get(y, 0) for y in yrs_s], w, label="Median Lines Added", color="#2ca02c")
    ax.bar(x,     [med_del.get(y, 0) for y in yrs_s], w, label="Median Lines Deleted", color="#d62728")
    # ax.bar(x + w, [med_loc.get(y, 0) for y in yrs_s], w, label="Est. Codebase LOC", color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels(yr_labels_s)
    ax.set_title("Project Scope: Lines Added, Deleted & Estimated LOC", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Lines (median)")
    ax.legend()
    save(fig, num, "project_scope_lines"); num += 1

    return num


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: COMMIT BEHAVIOR — VIBE-CODING CORE
# ──────────────────────────────────────────────────────────────────────────────

def plot_commit_behavior(df, cdf, num):
    section_header("SECTION 5: Commit Behavior (Vibe-Coding Indicators)")
    gh = df[(df["has_github"]) & (df["total_commits"].notna()) & (df["total_commits"] > 0)].copy()

    yrs = sorted(gh["year"].unique())
    yr_labels = [str(y) for y in yrs]

    # 22 — Average commits per project
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med = gh.groupby("year")["total_commits"].median()
    mean = gh.groupby("year")["total_commits"].mean()
    ax.bar(yr_labels, [med.get(y, 0) for y in yrs], alpha=0.6, label="Median", color="#a6d854")
    ax.plot(yr_labels, [mean.get(y, 0) for y in yrs], marker="s", color="#e41a1c", label="Mean", linewidth=2)
    ax.set_title("Commits Per Project Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Commits")
    ax.legend()
    save(fig, num, "commits_per_project"); num += 1

    # 23 — Commit count distribution (violin)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    clip_val = gh["total_commits"].quantile(0.95)
    vdata = [gh[gh["year"] == y]["total_commits"].clip(upper=clip_val).values for y in yrs]
    parts = ax.violinplot(vdata, positions=range(len(yrs)), showmedians=True)
    ax.set_xticks(range(len(yrs))); ax.set_xticklabels(yr_labels)
    ax.set_title("Commit Count Distribution by Year (Top 5% Clipped)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Commits")
    save(fig, num, "commit_count_distribution_violin"); num += 1

    # 24 — Single-commit repos fraction
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    single_rate = []
    few_rate = []  # <= 3 commits
    for y in yrs:
        sub = gh[gh["year"] == y]
        single_rate.append((sub["total_commits"] == 1).sum() / len(sub) * 100 if len(sub) else 0)
        few_rate.append((sub["total_commits"] <= 3).sum() / len(sub) * 100 if len(sub) else 0)
    ax.plot(yr_labels, single_rate, marker="o", label="Exactly 1 commit", linewidth=2.5, color="#e41a1c")
    ax.plot(yr_labels, few_rate, marker="s", label="≤ 3 commits", linewidth=2.5, color="#984ea3")
    ax.set_title("Low-Commit Repos: Fraction Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% of Repos")
    ax.legend()
    save(fig, num, "single_commit_repos_fraction"); num += 1

    # 25 — Additions per commit (mean)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    gh["adds_per_commit"] = gh["total_additions"] / gh["total_commits"]
    med_apc = gh.groupby("year")["adds_per_commit"].median()
    mean_apc = gh.groupby("year")["adds_per_commit"].mean()
    ax.bar(yr_labels, [med_apc.get(y, 0) for y in yrs], alpha=0.6, label="Median", color="#fdbf6f")
    ax.plot(yr_labels, [mean_apc.get(y, 0) for y in yrs], marker="s", color="#e31a1c", label="Mean", linewidth=2)
    ax.set_title("Lines Added Per Commit — Mean & Median", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Additions / Commit")
    ax.legend()
    save(fig, num, "additions_per_commit"); num += 1

    # 26 — Deletions per commit
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    gh["dels_per_commit"] = gh["total_deletions"] / gh["total_commits"]
    med_dpc = gh.groupby("year")["dels_per_commit"].median()
    ax.bar(yr_labels, [med_dpc.get(y, 0) for y in yrs], color="#fb9a99")
    ax.set_title("Median Lines Deleted Per Commit by Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Deletions / Commit")
    save(fig, num, "deletions_per_commit"); num += 1

    # 27 — Addition-to-deletion ratio
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    gh["add_del_ratio"] = gh["total_additions"] / gh["total_deletions"].replace(0, np.nan)
    med_ratio = gh.groupby("year")["add_del_ratio"].median()
    ax.plot(yr_labels, [med_ratio.get(y, 0) for y in yrs], marker="o", linewidth=2.5, color="#ff7f00")
    ax.set_title("Median Addition/Deletion Ratio by Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Additions / Deletions")
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="1:1 ratio")
    ax.legend()
    save(fig, num, "addition_deletion_ratio"); num += 1

    # 28 — "Bulk commit" detection: repos where single largest commit > 80% of additions
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    # We need commit-level data for this
    bulk_rates = []
    for y in yrs:
        year_commits = cdf[cdf["year"] == y]
        repos = year_commits.groupby("repo_full_name")
        total_repos = repos.ngroups
        bulk_count = 0
        for rname, rgroup in repos:
            total_add = rgroup["additions"].sum()
            if total_add > 0:
                max_add = rgroup["additions"].max()
                if max_add / total_add >= 0.8:
                    bulk_count += 1
        bulk_rates.append(bulk_count / max(total_repos, 1) * 100)
    ax.plot(yr_labels, bulk_rates, marker="o", linewidth=2.5, color="#e41a1c")
    ax.fill_between(yr_labels, bulk_rates, alpha=0.15, color="#e41a1c")
    ax.set_title("'Bulk Commit' Repos: Single Commit Has ≥80% of All Additions", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% of Repos")
    save(fig, num, "bulk_commit_repos"); num += 1

    # 29 — Total additions per project (total code output)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med_add = gh.groupby("year")["total_additions"].median()
    mean_add = gh.groupby("year")["total_additions"].mean()
    ax.bar(yr_labels, [med_add.get(y, 0) for y in yrs], alpha=0.6, label="Median", color="#b2df8a")
    ax.plot(yr_labels, [mean_add.get(y, 0) for y in yrs], marker="s", color="#33a02c", label="Mean", linewidth=2)
    ax.set_title("Total Lines Added Per Project Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Lines Added")
    ax.legend()
    save(fig, num, "total_additions_per_project"); num += 1

    # 30 — Total files changed per project
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med_fc = gh.groupby("year")["total_files_changed"].median()
    ax.bar(yr_labels, [med_fc.get(y, 0) for y in yrs], color="#cab2d6")
    ax.set_title("Median Total Files Changed (Across All Commits) by Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Files Changed")
    save(fig, num, "total_files_changed_per_project"); num += 1

    # 31 — Commit timespan (first to last commit)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    def timespan_hours(row):
        try:
            f = datetime.fromisoformat(row["first_commit_date"].replace("Z", "+00:00"))
            l = datetime.fromisoformat(row["last_commit_date"].replace("Z", "+00:00"))
            return (l - f).total_seconds() / 3600
        except:
            return np.nan
    gh["timespan_hours"] = gh.apply(timespan_hours, axis=1)
    med_ts = gh.groupby("year")["timespan_hours"].median()
    ax.bar(yr_labels, [med_ts.get(y, 0) for y in yrs], color="#fbb4ae")
    ax.set_title("Median Development Timespan (Hours, First→Last Commit)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Hours")
    save(fig, num, "development_timespan_hours"); num += 1

    return num


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: COMMIT MESSAGE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def plot_commit_messages(cdf, num):
    section_header("SECTION 6: Commit Message Analysis")

    yrs = sorted(cdf["year"].unique())
    yr_labels = [str(y) for y in yrs]

    # 32 — Average commit message length (chars)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med = cdf.groupby("year")["msg_len"].median()
    mean = cdf.groupby("year")["msg_len"].mean()
    ax.bar(yr_labels, [med.get(y, 0) for y in yrs], alpha=0.6, label="Median", color="#80b1d3")
    ax.plot(yr_labels, [mean.get(y, 0) for y in yrs], marker="s", color="#e41a1c", label="Mean", linewidth=2)
    ax.set_title("Commit Message Length (Characters) Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Characters")
    ax.legend()
    save(fig, num, "commit_message_length_chars"); num += 1

    # 33 — First-line length
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med_fl = cdf.groupby("year")["msg_first_line_len"].median()
    ax.plot(yr_labels, [med_fl.get(y, 0) for y in yrs], marker="o", linewidth=2.5, color="#8856a7")
    ax.set_title("Median Commit Message First Line Length Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Characters (first line)")
    save(fig, num, "commit_first_line_length"); num += 1

    # 34 — Word count
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med_wc = cdf.groupby("year")["msg_word_count"].median()
    ax.plot(yr_labels, [med_wc.get(y, 0) for y in yrs], marker="o", linewidth=2.5, color="#2ca25f")
    ax.set_title("Median Commit Message Word Count Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Words")
    save(fig, num, "commit_message_word_count"); num += 1

    # 35 — Empty/very short commit messages
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    short_rates = []
    empty_rates = []
    for y in yrs:
        sub = cdf[cdf["year"] == y]
        n = len(sub) if len(sub) else 1
        short_rates.append((sub["msg_first_line_len"] <= 10).sum() / n * 100)
        empty_rates.append((sub["msg_len"] == 0).sum() / n * 100)
    ax.plot(yr_labels, short_rates, marker="o", label="First line ≤ 10 chars", linewidth=2.5, color="#d95f02")
    ax.plot(yr_labels, empty_rates, marker="s", label="Empty message", linewidth=2.5, color="#e41a1c")
    ax.set_title("Short & Empty Commit Messages Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% of Commits")
    ax.legend()
    save(fig, num, "short_empty_commit_messages"); num += 1

    # 36 — Common commit message patterns
    fig, ax = plt.subplots(figsize=(14, 9))
    patterns = {
        "initial commit": r"^initial commit",
        "first commit": r"^first commit",
        "update": r"^update",
        "fix": r"^fix",
        "add": r"^add",
        "merge": r"^merge",
        "feat": r"^feat",
        "wip": r"^wip",
        "init": r"^init\b",
        "create": r"^create",
        "test": r"^test",
        "refactor": r"^refactor",
    }
    pattern_rates = {p: [] for p in patterns}
    for y in yrs:
        sub = cdf[cdf["year"] == y]
        n = max(len(sub), 1)
        for pname, patt in patterns.items():
            count = sub["msg_first_line"].str.lower().str.match(patt, na=False).sum()
            pattern_rates[pname].append(count / n * 100)
    for pname, vals in pattern_rates.items():
        ax.plot(yr_labels, vals, marker=".", label=pname, linewidth=1.5)
    ax.set_title("Common Commit Message Patterns Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% of Commits Starting With Pattern")
    ax.legend(fontsize=8, ncol=3)
    save(fig, num, "commit_message_patterns"); num += 1

    # 37 — Commit message length distribution (violin per year)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    clip = cdf["msg_first_line_len"].quantile(0.95)
    vdata = [cdf[cdf["year"] == y]["msg_first_line_len"].clip(upper=clip).values for y in yrs]
    # filter out empty arrays
    valid_yrs = [(y, v) for y, v in zip(yrs, vdata) if len(v) > 0]
    if valid_yrs:
        parts = ax.violinplot([v for _, v in valid_yrs],
                              positions=range(len(valid_yrs)), showmedians=True)
        ax.set_xticks(range(len(valid_yrs)))
        ax.set_xticklabels([str(y) for y, _ in valid_yrs])
    ax.set_title("Commit Message First-Line Length Distribution (Top 5% Clipped)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Characters")
    save(fig, num, "commit_msg_length_violin"); num += 1

    # 38 — Lines changed per commit distribution
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    clip_churn = cdf["total_churn"].quantile(0.95)
    vdata = [cdf[cdf["year"] == y]["total_churn"].clip(upper=clip_churn).values for y in yrs]
    valid_yrs = [(y, v) for y, v in zip(yrs, vdata) if len(v) > 0]
    if valid_yrs:
        parts = ax.violinplot([v for _, v in valid_yrs],
                              positions=range(len(valid_yrs)), showmedians=True)
        ax.set_xticks(range(len(valid_yrs)))
        ax.set_xticklabels([str(y) for y, _ in valid_yrs])
    ax.set_title("Lines Changed Per Commit Distribution (Top 5% Clipped)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Lines Changed (add+del)")
    save(fig, num, "lines_per_commit_violin"); num += 1

    # 39 — Files per commit distribution
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med_nf = cdf.groupby("year")["num_files"].median()
    mean_nf = cdf.groupby("year")["num_files"].mean()
    ax.bar(yr_labels, [med_nf.get(y, 0) for y in yrs], alpha=0.6, label="Median", color="#b3cde3")
    ax.plot(yr_labels, [mean_nf.get(y, 0) for y in yrs], marker="s", color="#e41a1c", label="Mean", linewidth=2)
    ax.set_title("Files Touched Per Commit Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Files")
    ax.legend()
    save(fig, num, "files_per_commit"); num += 1

    return num


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7: AUTHOR & CONTRIBUTOR PATTERNS
# ──────────────────────────────────────────────────────────────────────────────

def plot_author_patterns(df, cdf, num):
    section_header("SECTION 7: Author & Contributor Patterns")
    gh = df[(df["has_github"]) & (df["total_commits"].notna()) & (df["total_commits"] > 0)].copy()

    yrs = sorted(gh["year"].unique())
    yr_labels = [str(y) for y in yrs]

    # 40 — Unique commit authors per repo
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med = gh.groupby("year")["num_commit_authors"].median()
    mean = gh.groupby("year")["num_commit_authors"].mean()
    ax.plot(yr_labels, [med.get(y, 0) for y in yrs], marker="o", label="Median", linewidth=2.5, color="#1b9e77")
    ax.plot(yr_labels, [mean.get(y, 0) for y in yrs], marker="s", label="Mean", linewidth=2, color="#d95f02")
    ax.set_title("Unique Commit Authors Per Repo Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Authors")
    ax.legend()
    save(fig, num, "commit_authors_per_repo"); num += 1

    # 41 — Solo-author repos fraction
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    solo_rate = []
    for y in yrs:
        sub = gh[gh["year"] == y]
        solo_rate.append((sub["num_commit_authors"] == 1).sum() / max(len(sub), 1) * 100)
    ax.plot(yr_labels, solo_rate, marker="o", linewidth=2.5, color="#e7298a")
    ax.fill_between(yr_labels, solo_rate, alpha=0.15, color="#e7298a")
    ax.set_title("Solo-Author Repos (Only 1 Committer) Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% of Repos")
    save(fig, num, "solo_author_repos"); num += 1

    # 42 — Top contributor dominance
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med_dom = gh[gh["top_contributor_pct"].notna()].groupby("year")["top_contributor_pct"].median()
    ax.plot(yr_labels, [med_dom.get(y, 0) * 100 for y in yrs], marker="o", linewidth=2.5, color="#7570b3")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% (equal split in 2-person team)")
    ax.set_title("Top Contributor's Share of Contributions (Median)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% of Total Contributions")
    ax.legend()
    save(fig, num, "top_contributor_dominance"); num += 1

    # 43 — Gini coefficient of commits per author
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    def gini(values):
        values = np.sort(np.array(values, dtype=float))
        n = len(values)
        if n == 0 or values.sum() == 0:
            return 0
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))

    gini_by_year = []
    for y in yrs:
        year_commits = cdf[cdf["year"] == y]
        repos = year_commits.groupby("repo_full_name")
        ginis = []
        for rname, rgroup in repos:
            author_counts = rgroup.groupby(
                rgroup["author_login"].fillna(rgroup["author_email"].fillna("unknown"))
            ).size().values
            if len(author_counts) > 1:
                ginis.append(gini(author_counts))
        gini_by_year.append(np.median(ginis) if ginis else 0)
    ax.plot(yr_labels, gini_by_year, marker="o", linewidth=2.5, color="#e6ab02")
    ax.set_title("Median Gini Coefficient of Commit Authorship by Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Gini Coefficient (0=equal, 1=monopoly)")
    ax.set_ylim(0, 1)
    save(fig, num, "commit_gini_coefficient"); num += 1

    # 44 — Contributors vs commit authors
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med_contrib = gh.groupby("year")["num_contributors"].median()
    med_authors = gh.groupby("year")["num_commit_authors"].median()
    ax.plot(yr_labels, [med_contrib.get(y, 0) for y in yrs], marker="o", label="GitHub Contributors", linewidth=2.5)
    ax.plot(yr_labels, [med_authors.get(y, 0) for y in yrs], marker="s", label="Distinct Commit Authors", linewidth=2.5)
    ax.set_title("GitHub Contributors vs Commit Authors (Median)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Count")
    ax.legend()
    save(fig, num, "contributors_vs_authors"); num += 1

    return num


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8: SCATTER PLOTS & CORRELATIONS
# ──────────────────────────────────────────────────────────────────────────────

def plot_scatters_correlations(df, num):
    section_header("SECTION 8: Scatter Plots & Correlations")
    gh = df[(df["has_github"]) & (df["total_commits"].notna()) & (df["total_commits"] > 0)].copy()
    gh["adds_per_commit"] = gh["total_additions"] / gh["total_commits"]
    gh["year_str"] = gh["year"].astype(str)

    # 45 — Scatter: total additions vs commits, colored by year
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    # Clip for readability
    plot_df = gh[gh["total_additions"].notna() & (gh["total_additions"] > 0)].copy()
    plot_df["total_additions_clip"] = plot_df["total_additions"].clip(upper=plot_df["total_additions"].quantile(0.95))
    plot_df["total_commits_clip"] = plot_df["total_commits"].clip(upper=plot_df["total_commits"].quantile(0.95))
    scatter = ax.scatter(
        plot_df["total_commits_clip"], plot_df["total_additions_clip"],
        c=plot_df["year"], cmap="viridis", alpha=0.4, s=15, edgecolors="none"
    )
    plt.colorbar(scatter, ax=ax, label="Year")
    ax.set_title("Total Additions vs. Commits (Color = Year)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Commits (clipped at 95th pctl)"); ax.set_ylabel("Total Additions (clipped)")
    save(fig, num, "scatter_additions_vs_commits"); num += 1

    # 46 — Scatter: additions per commit vs message length
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    plot_df2 = gh[gh["adds_per_commit"].notna()].copy()
    plot_df2["adds_per_commit_clip"] = plot_df2["adds_per_commit"].clip(upper=plot_df2["adds_per_commit"].quantile(0.95))
    # We need avg message length per project from commit data — use total_additions as proxy
    # Actually let's compute it
    ax.scatter(
        plot_df2["adds_per_commit_clip"], plot_df2["total_commits"],
        c=plot_df2["year"], cmap="plasma", alpha=0.4, s=15, edgecolors="none"
    )
    ax.set_title("Additions Per Commit vs Total Commits (Color = Year)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Additions / Commit (clipped)"); ax.set_ylabel("Total Commits")
    save(fig, num, "scatter_adds_per_commit_vs_total_commits"); num += 1

    # 47 — Correlation matrix of key numeric features
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_cols = ["total_commits", "total_additions", "total_deletions", "total_files_changed",
                 "num_contributors", "num_commit_authors", "num_languages", "total_files",
                 "size_kb", "stars", "forks", "adds_per_commit"]
    corr_data = gh[corr_cols].dropna()
    if len(corr_data) > 10:
        corr_matrix = corr_data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, ax=ax, square=True, linewidths=0.5)
        ax.set_title("Correlation Matrix of Key Project Metrics", fontsize=15, fontweight="bold")
    save(fig, num, "correlation_matrix"); num += 1

    # 48 — Scatter: repo size vs stars
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    s_data = gh[(gh["size_kb"].notna()) & (gh["stars"].notna())].copy()
    s_data["size_clip"] = s_data["size_kb"].clip(upper=s_data["size_kb"].quantile(0.95))
    s_data["stars_clip"] = s_data["stars"].clip(upper=s_data["stars"].quantile(0.95))
    ax.scatter(s_data["size_clip"], s_data["stars_clip"], c=s_data["year"], cmap="coolwarm",
               alpha=0.4, s=15, edgecolors="none")
    ax.set_title("Repo Size vs Stars (Color = Year)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Size KB (clipped)"); ax.set_ylabel("Stars (clipped)")
    save(fig, num, "scatter_size_vs_stars"); num += 1

    # 49 — Scatter: num_contributors vs total_commits
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.scatter(gh["num_contributors"].clip(upper=20), gh["total_commits"].clip(upper=gh["total_commits"].quantile(0.95)),
               c=gh["year"], cmap="viridis", alpha=0.4, s=15, edgecolors="none")
    ax.set_title("Contributors vs Commits (Color = Year)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Contributors (clipped at 20)"); ax.set_ylabel("Commits (clipped)")
    save(fig, num, "scatter_contributors_vs_commits"); num += 1

    return num


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9: COMPOSITE VIBE-CODING SCORES
# ──────────────────────────────────────────────────────────────────────────────

def plot_vibe_scores(df, cdf, num):
    section_header("SECTION 9: Composite Vibe-Coding Scores")
    gh = df[(df["has_github"]) & (df["total_commits"].notna()) & (df["total_commits"] > 0)].copy()
    gh["adds_per_commit"] = gh["total_additions"] / gh["total_commits"]

    yrs = sorted(gh["year"].unique())
    yr_labels = [str(y) for y in yrs]

    # Compute per-project avg message length from commit data
    avg_msg_len = cdf.groupby(["year", "repo_full_name"])["msg_first_line_len"].mean().reset_index()
    avg_msg_len.columns = ["year", "repo_full_name", "avg_msg_len"]
    gh = gh.merge(avg_msg_len, on=["year", "repo_full_name"], how="left")

    # 50 — Vibe score: composite metric
    # Higher = more vibe-coding-like
    # Components: (1) low commits, (2) high adds/commit, (3) short messages, (4) solo author
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    # Normalize components to 0-1
    def norm_01(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    gh_score = gh.dropna(subset=["adds_per_commit", "avg_msg_len", "total_commits"]).copy()
    if len(gh_score) > 10:
        gh_score["score_low_commits"] = 1 - norm_01(gh_score["total_commits"].clip(upper=200))
        gh_score["score_high_adds"] = norm_01(gh_score["adds_per_commit"].clip(upper=gh_score["adds_per_commit"].quantile(0.99)))
        gh_score["score_short_msgs"] = 1 - norm_01(gh_score["avg_msg_len"].clip(upper=200))
        gh_score["score_solo"] = (gh_score["num_commit_authors"] == 1).astype(float)

        gh_score["vibe_score"] = (
            gh_score["score_low_commits"] * 0.25 +
            gh_score["score_high_adds"] * 0.35 +
            gh_score["score_short_msgs"] * 0.20 +
            gh_score["score_solo"] * 0.20
        )

        med_vibe = gh_score.groupby("year")["vibe_score"].median()
        mean_vibe = gh_score.groupby("year")["vibe_score"].mean()
        p75_vibe = gh_score.groupby("year")["vibe_score"].quantile(0.75)
        ax.plot(yr_labels, [med_vibe.get(y, 0) for y in yrs], marker="o", label="Median", linewidth=2.5, color="#e41a1c")
        ax.plot(yr_labels, [mean_vibe.get(y, 0) for y in yrs], marker="s", label="Mean", linewidth=2, color="#377eb8")
        ax.plot(yr_labels, [p75_vibe.get(y, 0) for y in yrs], marker="^", label="75th Pctl", linewidth=2, color="#984ea3", linestyle="--")
        ax.set_title("Composite Vibe-Coding Score Over Time", fontsize=15, fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("Vibe Score (0=traditional, 1=vibe-coded)")
        ax.legend()
    save(fig, num, "vibe_coding_score_composite"); num += 1

    # 51 — Vibe score distribution by year (violin)
    if len(gh_score) > 10:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        vdata = [gh_score[gh_score["year"] == y]["vibe_score"].values for y in yrs]
        valid = [(y, v) for y, v in zip(yrs, vdata) if len(v) > 0]
        if valid:
            parts = ax.violinplot([v for _, v in valid], positions=range(len(valid)), showmedians=True)
            ax.set_xticks(range(len(valid)))
            ax.set_xticklabels([str(y) for y, _ in valid])
        ax.set_title("Vibe-Coding Score Distribution by Year", fontsize=15, fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("Vibe Score")
        save(fig, num, "vibe_score_violin"); num += 1
    else:
        num += 1

    # 52 — Scatter: vibe score vs stars
    if len(gh_score) > 10 and "stars" in gh_score.columns:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        s_sub = gh_score[gh_score["stars"].notna()].copy()
        s_sub["stars_clip"] = s_sub["stars"].clip(upper=s_sub["stars"].quantile(0.95))
        ax.scatter(s_sub["vibe_score"], s_sub["stars_clip"], c=s_sub["year"],
                   cmap="viridis", alpha=0.4, s=15, edgecolors="none")
        ax.set_title("Vibe Score vs Stars (Color = Year)", fontsize=15, fontweight="bold")
        ax.set_xlabel("Vibe Score"); ax.set_ylabel("Stars (clipped)")
        save(fig, num, "scatter_vibe_score_vs_stars"); num += 1
    else:
        num += 1

    # 53 — Percentage of high vibe-score projects by year
    if len(gh_score) > 10:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        thresholds = [0.5, 0.6, 0.7]
        for thresh in thresholds:
            rates = []
            for y in yrs:
                sub = gh_score[gh_score["year"] == y]
                rates.append((sub["vibe_score"] >= thresh).sum() / max(len(sub), 1) * 100)
            ax.plot(yr_labels, rates, marker="o", linewidth=2, label=f"Score ≥ {thresh}")
        ax.set_title("Fraction of Projects Above Vibe Score Thresholds", fontsize=15, fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("% of Projects")
        ax.legend()
        save(fig, num, "vibe_score_threshold_fractions"); num += 1
    else:
        num += 1

    return num


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10: OUTLIERS & EXTREMES
# ──────────────────────────────────────────────────────────────────────────────

def plot_outliers(df, cdf, num):
    section_header("SECTION 10: Outliers & Extremes")
    gh = df[(df["has_github"]) & (df["total_commits"].notna()) & (df["total_commits"] > 0)].copy()
    gh["adds_per_commit"] = gh["total_additions"] / gh["total_commits"]

    # 54 — Top 20 projects by additions per commit
    fig, ax = plt.subplots(figsize=(FIG_W, 9))
    top = gh.nlargest(20, "adds_per_commit")[["repo_full_name", "year", "adds_per_commit", "total_commits"]].reset_index(drop=True)
    labels = [f"{r['repo_full_name']} ({r['year']}, {int(r['total_commits'])} commits)" for _, r in top.iterrows()]
    ax.barh(range(len(top)), top["adds_per_commit"].values, color=sns.color_palette("Reds_r", len(top)))
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_title("Top 20 Projects by Lines Added Per Commit", fontsize=15, fontweight="bold")
    ax.set_xlabel("Additions / Commit")
    save(fig, num, "top20_additions_per_commit"); num += 1

    # 55 — Top 20 projects by total additions (largest codebases)
    fig, ax = plt.subplots(figsize=(FIG_W, 9))
    top = gh.nlargest(20, "total_additions")[["repo_full_name", "year", "total_additions", "total_commits"]].reset_index(drop=True)
    labels = [f"{r['repo_full_name']} ({r['year']}, {int(r['total_commits'])} commits)" for _, r in top.iterrows()]
    ax.barh(range(len(top)), top["total_additions"].values, color=sns.color_palette("Blues_r", len(top)))
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_title("Top 20 Projects by Total Lines Added", fontsize=15, fontweight="bold")
    ax.set_xlabel("Total Additions")
    save(fig, num, "top20_total_additions"); num += 1

    # 56 — Projects with single commit but high additions
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    single = gh[gh["total_commits"] == 1].copy()
    if len(single) > 0:
        single_by_year = single.groupby("year").agg(
            count=("total_additions", "size"),
            median_additions=("total_additions", "median"),
            mean_additions=("total_additions", "mean"),
        )
        yrs_s = sorted(single_by_year.index)
        ax.bar([str(y) for y in yrs_s],
               [single_by_year.loc[y, "median_additions"] for y in yrs_s],
               alpha=0.6, label="Median additions", color="#fc8d62")
        ax.plot([str(y) for y in yrs_s],
                [single_by_year.loc[y, "mean_additions"] for y in yrs_s],
                marker="s", label="Mean additions", linewidth=2, color="#e41a1c")
        ax2 = ax.twinx()
        ax2.plot([str(y) for y in yrs_s],
                 [single_by_year.loc[y, "count"] for y in yrs_s],
                 marker="^", label="Count", linewidth=2, color="#66c2a5", linestyle="--")
        ax2.set_ylabel("Count of single-commit repos")
        ax.set_title("Single-Commit Repos: Size & Count Over Time", fontsize=15, fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("Lines Added")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    save(fig, num, "single_commit_repos_size_trend"); num += 1

    # 57 — Projects with extreme commit counts (highest)
    fig, ax = plt.subplots(figsize=(FIG_W, 9))
    top = gh.nlargest(20, "total_commits")[["repo_full_name", "year", "total_commits", "total_additions"]].reset_index(drop=True)
    labels = [f"{r['repo_full_name']} ({r['year']})" for _, r in top.iterrows()]
    ax.barh(range(len(top)), top["total_commits"].values, color=sns.color_palette("Greens_r", len(top)))
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_title("Top 20 Projects by Commit Count", fontsize=15, fontweight="bold")
    ax.set_xlabel("Commits")
    save(fig, num, "top20_most_commits"); num += 1

    return num


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11: TIME-WITHIN-HACKATHON ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def plot_time_analysis(cdf, num):
    section_header("SECTION 11: Time-Within-Hackathon Analysis")

    yrs = sorted(cdf["year"].unique())
    yr_labels = [str(y) for y in yrs]

    # Parse commit dates
    cdf_t = cdf.copy()
    cdf_t["date_parsed"] = pd.to_datetime(cdf_t["date"], errors="coerce", utc=True)
    cdf_t = cdf_t.dropna(subset=["date_parsed"])
    cdf_t["hour"] = cdf_t["date_parsed"].dt.hour

    # 58 — Commit hour distribution (all years combined)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.hist(cdf_t["hour"], bins=24, range=(0, 24), color="#7fcdbb", edgecolor="white")
    ax.set_title("Commit Hour Distribution (All Years, UTC)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Number of Commits")
    ax.set_xticks(range(0, 24))
    save(fig, num, "commit_hour_distribution_all"); num += 1

    # 59 — Commit hour heatmap by year
    fig, ax = plt.subplots(figsize=(14, 8))
    hour_matrix = []
    valid_yrs = []
    for y in yrs:
        sub = cdf_t[cdf_t["year"] == y]
        if len(sub) > 0:
            counts, _ = np.histogram(sub["hour"], bins=24, range=(0, 24))
            hour_matrix.append(counts / counts.sum() * 100)
            valid_yrs.append(y)
    if hour_matrix:
        hm_df = pd.DataFrame(hour_matrix, index=[str(y) for y in valid_yrs], columns=[str(h) for h in range(24)])
        sns.heatmap(hm_df, cmap="YlOrRd", annot=True, fmt=".1f", ax=ax, linewidths=0.5)
        ax.set_title("Commit Hour Distribution by Year (% of Commits, UTC)", fontsize=15, fontweight="bold")
        ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Year")
    save(fig, num, "commit_hour_heatmap_by_year"); num += 1

    # 60 — Commit cadence: time between consecutive commits (per repo median)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    cadences = []
    for y in yrs:
        sub = cdf_t[cdf_t["year"] == y].sort_values(["repo_full_name", "date_parsed"])
        repo_cadences = []
        for rname, rgroup in sub.groupby("repo_full_name"):
            if len(rgroup) < 2:
                continue
            diffs = rgroup["date_parsed"].diff().dt.total_seconds() / 60  # minutes
            median_cadence = diffs.dropna().median()
            if pd.notna(median_cadence) and median_cadence > 0:
                repo_cadences.append(median_cadence)
        cadences.append(np.median(repo_cadences) if repo_cadences else 0)
    ax.plot(yr_labels, cadences, marker="o", linewidth=2.5, color="#756bb1")
    ax.set_title("Median Time Between Commits (Minutes) by Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Minutes Between Commits (median of medians)")
    save(fig, num, "commit_cadence_minutes"); num += 1

    # 61 — Additions over time within hackathon (normalized timeline)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    sample_years = [y for y in [2018, 2020, 2022, 2024, 2025, 2026] if y in cdf_t["year"].unique()]
    for idx, y in enumerate(sample_years[:6]):
        ax = axes[idx]
        sub = cdf_t[cdf_t["year"] == y].sort_values(["repo_full_name", "date_parsed"])
        # For each repo, normalize timestamp to 0-1 within its own range
        for rname, rgroup in sub.groupby("repo_full_name"):
            if len(rgroup) < 2:
                continue
            t_min = rgroup["date_parsed"].min()
            t_max = rgroup["date_parsed"].max()
            span = (t_max - t_min).total_seconds()
            if span <= 0:
                continue
            t_norm = (rgroup["date_parsed"] - t_min).dt.total_seconds() / span
            cumulative = rgroup["additions"].cumsum()
            total = cumulative.iloc[-1]
            if total > 0:
                ax.plot(t_norm.values, (cumulative / total).values, alpha=0.05, color="#e41a1c", linewidth=0.5)
        ax.set_title(f"{y}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Normalized Time (0=first, 1=last commit)")
        ax.set_ylabel("Cumulative Additions (fraction)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.suptitle("Cumulative Code Addition Curves (Each Line = One Repo)", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, num, "cumulative_addition_curves"); num += 1

    return num


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12: PER-COMMIT DEEP DIVES
# ──────────────────────────────────────────────────────────────────────────────

def plot_commit_deep_dives(cdf, num):
    section_header("SECTION 12: Per-Commit Deep Dives")

    yrs = sorted(cdf["year"].unique())
    yr_labels = [str(y) for y in yrs]

    # 62 — Additions histogram per commit (log scale)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    pos_adds = cdf[cdf["additions"] > 0]["additions"]
    ax.hist(np.log10(pos_adds.clip(lower=1)), bins=60, color="#7fc97f", edgecolor="white", alpha=0.8)
    ax.set_title("Distribution of Lines Added Per Commit (Log10 Scale, All Years)", fontsize=15, fontweight="bold")
    ax.set_xlabel("log10(Additions)"); ax.set_ylabel("Count")
    save(fig, num, "additions_per_commit_log_hist"); num += 1

    # 63 — Large commits (>500 additions) fraction by year
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    thresholds = [500, 1000, 5000]
    for t in thresholds:
        rates = []
        for y in yrs:
            sub = cdf[cdf["year"] == y]
            rates.append((sub["additions"] >= t).sum() / max(len(sub), 1) * 100)
        ax.plot(yr_labels, rates, marker="o", linewidth=2, label=f"≥{t} additions")
    ax.set_title("Fraction of Large Commits Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% of Commits")
    ax.legend()
    save(fig, num, "large_commits_fraction"); num += 1

    # 64 — Zero-addition commits (e.g., merge commits, deletes only)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    zero_rates = []
    for y in yrs:
        sub = cdf[cdf["year"] == y]
        zero_rates.append((sub["additions"] == 0).sum() / max(len(sub), 1) * 100)
    ax.bar(yr_labels, zero_rates, color="#bc80bd")
    ax.set_title("Zero-Addition Commits (Merges/Deletes Only) by Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% of Commits")
    save(fig, num, "zero_addition_commits"); num += 1

    # 65 — Commit message length vs additions scatter (sampled)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sample = cdf[(cdf["additions"] > 0) & (cdf["msg_first_line_len"] > 0)].copy()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)
    sample["log_additions"] = np.log10(sample["additions"].clip(lower=1))
    ax.scatter(sample["msg_first_line_len"].clip(upper=120), sample["log_additions"],
               c=sample["year"], cmap="viridis", alpha=0.3, s=8, edgecolors="none")
    plt.colorbar(ax.collections[0], ax=ax, label="Year")
    ax.set_title("Commit Message Length vs Additions (Log Scale)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Message First Line Length (chars)"); ax.set_ylabel("log10(Additions)")
    save(fig, num, "scatter_msg_len_vs_additions"); num += 1

    # 66 — Ratio of additions to files touched per commit
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    cdf_nf = cdf[(cdf["num_files"] > 0) & (cdf["additions"] > 0)].copy()
    cdf_nf["adds_per_file"] = cdf_nf["additions"] / cdf_nf["num_files"]
    med_apf = cdf_nf.groupby("year")["adds_per_file"].median()
    ax.plot(yr_labels, [med_apf.get(y, 0) for y in yrs], marker="o", linewidth=2.5, color="#e6550d")
    ax.set_title("Median Lines Added Per File Touched (Per Commit)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Additions / File")
    save(fig, num, "additions_per_file_per_commit"); num += 1

    return num


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 13: YEAR-OVER-YEAR CHANGE RATES
# ──────────────────────────────────────────────────────────────────────────────

def plot_yoy_changes(df, cdf, num):
    section_header("SECTION 13: Year-over-Year Change Rates")
    gh = df[(df["has_github"]) & (df["total_commits"].notna()) & (df["total_commits"] > 0)].copy()
    gh["adds_per_commit"] = gh["total_additions"] / gh["total_commits"]

    yrs = sorted(gh["year"].unique())
    yr_labels = [str(y) for y in yrs]

    # Compute medians of key metrics per year
    metrics = {
        "Median Commits": gh.groupby("year")["total_commits"].median(),
        "Median Adds/Commit": gh.groupby("year")["adds_per_commit"].median(),
        "Median Contributors": gh.groupby("year")["num_contributors"].median(),
        "Median Files": gh.groupby("year")["total_files"].median(),
        "Median Size KB": gh.groupby("year")["size_kb"].median(),
    }

    # 67 — Indexed trend (2015=100)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for mname, mdata in metrics.items():
        vals = [mdata.get(y, np.nan) for y in yrs]
        base = next((v for v in vals if pd.notna(v) and v > 0), 1)
        indexed = [v / base * 100 if pd.notna(v) else np.nan for v in vals]
        ax.plot(yr_labels, indexed, marker="o", linewidth=2, label=mname)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Key Metrics Indexed to Earliest Year (=100)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Index (earliest year = 100)")
    ax.legend(fontsize=8)
    save(fig, num, "indexed_trend_all_metrics"); num += 1

    # 68 — YoY % change in median commits
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med_commits = [metrics["Median Commits"].get(y, np.nan) for y in yrs]
    yoy = [np.nan] + [(med_commits[i] - med_commits[i-1]) / max(med_commits[i-1], 0.01) * 100
                       if pd.notna(med_commits[i]) and pd.notna(med_commits[i-1]) else np.nan
                       for i in range(1, len(med_commits))]
    colors = ["#2ca02c" if (pd.notna(v) and v >= 0) else "#d62728" for v in yoy]
    ax.bar(yr_labels, [v if pd.notna(v) else 0 for v in yoy], color=colors)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("Year-over-Year Change in Median Commits (%)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% Change")
    save(fig, num, "yoy_change_median_commits"); num += 1

    # 69 — YoY % change in median additions per commit
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med_apc = [metrics["Median Adds/Commit"].get(y, np.nan) for y in yrs]
    yoy_apc = [np.nan] + [(med_apc[i] - med_apc[i-1]) / max(med_apc[i-1], 0.01) * 100
                           if pd.notna(med_apc[i]) and pd.notna(med_apc[i-1]) else np.nan
                           for i in range(1, len(med_apc))]
    colors = ["#d62728" if (pd.notna(v) and v >= 0) else "#2ca02c" for v in yoy_apc]  # reversed: rising is concerning
    ax.bar(yr_labels, [v if pd.notna(v) else 0 for v in yoy_apc], color=colors)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("Year-over-Year Change in Median Additions/Commit (%)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% Change")
    save(fig, num, "yoy_change_additions_per_commit"); num += 1

    return num


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 14: ADDITIONAL DEEP ANALYSES
# ──────────────────────────────────────────────────────────────────────────────

def plot_additional(df, cdf, num):
    section_header("SECTION 14: Additional Analyses")
    gh = df[(df["has_github"]) & (df["total_commits"].notna()) & (df["total_commits"] > 0)].copy()

    yrs = sorted(gh["year"].unique())
    yr_labels = [str(y) for y in yrs]

    # 70 — Wiki/Pages/Discussions features enabled rate
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for feat, color in [("has_wiki", "#1b9e77"), ("has_pages", "#d95f02"), ("has_discussions", "#7570b3")]:
        sub = gh[gh[feat].notna()]
        rate = sub.groupby("year")[feat].mean() * 100
        ax.plot(yr_labels, [rate.get(y, 0) for y in yrs], marker="o", label=feat.replace("has_", "").title(), linewidth=2, color=color)
    ax.set_title("GitHub Features Enabled Rate Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% of Repos")
    ax.legend()
    save(fig, num, "github_features_enabled"); num += 1

    # 71 — Archived repos by year
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    arch = gh[gh["archived"].notna()].groupby("year")["archived"].mean() * 100
    ax.bar(yr_labels, [arch.get(y, 0) for y in yrs], color="#fc8d59")
    ax.set_title("Archived Repos by Year (%)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% Archived")
    save(fig, num, "archived_repos_rate"); num += 1

    # 72 — Open issues trend
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    med_issues = gh.groupby("year")["open_issues"].median()
    mean_issues = gh.groupby("year")["open_issues"].mean()
    ax.bar(yr_labels, [med_issues.get(y, 0) for y in yrs], alpha=0.6, label="Median", color="#b3de69")
    ax.plot(yr_labels, [mean_issues.get(y, 0) for y in yrs], marker="s", label="Mean", color="#e41a1c", linewidth=2)
    ax.set_title("Open Issues Per Repo Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Open Issues")
    ax.legend()
    save(fig, num, "open_issues_trend"); num += 1

    # 73 — Commit message contains common AI-related terms
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ai_patterns = {
        "AI/ML terms": r"\b(ai|ml|gpt|llm|openai|claude|gemini|copilot|chatgpt|neural|transformer|model|inference)\b",
        "Auto-generated": r"\b(auto|generated|scaffold|boilerplate|template|starter)\b",
        "Initial/Setup": r"\b(initial|setup|init|bootstrap|create|first)\b",
    }
    for pname, patt in ai_patterns.items():
        rates = []
        for y in yrs:
            sub = cdf[cdf["year"] == y]
            count = sub["message"].str.lower().str.contains(patt, regex=True, na=False).sum()
            rates.append(count / max(len(sub), 1) * 100)
        ax.plot(yr_labels, rates, marker="o", linewidth=2, label=pname)
    ax.set_title("AI & Setup-Related Terms in Commit Messages", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% of Commits")
    ax.legend()
    save(fig, num, "ai_terms_in_commits"); num += 1

    # 74 — Repo has README proxy: .md file percentage
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    md_rates = []
    for y in yrs:
        sub = gh[gh["year"] == y]
        has_md = sum(1 for _, r in sub.iterrows()
                    if isinstance(r.get("by_extension"), dict) and ".md" in r.get("by_extension", {}))
        md_rates.append(has_md / max(len(sub), 1) * 100)
    ax.plot(yr_labels, md_rates, marker="o", linewidth=2.5, color="#41ab5d")
    ax.set_title("Repos with Markdown (.md) Files Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% of Repos")
    save(fig, num, "markdown_file_presence"); num += 1

    # 75 — Average file size (bytes per file) trend
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    gh["avg_file_size"] = gh["total_file_size_bytes"] / gh["total_files"].replace(0, np.nan)
    med_afs = gh.groupby("year")["avg_file_size"].median()
    ax.bar(yr_labels, [med_afs.get(y, 0) for y in yrs], color="#8c96c6")
    ax.set_title("Median Average File Size (Bytes) by Year", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Bytes / File")
    save(fig, num, "avg_file_size_bytes"); num += 1

    # 76 — Number of top-level dirs vs total files scatter
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    gh_td = gh[gh["by_top_level_dir"].apply(lambda x: isinstance(x, dict))].copy()
    gh_td["num_dirs"] = gh_td["by_top_level_dir"].apply(len)
    sample_td = gh_td.sample(min(3000, len(gh_td)), random_state=42) if len(gh_td) > 3000 else gh_td
    ax.scatter(sample_td["num_dirs"], sample_td["total_files"].clip(upper=sample_td["total_files"].quantile(0.95)),
               c=sample_td["year"], cmap="plasma", alpha=0.4, s=12, edgecolors="none")
    plt.colorbar(ax.collections[0], ax=ax, label="Year")
    ax.set_title("Top-Level Directories vs Total Files", fontsize=15, fontweight="bold")
    ax.set_xlabel("Number of Top-Level Directories"); ax.set_ylabel("Total Files (clipped)")
    save(fig, num, "scatter_dirs_vs_files"); num += 1

    return num


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 15: SUMMARY DASHBOARDS
# ──────────────────────────────────────────────────────────────────────────────

def plot_summary_dashboards(df, cdf, num):
    section_header("SECTION 15: Summary Dashboards")
    gh = df[(df["has_github"]) & (df["total_commits"].notna()) & (df["total_commits"] > 0)].copy()
    gh["adds_per_commit"] = gh["total_additions"] / gh["total_commits"]

    yrs = sorted(gh["year"].unique())
    yr_labels = [str(y) for y in yrs]

    # 77 — Multi-panel vibe-coding dashboard
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Panel 1: Median commits
    ax = axes[0, 0]
    med = gh.groupby("year")["total_commits"].median()
    ax.plot(yr_labels, [med.get(y, 0) for y in yrs], marker="o", color="#e41a1c", linewidth=2)
    ax.set_title("Median Commits/Project", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year"); ax.tick_params(axis='x', rotation=45)

    # Panel 2: Median adds/commit
    ax = axes[0, 1]
    mapc = gh.groupby("year")["adds_per_commit"].median()
    ax.plot(yr_labels, [mapc.get(y, 0) for y in yrs], marker="o", color="#ff7f00", linewidth=2)
    ax.set_title("Median Additions/Commit", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year"); ax.tick_params(axis='x', rotation=45)

    # Panel 3: Single-commit repo %
    ax = axes[0, 2]
    sr = [(gh[gh["year"] == y]["total_commits"] == 1).sum() / max(len(gh[gh["year"] == y]), 1) * 100 for y in yrs]
    ax.plot(yr_labels, sr, marker="o", color="#984ea3", linewidth=2)
    ax.set_title("Single-Commit Repos %", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year"); ax.tick_params(axis='x', rotation=45)

    # Panel 4: Solo author %
    ax = axes[1, 0]
    solo = [(gh[gh["year"] == y]["num_commit_authors"] == 1).sum() / max(len(gh[gh["year"] == y]), 1) * 100 for y in yrs]
    ax.plot(yr_labels, solo, marker="o", color="#e7298a", linewidth=2)
    ax.set_title("Solo-Author Repos %", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year"); ax.tick_params(axis='x', rotation=45)

    # Panel 5: Median msg first line len
    ax = axes[1, 1]
    mfl = cdf.groupby("year")["msg_first_line_len"].median()
    ax.plot(yr_labels, [mfl.get(y, 0) for y in yrs], marker="o", color="#1b9e77", linewidth=2)
    ax.set_title("Median Commit Msg Length", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year"); ax.tick_params(axis='x', rotation=45)

    # Panel 6: Large commit %
    ax = axes[1, 2]
    lc = [(cdf[cdf["year"] == y]["additions"] >= 1000).sum() / max(len(cdf[cdf["year"] == y]), 1) * 100 for y in yrs]
    ax.plot(yr_labels, lc, marker="o", color="#a65628", linewidth=2)
    ax.set_title("Commits ≥1000 Lines %", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year"); ax.tick_params(axis='x', rotation=45)

    plt.suptitle("Vibe-Coding Indicators Dashboard — Key Trends at a Glance",
                 fontsize=16, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, num, "vibe_coding_dashboard"); num += 1

    # 78 — Before vs After ChatGPT (pre-2023 vs 2023+)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    pre = gh[gh["year"] < 2023].copy()
    post = gh[gh["year"] >= 2023].copy()
    metrics_compare = {
        "Median Commits": (pre["total_commits"].median(), post["total_commits"].median()),
        "Median Adds/Commit": (pre["adds_per_commit"].median(), post["adds_per_commit"].median()),
        "Median Contributors": (pre["num_contributors"].median(), post["num_contributors"].median()),
        "Solo Author %": (
            (pre["num_commit_authors"] == 1).mean() * 100,
            (post["num_commit_authors"] == 1).mean() * 100
        ),
        "Median Files": (pre["total_files"].median(), post["total_files"].median()),
        "Median Languages": (pre["num_languages"].median(), post["num_languages"].median()),
    }
    names = list(metrics_compare.keys())
    pre_vals = [metrics_compare[m][0] for m in names]
    post_vals = [metrics_compare[m][1] for m in names]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, pre_vals, w, label="Pre-ChatGPT (2015-2022)", color="#4daf4a")
    ax.bar(x + w/2, post_vals, w, label="Post-ChatGPT (2023-2026)", color="#e41a1c")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_title("Pre vs Post ChatGPT Era: Key Metrics", fontsize=15, fontweight="bold")
    ax.legend()
    save(fig, num, "pre_vs_post_chatgpt"); num += 1

    # 79 — Percentage change from pre to post
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    pct_changes = []
    for m in names:
        pre_v, post_v = metrics_compare[m]
        if pre_v and pre_v != 0:
            pct_changes.append((post_v - pre_v) / abs(pre_v) * 100)
        else:
            pct_changes.append(0)
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in pct_changes]
    ax.barh(names, pct_changes, color=colors)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_title("% Change: Post-ChatGPT vs Pre-ChatGPT Era", fontsize=15, fontweight="bold")
    ax.set_xlabel("% Change")
    save(fig, num, "pct_change_pre_vs_post"); num += 1

    # 80 — Median commit size over normalized project timeline, by year ───────
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    cdf_t = cdf.dropna(subset=["date"]).copy()
    cdf_t["dt"] = pd.to_datetime(cdf_t["date"], errors="coerce", utc=True)
    cdf_t = cdf_t.dropna(subset=["dt"])
    cdf_t["commit_lines"] = cdf_t["additions"] + cdf_t["deletions"]

    n_bins = 20
    min_samples_per_bin = 5  # drop bins with fewer commits to avoid outlier spikes
    # Hand-picked palette with maximum perceptual distance for 12 years
    year_colors = [
        "#1f77b4",  # blue
        "#e41a1c",  # red
        "#4daf4a",  # green
        "#984ea3",  # purple
        "#ff7f00",  # orange
        "#a65628",  # brown
        "#f781bf",  # pink
        "#17becf",  # cyan
        "#bcbd22",  # olive
        "#7f7f7f",  # grey
        "#377eb8",  # another blue
        "#000000",  # black
    ]
    # Vary line styles too so overlapping colors are still distinguishable
    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]
    yr_list = sorted(cdf_t["year"].unique())

    for idx, yr in enumerate(yr_list):
        yr_commits = cdf_t[cdf_t["year"] == yr].copy()
        # Normalize each project's commits to [0, 1] based on time span
        normalised = []
        for (pi, repo), grp in yr_commits.groupby(["project_idx", "repo_full_name"]):
            if len(grp) < 2:
                continue
            ts = grp["dt"]
            t_min, t_max = ts.min(), ts.max()
            span = (t_max - t_min).total_seconds()
            if span <= 0:
                continue
            grp = grp.copy()
            grp["t_norm"] = grp["dt"].apply(lambda d: (d - t_min).total_seconds() / span)
            normalised.append(grp[["t_norm", "commit_lines"]])
        if not normalised:
            continue
        all_norm = pd.concat(normalised, ignore_index=True)
        # Bin into n_bins buckets across [0, 1]
        all_norm["bin"] = pd.cut(all_norm["t_norm"], bins=n_bins, labels=False)
        bin_stats = all_norm.groupby("bin")["commit_lines"].agg(["median", "count"])
        # Keep only bins with enough samples
        bin_stats = bin_stats[bin_stats["count"] >= min_samples_per_bin]
        if bin_stats.empty:
            continue
        bin_centers = [(i + 0.5) / n_bins for i in bin_stats.index]
        ax.plot(bin_centers, bin_stats["median"].values, label=str(yr),
                color=year_colors[idx % len(year_colors)],
                linestyle=line_styles[idx % len(line_styles)],
                linewidth=2.0, alpha=0.9)

    ax.set_title("Median Commit Size Over Normalized Project Timeline", fontsize=15, fontweight="bold")
    ax.set_xlabel("Project Timeline (0 = first commit, 1 = last commit)")
    ax.set_ylabel("Median Commit Size (lines added + deleted)")
    ax.legend(title="Year", fontsize=8, title_fontsize=9)
    ax.set_xlim(0, 1)
    save(fig, num, "commit_size_over_project_timeline"); num += 1

    return num


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_data()

    print("Building project-level rows …")
    proj_rows = build_project_rows(data)
    df = pd.DataFrame(proj_rows)
    print(f"  {len(df)} project rows across {df['year'].nunique()} years")

    print("Building commit-level rows …")
    commit_rows = build_commit_rows(data)
    cdf = pd.DataFrame(commit_rows)
    print(f"  {len(cdf)} commit rows")

    # Free memory from raw data
    del data

    print(f"\nGenerating graphs into {OUTPUT_DIR}/ …\n")

    num = 1
    num = plot_overview(df, num)
    num = plot_repo_basics(df, num)
    num = plot_language_trends(df, num)
    num = plot_codebase_composition(df, num)
    num = plot_commit_behavior(df, cdf, num)
    num = plot_commit_messages(cdf, num)
    num = plot_author_patterns(df, cdf, num)
    num = plot_scatters_correlations(df, num)
    num = plot_vibe_scores(df, cdf, num)
    num = plot_outliers(df, cdf, num)
    num = plot_time_analysis(cdf, num)
    num = plot_commit_deep_dives(cdf, num)
    num = plot_yoy_changes(df, cdf, num)
    num = plot_additional(df, cdf, num)
    num = plot_summary_dashboards(df, cdf, num)

    print(f"\n{'='*60}")
    print(f"  DONE — {num - 1} graphs saved to {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
