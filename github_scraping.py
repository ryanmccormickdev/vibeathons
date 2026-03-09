#!/usr/bin/env python3
"""
Enrich a TreeHacks-style projects JSON with GitHub metadata.

Configuration is at the top of the file.

Usage (assuming Python 3.9+):

    export TREEHACKS_SCRAPER_TOKEN="ghp_your_token_here"
    python enrich_github_metadata.py

Input JSON format (simplified):

{
  "treehacks_2026": [
    {
      "project_url": "...",
      "project_title": "Phone AI",
      "primary_github": "https://github.com/kanakapalli/phone_ai",
      "all_github": ["https://github.com/kanakapalli/phone_ai"]
    },
    ...
  ],
  "treehacks_2025": [...],
  ...
}
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

# =====================
# Configuration
# =====================

INPUT_JSON_PATH = "treehacks_projects.json"
OUTPUT_JSON_PATH = "treehacks_projects_enriched.json"

# Cache of per-repo metadata. Prevents re-hitting GitHub for the same repo.
REPO_CACHE_PATH = "github_repo_cache.json"

# How many commits per repo to fetch from GitHub. Adjust if you want more/less detail.
MAX_COMMITS_PER_REPO = 100

# How often to persist the repo cache to disk (every N newly-fetched repos)
SAVE_CACHE_EVERY_N_REPOS = 5

# GitHub API settings
GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"

# Network / rate-limit behavior
REQUEST_TIMEOUT_SECONDS = 30
RATE_LIMIT_FALLBACK_SLEEP_SECONDS = 60
RATE_LIMIT_RESET_BUFFER_SECONDS = 1


@dataclass
class GitHubConfig:
    token: str
    base_url: str = GITHUB_API_BASE_URL
    api_version: str = GITHUB_API_VERSION


# =====================
# Utility functions
# =====================

def load_github_token() -> str:
    token = os.environ.get("TREEHACKS_SCRAPER_TOKEN")
    if not token:
        print("ERROR: TREEHACKS_SCRAPER_TOKEN environment variable is not set.", file=sys.stderr)
        print("       Please export TREEHACKS_SCRAPER_TOKEN with your personal access token.", file=sys.stderr)
        sys.exit(1)
    return token


def parse_github_repo(url: str) -> Optional[Tuple[str, str]]:
    """
    Extract (owner, repo) from a GitHub URL.
    Handles variants like:
      - https://github.com/owner/repo
      - https://github.com/owner/repo.git
      - https://github.com/owner/repo/
      - https://github.com/owner/repo/tree/main
    Returns None if it does not look like a GitHub repo URL.
    """
    if not url or "github.com" not in url:
        return None

    # Strip protocol
    match = re.search(r"github\.com[:/]+([^/]+)/([^/#?]+)", url)
    if not match:
        return None

    owner = match.group(1)
    repo = match.group(2)

    # Strip trailing ".git" if present
    if repo.endswith(".git"):
        repo = repo[:-4]

    return owner, repo


def safe_get(d: Dict, *keys, default=None):
    """Nested dict get: safe_get(d, 'a', 'b', default=None) == d.get('a', {}).get('b', default)."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# =====================
# GitHub API helpers
# =====================

def github_request(
    cfg: GitHubConfig,
    path: str,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Dict[str, str]]:
    """
    Perform a single GitHub API GET request and return (json, headers).
    Raises RuntimeError with useful info on rate limiting or errors.
    """
    url = cfg.base_url.rstrip("/") + path
    headers = {
        "Authorization": f"Bearer {cfg.token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": cfg.api_version,
        "User-Agent": "treehacks-metadata-scraper/1.0",
    }

    while True:
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            raise RuntimeError(f"Network error calling GitHub: {e}") from e

        # Rate limit handling: wait and retry automatically.
        is_rate_limited = resp.status_code == 429 or (
            resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0"
        )
        if is_rate_limited:
            reset = resp.headers.get("X-RateLimit-Reset")
            wait_seconds = RATE_LIMIT_FALLBACK_SLEEP_SECONDS

            if reset:
                try:
                    reset_ts = int(reset)
                    wait_seconds = max(reset_ts - int(time.time()), 0) + RATE_LIMIT_RESET_BUFFER_SECONDS
                except ValueError:
                    pass

            if reset:
                print(
                    f"  !! GitHub API rate limit exceeded (reset at UNIX timestamp {reset}). "
                    f"Waiting {wait_seconds}s before retrying..."
                )
            else:
                print(
                    f"  !! GitHub API rate limit exceeded (no reset timestamp provided). "
                    f"Waiting {wait_seconds}s before retrying..."
                )

            time.sleep(wait_seconds)
            continue

        if resp.status_code == 404:
            # Let caller treat 404 specially if needed
            return None, resp.headers

        if resp.status_code >= 400:
            raise RuntimeError(f"GitHub API returned HTTP {resp.status_code}: {resp.text}")

        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(f"Failed to parse GitHub JSON response: {e}") from e

        return data, resp.headers


def github_get_paginated(
    cfg: GitHubConfig,
    path: str,
    per_page: int = 100,
    max_items: Optional[int] = None,
    extra_params: Optional[Dict[str, Any]] = None
) -> List[Any]:
    """
    Simple pagination helper for endpoints that return a JSON list.
    Uses `page` query parameter until empty or max_items is reached.
    """
    items: List[Any] = []
    page = 1

    while True:
        params = dict(extra_params or {})
        params.update({"per_page": per_page, "page": page})

        data, _ = github_request(cfg, path, params=params)
        if data is None:
            break
        if not isinstance(data, list):
            # Not a list endpoint; return as single element list
            return [data]

        if not data:
            break

        items.extend(data)
        if max_items is not None and len(items) >= max_items:
            break

        page += 1

    if max_items is not None:
        return items[:max_items]
    return items


# =====================
# Per-repo fetchers
# =====================

def fetch_basic_metadata(cfg: GitHubConfig, owner: str, repo: str) -> Optional[Dict[str, Any]]:
    data, headers = github_request(cfg, f"/repos/{owner}/{repo}")
    if data is None:
        return None

    basic = {
        "id": data.get("id"),
        "full_name": data.get("full_name"),
        "description": data.get("description"),
        "fork": data.get("fork"),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "pushed_at": data.get("pushed_at"),
        "homepage": data.get("homepage"),
        "stargazers_count": data.get("stargazers_count"),
        "watchers_count": data.get("watchers_count"),
        "forks_count": data.get("forks_count"),
        "open_issues_count": data.get("open_issues_count"),
        "size_kb": data.get("size"),
        "default_branch": data.get("default_branch"),
        "license": {
            "spdx_id": safe_get(data, "license", "spdx_id"),
            "name": safe_get(data, "license", "name"),
        },
        "archived": data.get("archived"),
        "disabled": data.get("disabled"),
        "visibility": data.get("visibility"),
        "primary_language": data.get("language"),
    }

    misc = {
        "has_issues": data.get("has_issues"),
        "has_projects": data.get("has_projects"),
        "has_downloads": data.get("has_downloads"),
        "has_wiki": data.get("has_wiki"),
        "has_pages": data.get("has_pages"),
        "has_discussions": data.get("has_discussions"),
        "is_template": data.get("is_template"),
    }

    return {"basic": basic, "misc": misc}


def fetch_languages(cfg: GitHubConfig, owner: str, repo: str) -> Dict[str, int]:
    data, _ = github_request(cfg, f"/repos/{owner}/{repo}/languages")
    if data is None:
        return {}
    # data is { "Python": 12345, ... }
    return data


def fetch_contributors(cfg: GitHubConfig, owner: str, repo: str) -> List[Dict[str, Any]]:
    contributors = github_get_paginated(
        cfg,
        f"/repos/{owner}/{repo}/contributors",
        per_page=100,
        max_items=None,
        extra_params=None,
    )
    result = []
    for c in contributors:
        result.append(
            {
                "login": c.get("login"),
                "id": c.get("id"),
                "contributions": c.get("contributions"),
            }
        )
    return result


def fetch_branch_head_sha(cfg: GitHubConfig, owner: str, repo: str, branch: str) -> Optional[str]:
    data, _ = github_request(cfg, f"/repos/{owner}/{repo}/branches/{branch}")
    if data is None:
        return None
    return safe_get(data, "commit", "sha")


def fetch_file_tree(
    cfg: GitHubConfig,
    owner: str,
    repo: str,
    head_sha: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Use the Git Tree API to get the file tree at a given commit SHA.
    Returns (files_summary, files_sample).
    """
    data, _ = github_request(cfg, f"/repos/{owner}/{repo}/git/trees/{head_sha}", params={"recursive": "1"})
    tree = data.get("tree", [])
    total_files = 0
    total_size = 0
    by_extension: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0, "size_bytes": 0})
    by_top_level: Dict[str, int] = defaultdict(int)

    files_sample: List[Dict[str, Any]] = []

    for entry in tree:
        if entry.get("type") != "blob":
            continue
        total_files += 1
        size = entry.get("size") or 0
        total_size += size

        path = entry.get("path", "")
        # Extension
        if "." in path:
            ext = "." + path.split(".")[-1]
        else:
            ext = ""
        by_extension[ext]["count"] += 1
        by_extension[ext]["size_bytes"] += size

        # Top-level dir
        if "/" in path:
            top = path.split("/", 1)[0]
        else:
            top = ""
        by_top_level[top] += 1

        files_sample.append(
            {
                "path": path,
                "size_bytes": size,
                "extension": ext,
            }
        )

    files_summary = {
        "total_files": total_files,
        "total_size_bytes": total_size,
        "by_extension": by_extension,
        "by_top_level_dir": by_top_level,
    }

    # Convert defaultdicts to regular dicts for JSON serialization
    files_summary["by_extension"] = dict(files_summary["by_extension"])
    files_summary["by_top_level_dir"] = dict(files_summary["by_top_level_dir"])

    return files_summary, files_sample


def fetch_commits_and_stats(
    cfg: GitHubConfig,
    owner: str,
    repo: str,
    branch: str,
    max_commits: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fetch up to max_commits from the given branch.
    Returns (commits_summary, commits_detailed, authors_detailed).
    """
    commits = github_get_paginated(
        cfg,
        f"/repos/{owner}/{repo}/commits",
        per_page=100,
        max_items=max_commits,
        extra_params={"sha": branch},
    )

    detailed_commits: List[Dict[str, Any]] = []
    authors_map: Dict[Tuple[Optional[str], Optional[str], Optional[str]], Dict[str, Any]] = {}
    total_additions = 0
    total_deletions = 0
    total_files_changed = 0

    for c in commits:
        sha = c.get("sha")
        commit_info = c.get("commit") or {}
        commit_author = commit_info.get("author") or {}
        author_name = commit_author.get("name")
        author_email = commit_author.get("email")
        date = commit_author.get("date")
        author_obj = c.get("author") or {}
        author_login = author_obj.get("login")

        # Fetch commit details for stats & files
        detail, _ = github_request(cfg, f"/repos/{owner}/{repo}/commits/{sha}")
        stats = detail.get("stats") or {}
        files = detail.get("files") or []

        additions = stats.get("additions", 0)
        deletions = stats.get("deletions", 0)
        total = stats.get("total", additions + deletions)

        total_additions += additions
        total_deletions += deletions
        total_files_changed += len(files)

        # Build lightweight per-file info (skip actual patch text to keep JSON smaller)
        files_info = []
        for f in files:
            files_info.append(
                {
                    "filename": f.get("filename"),
                    "status": f.get("status"),
                    "additions": f.get("additions"),
                    "deletions": f.get("deletions"),
                    "changes": f.get("changes"),
                }
            )

        detailed_commits.append(
            {
                "sha": sha,
                "author_login": author_login,
                "author_name": author_name,
                "author_email": author_email,
                "date": date,
                "message": commit_info.get("message"),
                "stats": {
                    "additions": additions,
                    "deletions": deletions,
                    "total": total,
                },
                "files": files_info,
            }
        )

        key = (author_name, author_email, author_login)
        entry = authors_map.get(key)
        if not entry:
            authors_map[key] = {
                "name": author_name,
                "email": author_email,
                "login": author_login,
                "total_commits": 1,
                "first_commit_date": date,
                "last_commit_date": date,
            }
        else:
            entry["total_commits"] += 1
            # Update first/last commit dates
            if entry["first_commit_date"] is None or (date and date < entry["first_commit_date"]):
                entry["first_commit_date"] = date
            if entry["last_commit_date"] is None or (date and date > entry["last_commit_date"]):
                entry["last_commit_date"] = date

    authors_detailed = list(authors_map.values())

    if detailed_commits:
        first_date = min(dc["date"] for dc in detailed_commits if dc["date"])
        last_date = max(dc["date"] for dc in detailed_commits if dc["date"])
    else:
        first_date = None
        last_date = None

    # Aggregate by author_login (GitHub username), useful for analyses
    by_author: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"commits": 0, "additions": 0, "deletions": 0})
    for dc in detailed_commits:
        login = dc.get("author_login") or dc.get("author_email") or "unknown"
        s = dc.get("stats") or {}
        by_author[login]["commits"] += 1
        by_author[login]["additions"] += s.get("additions", 0)
        by_author[login]["deletions"] += s.get("deletions", 0)
    by_author = dict(by_author)

    commits_summary = {
        "total_commits_fetched": len(detailed_commits),
        "first_commit_date": first_date,
        "last_commit_date": last_date,
        "total_additions": total_additions,
        "total_deletions": total_deletions,
        "total_files_changed": total_files_changed,
        "by_author": by_author,
    }

    return commits_summary, detailed_commits, authors_detailed


def fetch_repo_metadata(
    cfg: GitHubConfig,
    owner: str,
    repo: str
) -> Dict[str, Any]:
    """
    High-level per-repo fetch:
      - basic + misc metadata
      - language breakdown
      - contributors
      - head SHA & file tree summary
      - commits & per-commit stats
    Returns a dict ready to attach under project["github_metadata"].
    """
    full_name = f"{owner}/{repo}"
    print(f"Fetching GitHub metadata for {full_name}...")

    repo_data: Dict[str, Any] = {
        "repo_full_name": full_name,
        "owner": owner,
        "repo": repo,
    }

    # 1) Basic + misc
    try:
        basic_misc = fetch_basic_metadata(cfg, owner, repo)
    except RuntimeError as e:
        # If we hit a 404 or other fatal error, store it and bail on further fetches
        msg = str(e)
        if "HTTP 404" in msg:
            print(f"  -> Repo not found: {full_name}")
            repo_data["error"] = {"type": "not_found", "message": msg}
            return repo_data
        print(f"  !! Error fetching basic metadata for {full_name}: {e}")
        repo_data["error"] = {"type": "other_error", "message": msg}
        return repo_data

    if basic_misc is None:
        print(f"  -> Repo not accessible: {full_name}")
        repo_data["error"] = {"type": "not_accessible", "message": "404 or unknown"}
        return repo_data

    repo_data.update(basic_misc)
    default_branch = safe_get(basic_misc, "basic", "default_branch") or "main"

    # 2) Languages
    try:
        repo_data["languages"] = fetch_languages(cfg, owner, repo)
    except RuntimeError as e:
        print(f"  !! Error fetching languages for {full_name}: {e}")
        repo_data["languages"] = {}

    # 3) Contributors
    try:
        repo_data["contributors"] = fetch_contributors(cfg, owner, repo)
    except RuntimeError as e:
        print(f"  !! Error fetching contributors for {full_name}: {e}")
        repo_data["contributors"] = []

    # 4) File tree (head SHA)
    try:
        head_sha = fetch_branch_head_sha(cfg, owner, repo, default_branch)
        if head_sha:
            files_summary, files_sample = fetch_file_tree(cfg, owner, repo, head_sha)
            repo_data["files_summary"] = files_summary
            repo_data["files_sample"] = files_sample
        else:
            repo_data["files_summary"] = {}
            repo_data["files_sample"] = []
    except RuntimeError as e:
        print(f"  !! Error fetching file tree for {full_name}: {e}")
        repo_data["files_summary"] = {}
        repo_data["files_sample"] = []

    # 5) Commits & stats
    try:
        commits_summary, commits_detailed, authors_detailed = fetch_commits_and_stats(
            cfg,
            owner,
            repo,
            default_branch,
            max_commits=MAX_COMMITS_PER_REPO,
        )
        repo_data["commits_summary"] = commits_summary
        repo_data["commits_detailed"] = commits_detailed
        repo_data["authors_detailed"] = authors_detailed
    except RuntimeError as e:
        print(f"  !! Error fetching commits for {full_name}: {e}")
        repo_data["commits_summary"] = {}
        repo_data["commits_detailed"] = []
        repo_data["authors_detailed"] = []

    return repo_data


# =====================
# Caching
# =====================

def load_repo_cache(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as e:
        print(f"WARNING: Failed to load repo cache from {path}: {e}", file=sys.stderr)
        return {}


def save_repo_cache(path: str, cache: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


# =====================
# Main enrichment logic
# =====================

def collect_unique_repos(projects_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Iterate over the JSON structure and collect unique GitHub repos.

    Returns a dict:
      { "owner/repo": { "owner": owner, "repo": repo, "urls": [ ... ] } }
    """
    repos: Dict[str, Dict[str, Any]] = {}

    for year, projects in projects_json.items():
        if not isinstance(projects, list):
            continue
        for project in projects:
            urls: List[str] = []
            primary = project.get("primary_github")
            if primary:
                urls.append(primary)
            all_github = project.get("all_github") or []
            urls.extend(all_github)

            for url in urls:
                parsed = parse_github_repo(url)
                if not parsed:
                    continue
                owner, repo = parsed
                full_name = f"{owner}/{repo}"
                entry = repos.get(full_name)
                if not entry:
                    repos[full_name] = {
                        "owner": owner,
                        "repo": repo,
                        "urls": [url],
                    }
                else:
                    if url not in entry["urls"]:
                        entry["urls"].append(url)

    return repos


def enrich_projects_with_github(
    projects_json: Dict[str, Any],
    repo_metadata_cache: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach github_metadata to each project by looking up the repo cache.
    The output structure mirrors input, but each project gets:

      "github_metadata": [
        { ...repo metadata for first repo... },
        { ...repo metadata for second repo... },
        ...
      ]
    """
    # Build quick lookup: full_name -> metadata
    metadata_by_full_name: Dict[str, Any] = repo_metadata_cache

    output = {}
    for year, projects in projects_json.items():
        if not isinstance(projects, list):
            output[year] = projects
            continue

        new_projects = []
        for project in projects:
            # gather all repo full_names for this project
            repo_entries: List[Dict[str, Any]] = []
            urls: List[str] = []
            primary = project.get("primary_github")
            if primary:
                urls.append(primary)
            all_github = project.get("all_github") or []
            urls.extend(all_github)

            seen_full_names = set()
            for url in urls:
                parsed = parse_github_repo(url)
                if not parsed:
                    continue
                owner, repo = parsed
                full_name = f"{owner}/{repo}"
                if full_name in seen_full_names:
                    continue
                seen_full_names.add(full_name)
                meta = metadata_by_full_name.get(full_name)
                if meta:
                    repo_entries.append(meta)

            project_with_meta = dict(project)
            project_with_meta["github_metadata"] = repo_entries
            new_projects.append(project_with_meta)

        output[year] = new_projects

    return output


def main():
    token = load_github_token()
    cfg = GitHubConfig(token=token)

    # Load input JSON
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"ERROR: Input JSON file not found: {INPUT_JSON_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        projects_json = json.load(f)

    # Collect unique repos present in the dataset
    repos = collect_unique_repos(projects_json)
    print(f"Found {len(repos)} unique GitHub repos in input JSON.")

    # Load existing repo metadata cache (if any)
    repo_metadata_cache = load_repo_cache(REPO_CACHE_PATH)
    print(f"Loaded {len(repo_metadata_cache)} repos from cache ({REPO_CACHE_PATH}).")

    # Fetch missing repos and repos that had errors (e.g. rate-limited on a previous run)
    missing = [
        full_name for full_name in repos.keys()
        if full_name not in repo_metadata_cache
    ]
    errored = [
        full_name for full_name in repos.keys()
        if full_name in repo_metadata_cache and "error" in repo_metadata_cache[full_name]
    ]
    to_fetch = missing + errored
    print(f"{len(missing)} repos missing from cache, {len(errored)} repos had errors and will be re-fetched.")
    print(f"{len(to_fetch)} total repos to fetch from GitHub.")

    fetched_count = 0
    for i, full_name in enumerate(to_fetch, start=1):
        owner = repos[full_name]["owner"]
        repo = repos[full_name]["repo"]

        try:
            meta = fetch_repo_metadata(cfg, owner, repo)
        except RuntimeError as e:
            print(f"!! Fatal error fetching {full_name}: {e}")
            meta = {
                "repo_full_name": full_name,
                "owner": owner,
                "repo": repo,
                "error": {"type": "fatal_error", "message": str(e)},
            }

        repo_metadata_cache[full_name] = meta
        fetched_count += 1

        # Periodically save cache to avoid losing work
        if fetched_count % SAVE_CACHE_EVERY_N_REPOS == 0:
            print(f"Saving repo cache after {fetched_count} new repos...")
            save_repo_cache(REPO_CACHE_PATH, repo_metadata_cache)

        # Optional small sleep to be extra nice to the API
        time.sleep(0.1)

    # Final save of cache
    print("Saving final repo cache...")
    save_repo_cache(REPO_CACHE_PATH, repo_metadata_cache)

    # Attach metadata to projects and write output
    print("Attaching GitHub metadata to projects...")
    enriched = enrich_projects_with_github(projects_json, repo_metadata_cache)

    print(f"Writing enriched JSON to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, sort_keys=False)

    print("Done.")


if __name__ == "__main__":
    main()
