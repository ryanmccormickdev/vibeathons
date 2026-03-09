#!/usr/bin/env python3
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ------------- CONFIGURATION -------------

# All the TreeHacks Devpost galleries we want to scrape.
HACKATHONS = [
    {"name": "treehacks_2026", "gallery_url": "https://treehacks-2026.devpost.com/project-gallery"},
    {"name": "treehacks_2025", "gallery_url": "https://treehacks-2025.devpost.com/project-gallery"},
    {"name": "treehacks_2024", "gallery_url": "https://treehacks-2024.devpost.com/project-gallery"},
    {"name": "treehacks_2023", "gallery_url": "https://treehacks-2023.devpost.com/project-gallery"},
    {"name": "treehacks_2022", "gallery_url": "https://treehacks-2022.devpost.com/project-gallery"},
    {"name": "treehacks_2021", "gallery_url": "https://treehacks-2021.devpost.com/project-gallery"},
    {"name": "treehacks_2020", "gallery_url": "https://treehacks-2020.devpost.com/project-gallery"},
    {"name": "treehacks_2019", "gallery_url": "https://treehacks-2019.devpost.com/project-gallery"},
    {"name": "treehacks_2018", "gallery_url": "https://treehacks-2018.devpost.com/project-gallery"},
    {"name": "treehacks_2017", "gallery_url": "https://treehacks-2017.devpost.com/project-gallery"},
    {"name": "treehacks_2016", "gallery_url": "https://treehacks-2016.devpost.com/project-gallery"},
    {"name": "treehacks_2015", "gallery_url": "https://treehackswinter2015.devpost.com/project-gallery"},
]

# [INFO] Scraping hackathon treehacks_2026 from https://treehacks-2026.devpost.com/project-gallery
# [INFO]  Found 378 project URLs for treehacks_2026
# [INFO] Scraping hackathon treehacks_2025 from https://treehacks-2025.devpost.com/project-gallery
# [INFO]  Found 257 project URLs for treehacks_2025
# [INFO] Scraping hackathon treehacks_2024 from https://treehacks-2024.devpost.com/project-gallery
# [INFO]  Found 330 project URLs for treehacks_2024
# [INFO] Scraping hackathon treehacks_2023 from https://treehacks-2023.devpost.com/project-gallery
# [INFO]  Found 293 project URLs for treehacks_2023
# [INFO] Scraping hackathon treehacks_2022 from https://treehacks-2022.devpost.com/project-gallery
# [INFO]  Found 114 project URLs for treehacks_2022
# [INFO] Scraping hackathon treehacks_2021 from https://treehacks-2021.devpost.com/project-gallery
# [INFO]  Found 223 project URLs for treehacks_2021
# [INFO] Scraping hackathon treehacks_2020 from https://treehacks-2020.devpost.com/project-gallery
# [INFO]  Found 197 project URLs for treehacks_2020
# [INFO] Scraping hackathon treehacks_2019 from https://treehacks-2019.devpost.com/project-gallery
# [INFO]  Found 181 project URLs for treehacks_2019
# [INFO] Scraping hackathon treehacks_2018 from https://treehacks-2018.devpost.com/project-gallery
# [INFO]  Found 114 project URLs for treehacks_2018
# [INFO] Scraping hackathon treehacks_2017 from https://treehacks-2017.devpost.com/project-gallery
# [INFO]  Found 123 project URLs for treehacks_2017
# [INFO] Scraping hackathon treehacks_2016 from https://treehacks-2016.devpost.com/project-gallery
# [INFO]  Found 84 project URLs for treehacks_2016
# [INFO] Scraping hackathon treehacks_2015 from https://treehackswinter2015.devpost.com/project-gallery
# [INFO]  Found 127 project URLs for treehacks_2015

expected_counts = {
    "treehacks_2026": 378,
    "treehacks_2025": 257,
    "treehacks_2024": 330,
    "treehacks_2023": 293,
    "treehacks_2022": 114,
    "treehacks_2021": 223,
    "treehacks_2020": 197,
    "treehacks_2019": 181,
    "treehacks_2018": 114,
    "treehacks_2017": 123,
    "treehacks_2016": 84,
    "treehacks_2015": 127,
}

# JSON cache file we read from / write to
OUTPUT_JSON_PATH = Path("treehacks_projects.json")

# Skip scraping a hackathon if it already has non-empty entries in the JSON.
# (Matches your example: treehacks_2026 has data => skip; others empty => scrape.)
SKIP_IF_PRESENT_AND_NONEMPTY = True

# Be nice-ish to Devpost; tune as needed
REQUEST_DELAY_SECONDS = 0.5
TIMEOUT_SECONDS = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; treehacks-devpost-scraper/0.2; "
        "+https://example.com)"
    )
}

# ------------- LOW-LEVEL HELPERS -------------

def http_get(url: str) -> Optional[str]:
    """GET a URL and return response text, or None on error."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"[WARN] Failed to GET {url}: {e}", file=sys.stderr)
        return None
    finally:
        time.sleep(REQUEST_DELAY_SECONDS)


def normalize_github_url(url: str) -> str:
    """Normalize various github-ish URL forms into a cleaner one."""
    url = url.strip()
    if url.startswith("//"):
        url = "https:" + url
    if url.startswith("github.com/"):
        url = "https://" + url

    # strip trailing punctuation that might be attached in text
    url = url.rstrip(").,;\"'")
    return url


def is_github_url(url: str) -> bool:
    """Return True if URL points to GitHub."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    host = (parsed.netloc or "").lower()
    if not host and url.lower().startswith("github.com/"):
        return True
    return "github.com" in host


def load_existing_json(path: Path) -> dict[str, list[dict]]:
    """Load existing JSON if present; otherwise return empty dict."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure the shape we expect
        if isinstance(data, dict):
            return data  # type: ignore[return-value]
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}", file=sys.stderr)
    return {}


def should_scrape(hackathon_name: str, existing: dict[str, list[dict]]) -> bool:
    """
    Decide whether to scrape this hackathon based on what's already in the JSON.
    Behavior:
      - if key is missing or value is empty/non-list => scrape
      - if expected_counts has an entry and current count is lower => scrape
      - otherwise => skip
    """
    if not SKIP_IF_PRESENT_AND_NONEMPTY:
        return True

    val = existing.get(hackathon_name)
    current_count = len(val) if isinstance(val, list) else 0

    if current_count == 0:
        return True

    expected_count = expected_counts.get(hackathon_name)
    if expected_count is not None and current_count < expected_count:
        print(
            f"[INFO] Re-scraping {hackathon_name}: have {current_count}, expected at least {expected_count}",
            file=sys.stderr,
        )
        return True

    return False


# ------------- EXTRACTION LOGIC -------------

def extract_project_title(html: str) -> Optional[str]:
    """Very simple heuristic to get project title from project page."""
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)

    return None


def extract_github_urls_from_html(html: str) -> list[str]:
    """Extract all GitHub URLs from a project page's HTML."""
    soup = BeautifulSoup(html, "html.parser")
    urls: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = normalize_github_url(a["href"])
        if is_github_url(href):
            urls.add(href)

    text = soup.get_text(" ", strip=True)
    for match in re.findall(r"https?://github\.com/[^\s)]+", text, flags=re.I):
        urls.add(normalize_github_url(match))

    for match in re.findall(r"github\.com/[^\s)]+", text, flags=re.I):
        urls.add(normalize_github_url(match))

    return sorted(urls)


def get_primary_github_from_try_it_out(html: str) -> Optional[str]:
    """
    Try to find a single 'primary' GitHub link from the 'Try it out' section.
    Returns None if nothing obvious is found.
    """
    soup = BeautifulSoup(html, "html.parser")

    heading = soup.find(
        lambda tag: tag.name in ("h2", "h3", "h4")
        and tag.get_text(strip=True).lower().startswith("try it out")
    )
    if not heading:
        return None

    container = heading.find_next(["ul", "ol", "div"])
    if not container:
        return None

    for a in container.find_all("a", href=True):
        href = normalize_github_url(a["href"])
        if is_github_url(href):
            return href

    return None


def get_project_urls_for_hackathon(gallery_url: str, max_pages: int = 50) -> list[str]:
    """
    Enumerate all project links from a hackathon's project gallery, following ?page=N.
    Returns a sorted list of absolute project URLs.
    """
    project_urls: set[str] = set()

    for page in range(1, max_pages + 1):
        page_url = gallery_url if page == 1 else f"{gallery_url}?page={page}"
        html = http_get(page_url)
        if html is None:
            break

        soup = BeautifulSoup(html, "html.parser")
        found_this_page = 0

        for a in soup.find_all("a", href=True):
            full = urljoin(gallery_url, a["href"])

            if "devpost.com" in full and ("/software/" in full or "/submissions/" in full):
                if full not in project_urls:
                    project_urls.add(full)
                    found_this_page += 1

        if found_this_page == 0:
            break

    return sorted(project_urls)


def scrape_project(project_url: str) -> dict:
    """
    Scrape a single Devpost project page:
    - project URL
    - project title (best-effort)
    - primary GitHub URL (if found in Try it out)
    - all GitHub URLs (anywhere on the page)
    """
    html = http_get(project_url)
    if html is None:
        return {
            "project_url": project_url,
            "project_title": None,
            "primary_github": None,
            "all_github": [],
        }

    title = extract_project_title(html)
    all_github = extract_github_urls_from_html(html)
    primary_github = get_primary_github_from_try_it_out(html)

    return {
        "project_url": project_url,
        "project_title": title,
        "primary_github": primary_github,
        "all_github": all_github,
    }


# ------------- MAIN SCRIPT -------------

def main() -> None:
    existing = load_existing_json(OUTPUT_JSON_PATH)
    result: dict[str, list[dict]] = dict(existing)  # start from what we already have

    for hackathon in HACKATHONS:
        name = hackathon["name"]
        gallery_url = hackathon["gallery_url"]

        if not should_scrape(name, existing):
            print(
                f"[INFO] Skipping {name} (already present in {OUTPUT_JSON_PATH} with {len(existing.get(name, []))} entries)",
                file=sys.stderr,
            )
            continue

        print(f"[INFO] Scraping hackathon {name} from {gallery_url}", file=sys.stderr)

        projects = get_project_urls_for_hackathon(gallery_url)
        print(f"[INFO]  Found {len(projects)} project URLs for {name}", file=sys.stderr)

        year_entries: list[dict] = []
        for idx, project_url in enumerate(projects, start=1):
            print(f"[INFO]   [{name}] ({idx}/{len(projects)}) {project_url}", file=sys.stderr)
            year_entries.append(scrape_project(project_url))

        result[name] = year_entries

    # Write merged output
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Also dump to stdout (keeps old behavior)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    print()  # final newline


if __name__ == "__main__":
    main()
