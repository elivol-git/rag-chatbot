"""Seed data/documents with public-domain / CC-BY-SA architecture texts.

Wikipedia extracts come from the MediaWiki API (CC BY-SA 4.0); the classic
treatises come from Project Gutenberg (public domain). Every file gets a small
header the loader turns into chunk metadata.

Usage:  python scripts/fetch_corpus.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "data" / "documents"

WIKIPEDIA_TITLES = [
    "Architecture",
    "Gothic architecture",
    "Romanesque architecture",
    "Renaissance architecture",
    "Baroque architecture",
    "Neoclassical architecture",
    "Modern architecture",
    "Bauhaus",
    "Brutalist architecture",
    "Postmodern architecture",
    "Deconstructivism",
    "Le Corbusier",
    "Frank Lloyd Wright",
    "Zaha Hadid",
    "Ludwig Mies van der Rohe",
    "Antoni Gaudí",
    "Sustainable architecture",
    "Passive house",
    "Vernacular architecture",
    "Flying buttress",
    "Dome",
    "Arch",
    "Building code",
    "Structural engineering",
    "Urban design",
]

GUTENBERG_TEXTS = {
    "vitruvius-ten-books-on-architecture": (
        "https://www.gutenberg.org/cache/epub/20239/pg20239.txt",
        "Vitruvius — The Ten Books on Architecture",
    ),
    "ruskin-seven-lamps-of-architecture": (
        "https://www.gutenberg.org/cache/epub/35898/pg35898.txt",
        "John Ruskin — The Seven Lamps of Architecture",
    ),
    "sullivan-kindergarten-chats": (
        "https://www.gutenberg.org/cache/epub/64909/pg64909.txt",
        "Louis Sullivan — Kindergarten Chats",
    ),
}

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "RAG-chatbot-coursework/1.0 (educational use)"}


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def write_doc(path: Path, title: str, source_url: str, license_: str, body: str) -> None:
    header = (
        f"title: {title}\n"
        f"source_url: {source_url}\n"
        f"license: {license_}\n"
        f"---\n\n"
    )
    path.write_text(header + body.strip() + "\n", encoding="utf-8")


def fetch_wikipedia(title: str, attempts: int = 4) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "titles": title,
    }
    for attempt in range(attempts):
        response = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=60)
        # The API throttles bursts; back off and retry rather than losing the page.
        if response.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"       429 on {title}, retrying in {wait}s")
            time.sleep(wait)
            continue
        response.raise_for_status()
        pages = response.json()["query"]["pages"]
        page = next(iter(pages.values()))
        return page.get("extract", "")
    raise requests.RequestException(f"rate limited after {attempts} attempts")


def strip_gutenberg_boilerplate(text: str) -> str:
    start = re.search(r"\*\*\* START OF TH[EIS]+ PROJECT GUTENBERG.*?\*\*\*", text)
    end = re.search(r"\*\*\* END OF TH[EIS]+ PROJECT GUTENBERG.*?\*\*\*", text)
    body = text[start.end() : end.start()] if start and end else text
    return re.sub(r"\n{3,}", "\n\n", body)


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0

    for title in WIKIPEDIA_TITLES:
        path = DOCS_DIR / f"wiki-{slugify(title)}.md"
        if path.exists():
            print(f"skip   {path.name}")
            continue
        try:
            body = fetch_wikipedia(title)
        except requests.RequestException as exc:
            print(f"FAIL   {title}: {exc}", file=sys.stderr)
            continue
        if len(body) < 500:
            print(f"FAIL   {title}: extract too short", file=sys.stderr)
            continue
        write_doc(
            path,
            title,
            f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "CC BY-SA 4.0",
            body,
        )
        written += 1
        print(f"wrote  {path.name} ({len(body):,} chars)")
        time.sleep(1.5)

    for slug, (url, title) in GUTENBERG_TEXTS.items():
        path = DOCS_DIR / f"{slug}.md"
        if path.exists():
            print(f"skip   {path.name}")
            continue
        try:
            response = requests.get(url, headers=HEADERS, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"FAIL   {title}: {exc}", file=sys.stderr)
            continue
        body = strip_gutenberg_boilerplate(response.text)
        write_doc(path, title, url, "Public domain (Project Gutenberg)", body)
        written += 1
        print(f"wrote  {path.name} ({len(body):,} chars)")

    print(f"\n{written} new document(s) in {DOCS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
