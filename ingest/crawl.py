"""
Crawl the pages in ingest/sources.yaml and build the corpus the assistant
answers from.

    python -m ingest.crawl                      # crawl + chunk
    python -m ingest.crawl --offline            # re-chunk from cached HTML only

Writes:
    corpus/raw/<hash>.html    cached page HTML (re-used with --offline)
    corpus/pages.jsonl        one record per page: url, title, fetched_at
    corpus/chunks.jsonl       the chunks the index is built from

Polite by default: honours robots.txt, one request per second, identifies
itself with a User-Agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import yaml

from app.document_loader import chunk_sections
from app.schema import Chunk, write_chunks
from ingest.html_parse import extract_links, parse_html

logger = logging.getLogger(__name__)

USER_AGENT = "campus-rag-assistant/1.0 (student research project; +https://github.com/manu5703/RAG)"
SKIP_EXT = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".docx", ".xlsx", ".pptx", ".mp4")


def _key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


class Fetcher:
    def __init__(self, raw_dir: Path, delay_s: float = 1.0, offline: bool = False):
        self.raw_dir = raw_dir
        self.delay_s = delay_s
        self.offline = offline
        self._robots: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last = 0.0
        raw_dir.mkdir(parents=True, exist_ok=True)

    def _allowed(self, url: str) -> bool:
        root = "{0.scheme}://{0.netloc}".format(urlparse(url))
        if root not in self._robots:
            rp = urllib.robotparser.RobotFileParser(root + "/robots.txt")
            try:
                rp.read()
            except Exception:  # unreachable robots.txt -> allow, like most crawlers
                rp = None
            self._robots[root] = rp
        rp = self._robots[root]
        return True if rp is None else rp.can_fetch(USER_AGENT, url)

    def get(self, url: str) -> str | None:
        path = self.raw_dir / f"{_key(url)}.html"
        if self.offline:
            return path.read_text(encoding="utf-8") if path.exists() else None
        if not self._allowed(url):
            logger.warning("robots.txt disallows %s", url)
            return None

        import requests  # imported lazily so --offline needs no network stack

        wait = self.delay_s - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        except requests.RequestException as e:
            logger.warning("fetch failed %s: %s", url, e)
            return None
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
            logger.warning("skip %s (status %s, %s)", url, r.status_code, r.headers.get("content-type"))
            return None
        path.write_text(r.text, encoding="utf-8")
        return r.text


def crawl(sources_path: Path, out_dir: Path, offline: bool = False, max_pages: int = 300) -> List[Chunk]:
    cfg = yaml.safe_load(sources_path.read_text())
    fetcher = Fetcher(out_dir / "raw", offline=offline)
    seen: set[str] = set()
    pages, chunks = [], []

    for site in cfg["sites"]:
        queue = list(site["seeds"])
        seeds = set(site["seeds"])
        prefixes = site.get("allow_prefixes", [])
        while queue and len(seen) < max_pages:
            url = queue.pop(0)
            if url in seen or url.lower().endswith(SKIP_EXT):
                continue
            seen.add(url)
            html = fetcher.get(url)
            if html is None:
                continue
            title, sections = parse_html(html)
            page_chunks = chunk_sections(sections, url=url, title=title)
            if not page_chunks:
                continue
            chunks.extend(page_chunks)
            pages.append({
                "url": url,
                "title": title,
                "site": site["name"],
                "n_chunks": len(page_chunks),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            logger.info("%3d chunks  %s", len(page_chunks), url)

            if site.get("follow_links") and url in seeds:
                for link in extract_links(html, url):
                    if any(link.startswith(p) for p in prefixes) and link not in seen:
                        queue.append(link)

    # identical boilerplate blocks repeat across pages: keep the first copy
    unique, texts = [], set()
    for c in chunks:
        if c.text not in texts:
            texts.add(c.text)
            unique.append(c)

    write_chunks(unique, out_dir / "chunks.jsonl")
    with (out_dir / "pages.jsonl").open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p) + "\n")
    logger.info("Corpus: %d pages, %d chunks (%d duplicates dropped) -> %s",
                len(pages), len(unique), len(chunks) - len(unique), out_dir)
    return unique


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="ingest/sources.yaml", type=Path)
    ap.add_argument("--out", default="corpus", type=Path)
    ap.add_argument("--offline", action="store_true", help="only use cached HTML in corpus/raw")
    ap.add_argument("--max-pages", type=int, default=300)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    crawl(args.sources, args.out, offline=args.offline, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
