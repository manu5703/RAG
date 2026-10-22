"""
HTML -> (title, sections). Pure functions, no network, so they are unit-tested.

University pages wrap the useful content in nav bars, footers and sidebars.
We drop those, then split the remaining text at h1-h4 headings so each chunk
carries the heading it sits under (that heading is often what the question
is about: "Application Window", "Offer Letter Requirements", ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple
from urllib.parse import urljoin, urldefrag

from bs4 import BeautifulSoup, NavigableString, Tag

DROP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "form", "svg", "iframe", "aside"]
DROP_CLASS_HINTS = ("nav", "menu", "breadcrumb", "footer", "header", "sidebar", "skip", "cookie", "social")
HEADINGS = {"h1", "h2", "h3", "h4"}
BLOCKS = {"p", "li", "td", "th", "dd", "dt", "pre", "blockquote", "caption"}


@dataclass
class Section:
    heading: str
    text: str


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _looks_like_chrome(tag: Tag) -> bool:
    attrs = " ".join([tag.get("id") or ""] + list(tag.get("class") or [])).lower()
    return any(h in attrs for h in DROP_CLASS_HINTS)


def _main_container(soup: BeautifulSoup) -> Tag:
    for sel in ("main", "article", "[role=main]", "#main", "#content", ".content"):
        found = soup.select_one(sel)
        if found and len(found.get_text(strip=True)) > 200:
            return found
    return soup.body or soup


def _table_rows(table: Tag) -> List[str]:
    rows = []
    for tr in table.find_all("tr"):
        cells = [_clean(c.get_text(" ")) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return rows


def parse_html(html: str) -> Tuple[str, List[Section]]:
    soup = BeautifulSoup(html, "html.parser")
    title = _clean(soup.title.get_text()) if soup.title else ""

    for t in soup.find_all(DROP_TAGS):
        t.decompose()
    for t in soup.find_all(True):
        if t.attrs is not None and t.name not in ("html", "body", "main", "article") and _looks_like_chrome(t):
            t.decompose()

    root = _main_container(soup)
    sections: List[Section] = []
    heading, buf = "", []

    def flush():
        text = "\n".join(b for b in buf if b)
        if text.strip():
            sections.append(Section(heading=heading, text=text))

    for el in root.descendants:
        if not isinstance(el, Tag):
            continue
        if el.name in HEADINGS:
            flush()
            heading, buf = _clean(el.get_text(" ")), []
        elif el.name == "table":
            buf.extend(_table_rows(el))
        elif el.name in BLOCKS and not el.find_parent("table"):
            # skip blocks nested in another block we already take whole
            if el.find_parent(list(BLOCKS - {"blockquote"})):
                continue
            buf.append(_clean(el.get_text(" ")))
    flush()

    if not sections:  # page with no block tags at all
        text = _clean(root.get_text(" "))
        if text:
            sections = [Section(heading="", text=text)]

    page_h1 = root.find("h1")
    if page_h1 and _clean(page_h1.get_text(" ")):
        title = _clean(page_h1.get_text(" "))
    return title, sections


def extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if isinstance(href, NavigableString):
            href = str(href)
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        url, _ = urldefrag(urljoin(base_url, href))
        out.append(url)
    return sorted(set(out))
