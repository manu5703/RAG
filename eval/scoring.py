"""
Scoring helpers shared by the retrieval and answer evals. Pure functions.

Answer labels
-------------
  correct          answerable question, answered, all key facts present
  incorrect        answerable question, answered, a key fact missing or wrong
  missed           answerable question, but the model abstained
  abstained_ok     unanswerable question, model abstained (the right call)
  hallucinated     unanswerable question, model answered anyway

"Hallucination" for the detector = any answered response that is wrong
(incorrect + hallucinated).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


def load_qa(path: str | Path) -> List[dict]:
    with Path(path).open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate question ids in {path}")
    return items


def norm_url(url: str) -> str:
    u = url.strip().lower()
    u = re.sub(r"^https?://(www\.)?", "", u)
    u = re.sub(r"(/index\.(php|html?))?/?$", "", u)
    return u


def _norm_text(s: str) -> str:
    s = s.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s)


def term_present(term: str, text: str) -> bool:
    """Numbers must match as whole numbers ("3" doesn't match "30"); other terms as substrings."""
    t, x = _norm_text(term), _norm_text(text)
    if re.fullmatch(r"[\d.,]+", t):
        return re.search(rf"(?<![\d.]){re.escape(t)}(?![\d])", x) is not None
    return t in x


def facts_present(must_include: Sequence[Sequence[str]], answer: str) -> bool:
    """`must_include` is a list of any-of groups; every group needs one match."""
    return all(any(term_present(t, answer) for t in group) for group in must_include)


def label_answer(item: dict, answer: str, abstained: bool) -> str:
    if not item["answerable"]:
        return "abstained_ok" if abstained else "hallucinated"
    if abstained:
        return "missed"
    return "correct" if facts_present(item["must_include"], answer) else "incorrect"


def is_wrong_answer(label: str) -> bool:
    return label in ("incorrect", "hallucinated")


def recall_at_k(ranked_urls: Sequence[str], gold_url: str, k: int) -> float:
    g = norm_url(gold_url)
    return float(any(norm_url(u) == g for u in ranked_urls[:k]))


def reciprocal_rank(ranked_urls: Sequence[str], gold_url: str, k: int = 10) -> float:
    g = norm_url(gold_url)
    for i, u in enumerate(ranked_urls[:k], 1):
        if norm_url(u) == g:
            return 1.0 / i
    return 0.0


def pct(values: Iterable[float], q: float) -> float:
    v = np.asarray(list(values), dtype=float)
    return float(np.percentile(v, q)) if v.size else float("nan")


def md_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.3f}" if abs(x) < 10 else f"{x:.0f}"
        return str(x)

    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows]
    return "\n".join(out)
