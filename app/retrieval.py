"""
Retrievers: BM25, dense, hybrid (weighted or reciprocal-rank fusion), and a
cross-encoder reranker. Every retriever returns Hit(chunk, score), so the
eval harness can swap them to run the ablation.

Heavy models (sentence-transformers, FAISS) load lazily, so unit tests can
inject a small fake embedder and run without downloads.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from app.schema import Chunk, Hit

logger = logging.getLogger(__name__)

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")

# ─── Tokenizer for BM25 ───────────────────────────────────────────────────────

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.:/=][a-z0-9]+)*")
_STOP = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its "
    "my of on or that the their there this to was what when where which who why "
    "will with you your".split()
)


def tokenize(text: str) -> List[str]:
    """
    Lowercase word tokens that keep form names and flags intact
    ("i-765", "--gres=gpu:1", "3.5") and also emit their parts, so
    "I-765" matches "765" and "gres" matches "--gres=gpu".
    """
    out: List[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP:
            continue
        out.append(tok)
        parts = re.split(r"[-_.:/=]", tok)
        if len(parts) > 1:
            out.extend(p for p in parts if p and p not in _STOP)
    return out


# ─── Interfaces ───────────────────────────────────────────────────────────────

class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(list(texts), convert_to_numpy=True, batch_size=64,
                                 show_progress_bar=len(texts) > 256)


def _l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


# ─── Retrievers ───────────────────────────────────────────────────────────────

class BM25Retriever:
    name = "bm25"

    def build(self, chunks: List[Chunk]) -> "BM25Retriever":
        from rank_bm25 import BM25Okapi

        if not chunks:
            raise ValueError("Cannot build index over an empty corpus.")
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.indexed_text) or ["<empty>"] for c in chunks])
        return self

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype="float32")

    def search(self, query: str, k: int) -> List[Hit]:
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [Hit(self.chunks[i], float(s[i])) for i in top]


class DenseRetriever:
    name = "dense"

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index = None

    def build(self, chunks: List[Chunk]) -> "DenseRetriever":
        if not chunks:
            raise ValueError("Cannot build index over an empty corpus.")
        self.chunks = chunks
        self.emb = _l2norm(self.embedder.encode([c.indexed_text for c in chunks]))
        try:
            import faiss

            self.index = faiss.IndexFlatIP(self.emb.shape[1])
            self.index.add(self.emb)
        except ImportError:  # numpy fallback is fine for a few thousand chunks
            self.index = None
        return self

    def query_vec(self, query: str) -> np.ndarray:
        return _l2norm(self.embedder.encode([query]))[0]

    def scores(self, query: str) -> np.ndarray:
        """Cosine similarity to every chunk."""
        return self.emb @ self.query_vec(query)

    def search(self, query: str, k: int) -> List[Hit]:
        k = min(k, len(self.chunks))
        if self.index is not None:
            s, ids = self.index.search(self.query_vec(query)[None, :], k)
            return [Hit(self.chunks[i], float(v)) for i, v in zip(ids[0], s[0]) if i >= 0]
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [Hit(self.chunks[i], float(s[i])) for i in top]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


class HybridRetriever:
    """
    Fuses BM25 and dense rankings.

    fusion="rrf"       reciprocal-rank fusion: sum of 1 / (rrf_k + rank). Scale-free,
                       so it needs no tuning of how BM25 and cosine scores compare.
    fusion="weighted"  alpha * dense + (1 - alpha) * bm25, both min-max normalised
                       over the whole corpus. Every candidate gets its real dense
                       score (the old version gave BM25-only candidates a dense score of 0).
    """

    name = "hybrid"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever,
                 fusion: str = "rrf", alpha: float = 0.6, rrf_k: int = 60):
        if fusion not in ("rrf", "weighted"):
            raise ValueError(f"unknown fusion {fusion!r}")
        self.dense, self.bm25 = dense, bm25
        self.fusion, self.alpha, self.rrf_k = fusion, alpha, rrf_k

    def build(self, chunks: List[Chunk]) -> "HybridRetriever":
        self.chunks = chunks
        self.dense.build(chunks)
        self.bm25.build(chunks)
        return self

    def search(self, query: str, k: int) -> List[Hit]:
        d = self.dense.scores(query)
        b = self.bm25.scores(query)
        if self.fusion == "weighted":
            fused = self.alpha * _minmax(d) + (1 - self.alpha) * _minmax(b)
        else:
            rank_d = np.empty(len(d), dtype=np.int64)
            rank_d[np.argsort(-d)] = np.arange(1, len(d) + 1)
            rank_b = np.empty(len(b), dtype=np.int64)
            rank_b[np.argsort(-b)] = np.arange(1, len(b) + 1)
            # chunks with no BM25 term overlap shouldn't earn BM25 rank credit
            bm25_part = np.where(b > 0, 1.0 / (self.rrf_k + rank_b), 0.0)
            fused = 1.0 / (self.rrf_k + rank_d) + bm25_part
        top = np.argsort(-fused)[:k]
        return [Hit(self.chunks[i], float(fused[i])) for i in top]


# ─── Reranker ─────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker: %s", model_name)
        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.predict([(query, t) for t in texts]), dtype="float32")


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> np.ndarray: ...


def rerank(reranker: Reranker, query: str, hits: List[Hit], k: int) -> List[Hit]:
    if not hits:
        return []
    s = reranker.score(query, [h.chunk.indexed_text for h in hits])
    order = np.argsort(-s)[:k]
    return [Hit(hits[i].chunk, float(s[i])) for i in order]


# ─── Factory ──────────────────────────────────────────────────────────────────

class Retrieval:
    """One configured retrieval stage: first-pass retriever + optional reranker."""

    def __init__(self, mode: str, first_pass, reranker: Optional[Reranker], pool: int):
        self.mode, self.first_pass, self.reranker, self.pool = mode, first_pass, reranker, pool

    def search(self, query: str, k: int, timings: Optional[Dict[str, float]] = None) -> List[Hit]:
        import time

        t0 = time.perf_counter()
        hits = self.first_pass.search(query, self.pool if self.reranker else k)
        t1 = time.perf_counter()
        if self.reranker is not None:
            hits = rerank(self.reranker, query, hits, k)
        t2 = time.perf_counter()
        if timings is not None:
            timings["retrieve_ms"] = (t1 - t0) * 1000
            timings["rerank_ms"] = (t2 - t1) * 1000
        return hits


def build_retrieval(
    mode: str,
    chunks: List[Chunk],
    embedder: Optional[Embedder] = None,
    reranker: Optional[Reranker] = None,
    fusion: str = "rrf",
    alpha: float = 0.6,
    pool: int = 20,
) -> Retrieval:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "bm25":
        return Retrieval(mode, BM25Retriever().build(chunks), None, pool)
    if embedder is None:
        from app.config import settings

        embedder = SentenceTransformerEmbedder(settings.embed_model)
    if mode == "dense":
        return Retrieval(mode, DenseRetriever(embedder).build(chunks), None, pool)
    hybrid = HybridRetriever(DenseRetriever(embedder), BM25Retriever(), fusion=fusion, alpha=alpha).build(chunks)
    if mode == "hybrid":
        return Retrieval(mode, hybrid, None, pool)
    if reranker is None:
        from app.config import settings

        reranker = CrossEncoderReranker(settings.rerank_model)
    return Retrieval(mode, hybrid, reranker, pool)
