"""
RAG pipeline: retrieval (BM25 / dense / hybrid, optional cross-encoder rerank)
-> grounded generation with citations -> per-stage latency.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence, Union

from app.config import settings
from app.generators import build_generator, cited_indices, is_abstention
from app.retrieval import Embedder, Reranker, build_retrieval
from app.schema import Chunk

logger = logging.getLogger(__name__)


def as_chunks(items: Sequence[Union[str, Chunk]]) -> List[Chunk]:
    """Accept plain strings (e.g. chunks from an uploaded PDF) or Chunk objects."""
    return [c if isinstance(c, Chunk) else Chunk(text=c) for c in items]


class RAGPipeline:
    def __init__(
        self,
        chunks: Sequence[Union[str, Chunk]],
        mode: Optional[str] = None,
        generator=None,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
    ):
        self.mode = mode or settings.retrieval_mode
        self._embedder, self._reranker = embedder, reranker
        self.rebuild(chunks)
        self.llm = generator if generator is not None else build_generator()
        logger.info("Pipeline ready: %d chunks, mode=%s", len(self.chunks), self.mode)

    def rebuild(self, chunks: Sequence[Union[str, Chunk]]) -> None:
        """Re-index a new corpus, reusing the already-loaded models."""
        self.chunks = as_chunks(chunks)
        self.retrieval = build_retrieval(
            self.mode, self.chunks, embedder=self._embedder, reranker=self._reranker,
            fusion=settings.fusion, alpha=settings.alpha, pool=settings.top_k_retrieve,
        )
        fp = self.retrieval.first_pass
        dense = getattr(fp, "dense", fp if hasattr(fp, "embedder") else None)
        if dense is not None:
            self._embedder = dense.embedder
        if self.retrieval.reranker is not None:
            self._reranker = self.retrieval.reranker

    def retrieve(self, question: str, timings: Optional[dict] = None) -> List[Chunk]:
        hits = self.retrieval.search(question, settings.top_k_rerank, timings=timings)
        return [h.chunk for h in hits]

    def query(self, question: str) -> dict:
        timings: dict = {}
        t0 = time.perf_counter()
        context = self.retrieve(question, timings)
        t1 = time.perf_counter()
        gen = self.llm.generate(question, context)
        timings["generate_ms"] = (time.perf_counter() - t1) * 1000
        latency_ms = (time.perf_counter() - t0) * 1000

        cited = cited_indices(gen.text, len(context))
        sources = [
            {"n": i, "url": c.url, "title": c.title, "section": c.section, "cited": i in cited}
            for i, c in enumerate(context, 1)
        ]
        logger.info("query=%r mode=%s latency_ms=%.0f", question, self.mode, latency_ms)
        return {
            "question":     question,
            "answer":       gen.text,
            "abstained":    is_abstention(gen.text),
            "sources":      sources,
            "context_used": [c.text for c in context],
            "latency_ms":   round(latency_ms, 2),
            "timings_ms":   {k: round(v, 2) for k, v in timings.items()},
            "_gen":         gen,   # raw generation (log-probs, token ids) for eval; not serialised
        }
