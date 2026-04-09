"""
RAG Pipeline — Hybrid Retrieval (Dense + BM25) + Cross-Encoder Reranking + Qwen generation.
"""

import logging
import time
from typing import List, Tuple

import numpy as np
import torch
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings

logger = logging.getLogger(__name__)


# ─── Hybrid Index ─────────────────────────────────────────────────────────────

class HybridIndex:
    def __init__(self):
        logger.info("Loading embedding model: %s", settings.embed_model)
        self.embedder    = SentenceTransformer(settings.embed_model)
        self.chunks: List[str] = []
        self.faiss_index = None
        self.bm25        = None

    def build(self, chunks: List[str]) -> None:
        if not chunks:
            raise ValueError("Cannot build index over an empty corpus.")
        self.chunks = chunks

        # Dense index — cosine similarity via inner product on L2-normed vectors
        embeddings = self.embedder.encode(
            chunks, show_progress_bar=True, convert_to_numpy=True, batch_size=64
        )
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings.astype("float32"))

        # Sparse index
        tokenized = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        logger.info("Index built over %d chunks.", len(chunks))

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        top_k = top_k or settings.top_k_retrieve

        # ── Dense scores ──────────────────────────────────────────
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        dense_scores, dense_ids = self.faiss_index.search(q_emb.astype("float32"), top_k)
        dense_scores = dense_scores[0]
        dense_ids    = dense_ids[0].tolist()

        # O(1) lookup instead of O(n) .index() call
        dense_score_map = {idx: float(score) for idx, score in zip(dense_ids, dense_scores)}

        # ── BM25 scores (normalised 0-1) ──────────────────────────
        bm25_raw  = np.array(self.bm25.get_scores(query.lower().split()))
        bm25_max  = bm25_raw.max() or 1.0
        bm25_norm = bm25_raw / bm25_max

        # ── Fuse candidates from both retrievers ──────────────────
        sparse_top = set(np.argsort(bm25_norm)[-top_k:].tolist())
        candidate_ids = list(set(dense_ids) | sparse_top)

        fused = [
            (idx, settings.alpha * dense_score_map.get(idx, 0.0)
                  + (1 - settings.alpha) * float(bm25_norm[idx]))
            for idx in candidate_ids
        ]
        fused.sort(key=lambda x: x[1], reverse=True)
        return [(self.chunks[i], score) for i, score in fused[:top_k]]


# ─── Cross-Encoder Reranker ───────────────────────────────────────────────────

class Reranker:
    def __init__(self):
        logger.info("Loading reranker: %s", settings.rerank_model)
        self.model = CrossEncoder(settings.rerank_model)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        top_k: int = None,
    ) -> List[str]:
        top_k  = top_k or settings.top_k_rerank
        pairs  = [(query, chunk) for chunk, _ in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip([c for c, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [chunk for chunk, _ in ranked[:top_k]]


# ─── Qwen Generator ───────────────────────────────────────────────────────────

class QwenGenerator:
    """
    Lazy-loading Qwen wrapper.
    The tokenizer and model are NOT loaded at construction time — they load
    on the first call to generate() or when .tokenizer / .model are accessed.
    This keeps ~2 GB of RAM free during app startup and PDF indexing.
    """

    def __init__(self):
        self._tokenizer = None
        self._model     = None

    def _load(self):
        if self._model is not None:
            return
        logger.info("Loading LLM: %s (first query)", settings.llm_model)
        self._tokenizer = AutoTokenizer.from_pretrained(
            settings.llm_model, trust_remote_code=True
        )
        if torch.cuda.is_available():
            load_kwargs = dict(torch_dtype=torch.float16, device_map="auto")
        else:
            load_kwargs = dict(torch_dtype=torch.float32)

        self._model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model,
            trust_remote_code=True,
            **load_kwargs,
        )
        self._model.eval()
        logger.info("LLM loaded on device: %s", next(self._model.parameters()).device)

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    @property
    def model(self):
        self._load()
        return self._model

    def generate(self, query: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))

        # Qwen1.5-Chat requires the chat template — raw prompts cause empty output
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer using ONLY the numbered "
                    "context provided. If the context does not contain the answer, "
                    "say 'I don't know'."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                do_sample=settings.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens — not the prompt
        new_token_ids = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


# ─── Full Pipeline ────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, chunks: List[str]):
        self.index    = HybridIndex()
        self.reranker = Reranker()
        self.llm      = QwenGenerator()
        self.index.build(chunks)

    def query(self, question: str) -> dict:
        t0 = time.perf_counter()

        candidates   = self.index.retrieve(question)
        final_chunks = self.reranker.rerank(question, candidates)
        answer       = self.llm.generate(question, final_chunks)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("query=%r latency_ms=%.1f", question, latency_ms)
        return {
            "question":     question,
            "answer":       answer,
            "context_used": final_chunks,
            "latency_ms":   latency_ms,
        }
