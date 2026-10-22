"""
FastAPI service — RAG query endpoint + interpretability endpoint.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.rag_pipeline import RAGPipeline
from app.schema import read_chunks

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Corpus ───────────────────────────────────────────────────────────────────
# Built by `python -m ingest.crawl`. Falls back to a tiny sample so the API
# still starts on a fresh checkout.

SAMPLE_CHUNKS = [
    "The Eiffel Tower is located in Paris, France.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is widely used for data science and ML development.",
    "RAG combines retrieval systems with generative language models.",
    "Docker containers package code and dependencies together.",
]

# ─── Application lifespan ─────────────────────────────────────────────────────

pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Initialising RAG pipeline …")
    if Path(settings.corpus_path).exists():
        chunks = read_chunks(settings.corpus_path)
        logger.info("Loaded %d chunks from %s", len(chunks), settings.corpus_path)
    else:
        logger.warning("%s not found; using the 5-line sample corpus. Run `python -m ingest.crawl`.",
                       settings.corpus_path)
        chunks = SAMPLE_CHUNKS
    pipeline = RAGPipeline(chunks=chunks)
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Campus RAG Assistant API",
    description="Answers University of Utah student questions (ISSS, Registrar, CHPC) "
                "with cited sources. Hybrid retrieval + cross-encoder reranking + Qwen.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Request timing middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    return response


# ─── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

    model_config = {"json_schema_extra": {"example": {"question": "How many days of unemployment are allowed on OPT?"}}}


class Source(BaseModel):
    n:       int
    url:     str
    title:   str
    section: str
    cited:   bool


class QueryResponse(BaseModel):
    question:     str
    answer:       str
    abstained:    bool
    sources:      list[Source]
    context_used: list[str]
    latency_ms:   float
    timings_ms:   dict[str, float]


class InterpretRequest(BaseModel):
    prompt:       str
    target_token: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Context:\n[1] The Eiffel Tower is in Paris.\n\nQuestion: Where is it?\n\nAnswer:",
                "target_token": " Paris",
            }
        }
    }


class InterpretResponse(BaseModel):
    logit_lens: dict
    dla:        dict
    hallucination_diagnosis: dict


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = pipeline.query(req.question)
    result.pop("_gen", None)
    return QueryResponse(**result)


@app.post("/interpretability", response_model=InterpretResponse, tags=["interpretability"])
def interpretability(req: InterpretRequest):
    """
    Run logit-lens and DLA on the shared Qwen model.
    Returns per-layer probabilities and logit attributions for `target_token`.
    """
    from interpretability.faithfulness import (
        detect_context_dropout,
        direct_logit_attribution,
        logit_lens,
    )

    if not hasattr(pipeline.llm, "tokenizer"):
        raise HTTPException(status_code=400, detail="Interpretability needs LLM_BACKEND=hf (in-process model).")
    tokenizer = pipeline.llm.tokenizer
    model     = pipeline.llm.model

    ll_result  = logit_lens(tokenizer, model, req.prompt, req.target_token)
    dla_result = direct_logit_attribution(tokenizer, model, req.prompt, req.target_token)
    diagnosis  = detect_context_dropout(ll_result)

    return InterpretResponse(
        logit_lens=ll_result,
        dla=dla_result,
        hallucination_diagnosis=diagnosis,
    )
