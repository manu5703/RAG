"""
FastAPI service — RAG query endpoint + interpretability endpoint.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.rag_pipeline import RAGPipeline

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Sample corpus — replace with your document loader ────────────────────────

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
    pipeline = RAGPipeline(chunks=SAMPLE_CHUNKS)
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Pipeline API",
    description="Hybrid retrieval + cross-encoder reranking + Qwen generation "
                "with logit-lens / DLA faithfulness analysis.",
    version="1.0.0",
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

    model_config = {"json_schema_extra": {"example": {"question": "Where is the Eiffel Tower?"}}}


class QueryResponse(BaseModel):
    question:     str
    answer:       str
    context_used: list[str]
    latency_ms:   float


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
