"""
Centralised settings — all tuneable knobs live here.
Environment variables (or a .env file) override any default, e.g.
    RETRIEVAL_MODE=hybrid LLM_BACKEND=openai LLM_BASE_URL=http://localhost:8001/v1
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Corpus ────────────────────────────────────────────────────
    corpus_path: str = "corpus/chunks.jsonl"   # built by `python -m ingest.crawl`

    # ── Models ────────────────────────────────────────────────────
    embed_model: str  = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str    = "Qwen/Qwen2.5-1.5B-Instruct"

    # ── Generator backend ─────────────────────────────────────────
    llm_backend: str  = "hf"        # "hf" (in-process) or "openai" (any OpenAI-compatible server)
    llm_base_url: str = "http://localhost:8001/v1"
    llm_api_key: str  = ""

    # ── Retrieval ─────────────────────────────────────────────────
    retrieval_mode: str = "hybrid_rerank"   # bm25 | dense | hybrid | hybrid_rerank
    fusion: str         = "rrf"             # rrf | weighted
    alpha: float        = 0.6               # dense weight when fusion="weighted"
    top_k_retrieve: int = 20                # candidate pool for the reranker
    top_k_rerank: int   = 5                 # chunks sent to the LLM

    # ── Generation ────────────────────────────────────────────────
    max_new_tokens: int = 256
    temperature: float  = 0.0               # greedy: reproducible eval runs

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── RAGAS (optional) uses an OpenAI model as judge ────────────
    openai_api_key: str = ""


settings = Settings()
