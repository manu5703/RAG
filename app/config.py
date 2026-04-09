"""
Centralised settings — all tuneable knobs live here.
Environment variables (or a .env file) override any default.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Models ────────────────────────────────────────────────────
    embed_model: str  = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str    = "Qwen/Qwen1.5-0.5B-Chat"

    # ── Retrieval ─────────────────────────────────────────────────
    top_k_retrieve: int = 20   # hybrid candidates
    top_k_rerank: int   = 5    # chunks sent to LLM
    alpha: float        = 0.6  # dense weight in score fusion

    # ── Generation ────────────────────────────────────────────────
    max_new_tokens: int = 256
    temperature: float  = 0.2

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── External services (RAGAS uses OpenAI for LLM-based metrics) ──
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")


settings = Settings()
