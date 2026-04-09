# ── Stage 1: dependency layer (cached separately from app code) ────────────────
FROM python:3.10-slim AS deps

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ─────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed packages from the deps stage
COPY --from=deps /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source (respects .dockerignore)
COPY app/         ./app/
COPY interpretability/ ./interpretability/

EXPOSE 8000

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
