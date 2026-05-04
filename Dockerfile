# ── SA Credit Stress Monitor — Production Dockerfile ─────────────────────────
#
# Multi-stage build:
#   Stage 1 (builder) — install Python deps into a clean venv
#   Stage 2 (runtime) — slim image with only what's needed to serve
#
# Target: GCP Cloud Run (linux/amd64)
# Port:   8080 (Cloud Run default)
#
# Build:  docker build -t sa-credit-stress-monitor .
# Run:    docker run -p 8080:8080 sa-credit-stress-monitor
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements into an isolated venv
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        pydantic \
        xgboost \
        shap \
        scikit-learn \
        pandas \
        numpy \
        joblib \
        pyarrow \
        python-dotenv \
        optbinning


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the venv from builder (no compiler needed at runtime)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY src/ ./src/

# Copy model artefacts (trained in Step 2)
# In production CI/CD, these would be pulled from GCS at startup
COPY data/processed/ ./data/processed/

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Cloud Run injects PORT env var; default to 8080
ENV PORT=8080
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Health check — Cloud Run uses this to determine readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Entrypoint: uvicorn with production settings
CMD uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --loop uvloop \
    --access-log
