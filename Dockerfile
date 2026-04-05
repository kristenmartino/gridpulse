# ─── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system deps for Prophet (C/C++ compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Energy Forecast Dashboard"
LABEL description="Weather-aware energy demand forecasting for 8 US BAs"

WORKDIR /app

# Copy only the installed packages from builder (no gcc/g++ in runtime)
COPY --from=builder /install /usr/local

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy application code
COPY . .

# Create directories for cache and models, owned by appuser
RUN mkdir -p /app/trained_models /app/cache && \
    chown -R appuser:appuser /app

USER appuser

# Environment defaults
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CACHE_DB_PATH=/app/cache/cache.db \
    MODEL_DIR=/app/trained_models \
    DASH_DEBUG=false \
    PRECOMPUTE_ENABLED=true \
    PRECOMPUTE_ALL_REGIONS=true

EXPOSE 8080

# Health check (Cloud Run also has its own, this is for Docker-native)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Gunicorn with:
# - 2 gthread workers (Cloud Run 2Gi memory budget)
# - 2 threads per worker: one serves requests, one runs precompute
# - 300s timeout (long-running model training callbacks)
# - Access log to stdout for Cloud Run
# NOTE: --preload is intentionally omitted. Workers load the app module
# after gunicorn binds to PORT, allowing Cloud Run's startup probe to
# connect immediately. Precompute runs in a background thread so workers
# can serve /health while caches warm up.
CMD ["gunicorn", "app:server", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "2", \
     "--threads", "2", \
     "--worker-class", "gthread", \
     "--timeout", "300", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
