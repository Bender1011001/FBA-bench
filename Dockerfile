# FBA-Bench Dockerfile
# Production backend image for FastAPI API (uvicorn)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies required for building some wheels and healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip tooling early
RUN python -m pip install --upgrade pip setuptools wheel

# Copy project files
COPY . .

# Install project with PEP 517 using poetry-core build backend
# This installs runtime dependencies from pyproject.toml
RUN pip install --no-cache-dir .

# Expose API port
EXPOSE 8000

# Health check for FastAPI app
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

# Start FastAPI using uvicorn
CMD ["uvicorn", "fba_bench_api.main:app", "--host", "0.0.0.0", "--port", "8000"]