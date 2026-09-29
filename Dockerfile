# syntax=docker/dockerfile:1
# Ethiopia Food Prices dashboard (cereals & tubers) - Streamlit app.
#
# Build:  docker build -t ethiopia-food-prices .
# Run:    docker run --rm -p 8501:8501 ethiopia-food-prices
# Then open http://localhost:8501

FROM python:3.12-slim AS base

# Fail fast and don't buffer output, so `docker logs` shows errors immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System packages: curl is used only by the HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer is cached across code-only changes.
# requirements-dev.txt is copied too (it just adds pytest/pytest-cov) so `docker compose run test`
# can install it without a second build stage; it is not installed here in the runtime image.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the artifacts the app reads at runtime.
# (models/experiments/ and notebooks/ are excluded via .dockerignore - not needed to serve the app.)
COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/
COPY tests/ tests/
COPY data/ data/
COPY models/ models/
COPY .streamlit/ .streamlit/
COPY pytest.ini .

# Run as a non-root user (good practice for any container that faces the network).
RUN groupadd --system app && useradd --system --gid app --home /app app \
    && chown -R app:app /app
USER app

EXPOSE 8501

# Streamlit's built-in health endpoint; the container is only "healthy" once the app has started.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/Home.py", \
            "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
