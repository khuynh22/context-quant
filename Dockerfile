FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    && rm -rf /var/lib/apt/lists/*

# ── uv ────────────────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# ── Install dependencies (cached layer) ──────────────────────────────────────
# Copy only dependency files first so this layer is only rebuilt when deps change
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# ── Copy project ──────────────────────────────────────────────────────────────
COPY . .

# Ensure runtime directories exist and scripts are executable
RUN mkdir -p paper_trading logs models && \
    chmod +x /app/docker/paper-trader-entrypoint.sh

# ── Default command (overridden per service in docker-compose.yml) ────────────
CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true"]
