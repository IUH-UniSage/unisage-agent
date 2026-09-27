# Built and used only by the integration harness (tests/e2e/docker-compose.integration.yml)
# today - the agent has no production Dockerfile of its own yet. Same image
# serves four roles in that compose file (gunicorn app, celery worker, celery
# beat, test-runner) - only the command differs, so one image keeps them from
# drifting apart.
FROM python:3.12-slim AS base

WORKDIR /app

# uv installs fast from the lockfile - see https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock ./
# Includes the `dev` extra (pytest, dnslib, cryptography, ...) - the
# test-runner service needs it and re-deriving a slimmer image just for the
# other three roles isn't worth the duplication for a test-only image.
RUN uv sync --frozen --extra dev

# Bakes the cl100k_base BPE file into the image at build time (this stage has
# normal internet access) so nothing at container runtime needs to reach
# openaipublic.blob.core.windows.net - the integration harness's network is
# intentionally DNS/egress-restricted (SSRF guard, rebinding-dns resolver),
# and app/rag/chunking/semantic.py loads this encoding at import time, which
# would otherwise fail every role built from this image (gunicorn, celery
# worker/beat, test-runner) before it ever got to serve a request.
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken_cache
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY app ./app
COPY tests ./tests
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "app.main:app"]
