# Minimal image for test_ssrf_rebinding.py's real-DNS-rebinding test - see
# ../README.md's "What's verifiable without Docker" and this repo's
# app/core/ssrf_guard.py / app/core/llm/http_client.py, which are the only
# app code this image needs. Deliberately NOT the full unisage-agent image
# (Dockerfile at the repo root) - that needs Postgres/Redis/Qdrant/MinIO
# clients and the whole `uv sync --extra dev` dependency set just to import,
# none of which this test touches. Only what build_provider_http_client_sync
# and PinnedNetworkBackendSync actually import.
FROM python:3.12-slim

WORKDIR /srv

RUN pip install --no-cache-dir \
    "httpx>=0.28.1,<0.29" \
    "httpcore>=1.0.9,<1.1" \
    "anyio>=4.0.0" \
    "certifi"

COPY app/core/ssrf_guard.py /srv/app/core/ssrf_guard.py
COPY app/core/llm/http_client.py /srv/app/core/llm/http_client.py
COPY tests/e2e/ssrf_probe/probe.py /srv/tests/e2e/ssrf_probe/probe.py
RUN touch /srv/app/__init__.py /srv/app/core/__init__.py /srv/app/core/llm/__init__.py

ENV PYTHONPATH=/srv

CMD ["python", "/srv/tests/e2e/ssrf_probe/probe.py"]
