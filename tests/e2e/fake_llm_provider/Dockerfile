# Fake OpenAI-compatible LLM provider for the integration harness only - see
# ../README.md. Deliberately its own small image (fastapi+uvicorn+cryptography),
# not the full unisage-agent image: it has none of the agent's own dependencies.
FROM python:3.12-slim

WORKDIR /srv

RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "cryptography>=43.0.0"

COPY tests/e2e/fake_llm_provider /srv/tests/e2e/fake_llm_provider
COPY tests/e2e/tls /srv/tests/e2e/tls
RUN touch /srv/tests/__init__.py /srv/tests/e2e/__init__.py

ENV PYTHONPATH=/srv
EXPOSE 8000 8443

CMD ["python", "-m", "tests.e2e.fake_llm_provider.run"]
