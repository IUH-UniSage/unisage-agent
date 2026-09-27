# DNS-rebinding test server for the SSRF integration test (plan.md "SSRF
# policy") - see ../README.md. Own small image, not the agent's: only needs
# dnslib.
FROM python:3.12-slim

WORKDIR /srv

RUN pip install --no-cache-dir "dnslib>=0.9.23,<1.0"

COPY tests/e2e/rebinding_dns /srv/tests/e2e/rebinding_dns
RUN touch /srv/tests/__init__.py /srv/tests/e2e/__init__.py

ENV PYTHONPATH=/srv

# UDP 53 needs root inside the container (privileged low port) - acceptable
# here since this container exists only in the integration compose network.
CMD ["python", "-m", "tests.e2e.rebinding_dns.server"]
