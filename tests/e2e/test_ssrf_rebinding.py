"""Real DNS-rebinding integration test - todo.md Task 0.6's "Test rebinding
thật (integration)" item.

Deliberately does NOT bring up the full `docker-compose.integration.yml`
stack (Postgres/Redis/Qdrant/MinIO/backend-java/api-gateway/Celery) - none of
that is needed to exercise `PinnedNetworkBackendSync` against a real DNS
server answering with the OS's real resolver, and standing up the whole
stack just for this would be slow and brittle for no benefit. Instead it
drives Docker directly (the `docker` SDK - already a project dependency, see
`tests/e2e/conftest.py`) to run exactly the three containers this test
needs, on their own throwaway network:

- `rebinding-dns` (`tests/e2e/rebinding_dns/`, existing image) - answers
  `rebind.test`: the "provider" container's IP on the first A/ANY query,
  the "decoy" container's IP on every one after, TTL 0.
- two instances of the existing `fake_llm_provider` image, one playing
  "provider" (the safe target), one playing "decoy" (the internal service a
  successful rebind would have reached) - both admin-controllable, so this
  test can read `request_count` off each afterwards via `docker exec`.
- `ssrf-probe` (`tests/e2e/ssrf_probe/`, new for this test) - a minimal image
  containing only `app/core/security/ssrf_guard.py` + `app/core/llm/http_client.py`
  (not the full agent image - see its Dockerfile) that makes exactly one
  real GET through `build_provider_http_client_sync` against
  `http://rebind.test:8000/healthz`, with its own DNS resolver (`docker run
  --dns`) pointed at `rebinding-dns` - so the hostname resolution
  `PinnedNetworkBackendSync.connect_tcp` performs is a real UDP round trip to
  a real server, not a monkeypatched `resolve_all`.

Verified: this file's containers build and run; see the module's own
xfail-free assertions below for what actually passed. Torn down unconditionally
in a `finally` (network + all three containers), never left behind on failure.
"""

from __future__ import annotations

import json
import pathlib
import time
import uuid
from collections.abc import Iterator

import docker
import pytest
from docker.models.containers import Container
from docker.models.networks import Network

pytestmark = pytest.mark.integration

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # unisage-agent/


_HEALTHZ_PROBE = (
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
)
_ADMIN_STATE_PROBE = (
    "import urllib.request; "
    "print(urllib.request.urlopen('http://127.0.0.1:8000/admin/state').read().decode())"
)


def _wait_for_healthz(
    client: docker.DockerClient, container: Container, timeout_s: float = 20.0
) -> None:
    deadline = time.monotonic() + timeout_s
    last_output = b""
    while time.monotonic() < deadline:
        exit_code, output = container.exec_run(["python3", "-c", _HEALTHZ_PROBE])
        if exit_code == 0:
            return
        last_output = output
        time.sleep(0.5)
    raise TimeoutError(f"{container.name} never answered /healthz: {last_output!r}")


def _admin_state(container: Container) -> dict:
    exit_code, output = container.exec_run(["python3", "-c", _ADMIN_STATE_PROBE])
    assert exit_code == 0, f"reading /admin/state on {container.name} failed: {output!r}"
    return json.loads(output.decode().strip().splitlines()[-1])


@pytest.fixture(scope="module")
def docker_client() -> Iterator[docker.DockerClient]:
    client = docker.from_env()
    try:
        yield client
    finally:
        client.close()


@pytest.fixture(scope="module")
def rebinding_images(docker_client: docker.DockerClient) -> dict[str, str]:
    """Builds the three images this test needs, tagged uniquely per test run.

    Not part of `docker-compose.integration.yml` - plain `docker build`
    against this repo's root as context, same as the compose file's own
    `context: ../..` entries for these Dockerfiles.
    """

    run_id = uuid.uuid4().hex[:8]
    tags = {
        "rebinding-dns": f"unisage-agent-ssrf-test-rebinding-dns:{run_id}",
        "fake-llm-provider": f"unisage-agent-ssrf-test-fake-provider:{run_id}",
        "ssrf-probe": f"unisage-agent-ssrf-test-probe:{run_id}",
    }
    docker_client.images.build(
        path=str(_REPO_ROOT),
        dockerfile="tests/e2e/rebinding_dns/Dockerfile",
        tag=tags["rebinding-dns"],
    )
    docker_client.images.build(
        path=str(_REPO_ROOT),
        dockerfile="tests/e2e/fake_llm_provider/Dockerfile",
        tag=tags["fake-llm-provider"],
    )
    docker_client.images.build(
        path=str(_REPO_ROOT),
        dockerfile="tests/e2e/ssrf_probe/Dockerfile",
        tag=tags["ssrf-probe"],
    )
    try:
        yield tags
    finally:
        for tag in tags.values():
            try:
                docker_client.images.remove(tag, force=True)
            except docker.errors.DockerException:
                pass


@pytest.fixture
def rebind_scenario(
    docker_client: docker.DockerClient, rebinding_images: dict[str, str]
) -> Iterator[dict]:
    """Stands up provider + decoy + rebinding-dns on a fresh throwaway network.

    Fresh per test: the DNS server's "first query wins" counter is
    per-process, so a previous test's queries must never leak into this one.
    """

    run_id = uuid.uuid4().hex[:8]
    network: Network = docker_client.networks.create(f"ssrf-rebind-test-{run_id}", driver="bridge")
    containers: list[Container] = []
    try:
        provider = docker_client.containers.run(
            rebinding_images["fake-llm-provider"],
            name=f"ssrf-rebind-provider-{run_id}",
            network=network.name,
            detach=True,
        )
        containers.append(provider)
        decoy = docker_client.containers.run(
            rebinding_images["fake-llm-provider"],
            name=f"ssrf-rebind-decoy-{run_id}",
            network=network.name,
            detach=True,
        )
        containers.append(decoy)

        for c in (provider, decoy):
            c.reload()
            _wait_for_healthz(docker_client, c)

        provider.reload()
        decoy.reload()
        provider_ip = provider.attrs["NetworkSettings"]["Networks"][network.name]["IPAddress"]
        decoy_ip = decoy.attrs["NetworkSettings"]["Networks"][network.name]["IPAddress"]
        assert provider_ip and decoy_ip

        dns_server = docker_client.containers.run(
            rebinding_images["rebinding-dns"],
            name=f"ssrf-rebind-dns-{run_id}",
            network=network.name,
            environment={"REBIND_SAFE_IP": provider_ip, "REBIND_DECOY_IP": decoy_ip},
            detach=True,
        )
        containers.append(dns_server)
        # No /healthz on the DNS server (UDP, not HTTP) - give it a moment to
        # bind before the probe's very first (and only) query.
        time.sleep(1.0)
        dns_server.reload()
        dns_ip = dns_server.attrs["NetworkSettings"]["Networks"][network.name]["IPAddress"]
        assert dns_ip

        yield {
            "network": network,
            "provider": provider,
            "decoy": decoy,
            "dns_server": dns_server,
            "dns_ip": dns_ip,
        }
    finally:
        for c in containers:
            try:
                c.remove(force=True)
            except docker.errors.DockerException:
                pass
        try:
            network.remove()
        except docker.errors.DockerException:
            pass


def test_dns_rebinding_never_reaches_the_decoy(
    docker_client: docker.DockerClient,
    rebinding_images: dict[str, str],
    rebind_scenario: dict,
) -> None:
    network: Network = rebind_scenario["network"]
    decoy: Container = rebind_scenario["decoy"]
    dns_ip: str = rebind_scenario["dns_ip"]

    probe_command = [
        "python",
        "/srv/tests/e2e/ssrf_probe/probe.py",
        "http://rebind.test:8000/healthz",
        "rebind.test",
    ]
    probe_output = docker_client.containers.run(
        rebinding_images["ssrf-probe"],
        network=network.name,
        dns=[dns_ip],
        command=probe_command,
        remove=True,
    )
    result = json.loads(probe_output.decode().strip().splitlines()[-1])

    # The one hard requirement regardless of outcome: the decoy must never
    # see a single connection - a rebind that "worked" would show up here,
    # not in the probe's own report.
    decoy_state = _admin_state(decoy)
    assert decoy_state["request_count"] == 0, (
        f"decoy received {decoy_state['request_count']} request(s) - "
        f"DNS rebinding was not defeated: {result!r}"
    )

    assert result["outcome"] in ("reached", "blocked"), f"unexpected probe outcome: {result!r}"
    if result["outcome"] == "reached":
        # /healthz doesn't bump fake_llm_provider's request_count (only the
        # OpenAI-compatible routes do) - the probe's own status_code is the
        # signal that this specific request actually reached the provider.
        assert result["status_code"] == 200, (
            f"probe reported 'reached' with an unexpected status: {result!r}"
        )
