# Cross-repo integration harness

Implements `todo.md`'s "Task 0.5: Khung test tích hợp cross-repo" (see
`backend-java/changes/23-09-2026-Dynamic-Model-Registry-Runtime-Failover/{plan,todo}.md`
for the full spec - `plan.md`'s "Hot-reload consistency" and "SSRF policy"
sections in particular). Brings up Postgres, Redis, Qdrant, MinIO,
`backend-java`, `api-gateway`, `unisage-agent` (2 gunicorn workers), one
Celery worker, exactly one Celery Beat, a fake OpenAI-compatible LLM
provider, and a DNS-rebinding test server, all on one fixed-subnet network,
and runs pytest against them from a `test-runner` service inside that same
network.

## Status: infra only, business logic pending backend-java

This harness is buildable and its own pieces (fake provider, DNS server,
Celery Redis split, fixtures, smoke test) are verified. **It cannot yet
verify anything at the model-registry business-logic level** because two
pieces `backend-java` needs to expose don't exist yet:

- `ModelRegistryIntegrationSeeder` - a `@Profile("integration")` bean that
  seeds `ChatModel` rows through the real service layer (so `ApiKeyConverter`
  encrypts them - never raw SQL inserts), idempotent on restart.
- `POST /internal/test/registry/reset` - also `@Profile("integration")`-only,
  wipes verification jobs and reseeds inside one transaction, 409s if a
  verification job is still `RUNNING` with an unexpired lease.

Both are specified in `todo.md`'s Task 0.5 (`ModelRegistryIntegrationSeeder`
is explicitly listed under "Files likely touched" there). **They are
intentionally not implemented in this change** - `backend-java` was out of
scope for the work that produced this harness (a different, concurrent
change was in flight there). Until they land:

- `test_model_registry_smoke.py` only asserts infra reachability (Redis
  pub/sub, Beat ticking, fake provider reachable, the reset fixture running
  twice without error) - see its module docstring.
- `conftest.py`'s `registry_reset_fn` calls the reset endpoint but treats a
  404/connection error as a warning, not a failure, so the rest of the reset
  sequence (Beat stop/start, queue purge, Redis key sweep, fake-provider
  reset) still runs and still gets exercised.
- A full `docker compose up` brings every container up, but nothing that
  needs seeded `ChatModel` data (Task 1 onward) will work until the seeder
  exists.

## Running it

Build `backend-java` and `api-gateway`'s jars first - their `Dockerfile`s
(`COPY target/*.jar app.jar`) expect one to already exist:

```sh
(cd ../../../backend-java && ./mvnw package -DskipTests)
(cd ../../../api-gateway && ./mvnw package -DskipTests)
```

Then, from this directory:

```sh
# A fresh queue/prefix namespace per run - see "Redis isolation" below.
export IT_RUN_PREFIX="it-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"

docker compose -f docker-compose.integration.yml build
docker compose -f docker-compose.integration.yml run --rm test-runner pytest -m integration
```

The report lands in the `it_reports` volume as `/reports/junit.xml` inside
the containers - `docker compose -f docker-compose.integration.yml run --rm
test-runner cat /reports/junit.xml` to read it from the host, or mount that
volume path via the debug override if you want it directly on disk.

Tear down (drops the network + named volumes, including Postgres data):

```sh
docker compose -f docker-compose.integration.yml down -v
```

## No ports published to the host

Nothing in `docker-compose.integration.yml` publishes a port to the host -
including `api-gateway`. The network has a fixed subnet
(`172.30.0.0/24`) and every service gets a static IP so `backend-java`'s
`INTERNAL_ALLOWED_CIDRS` can be scoped to exactly that subnet: `test-runner`
(inside the network) can call `/internal/**` directly, the host cannot.

Need to poke the stack manually while debugging? Use the override file,
which publishes `api-gateway` on `127.0.0.1` only, and only there:

```sh
docker compose -f docker-compose.integration.yml -f docker-compose.integration.debug.yml up -d
```

Never add a `ports:` entry to `docker-compose.integration.yml` itself for
this reason - use the debug override instead.

## The `/var/run/docker.sock` mount - risk note

`test-runner` is the **only** service, in the **only** compose file in this
repo, that mounts `/var/run/docker.sock`. It needs this so the reset fixture
(`registry_reset_fn` in `conftest.py`) can stop and restart the `celery-beat`
container between test modules (via the `docker` Python SDK, not the `docker
compose` CLI - no compose plugin is installed in the image).

Mounting the host's Docker socket into a container gives that container
root-equivalent control over the host's Docker daemon - anything running as
`test-runner` can start, stop, or inspect **any** container on the host, not
just the ones in this compose network. Acceptable here because:

- This compose file only ever runs on a developer machine or an ephemeral CI
  runner, never anywhere persistent or multi-tenant.
- `test-runner`'s own image and command are fully defined in this repo -
  nothing here executes untrusted code.
- No other compose file in this repo mounts the socket, and it should stay
  that way - if a future change needs container control from inside a
  non-integration-test container, that is a new risk decision, not an
  extension of this one.

## Redis isolation

`REDIS_URL` (DB 0) is for registry/circuit-breaker/lock/event use only, key
prefix `mr:`. Celery's own broker/backend now live on their own DBs
(`CELERY_BROKER_URL` DB 1, `CELERY_RESULT_BACKEND` DB 2) - see
`app/core/config.py` and `app/worker/celery_app.py`. Queue names are prefixed
by `CELERY_QUEUE_PREFIX` (`IT_RUN_PREFIX` env var, set per run - see
"Running it" above) so the reset fixture's `queue_purge` can only ever touch
this run's own queue, never another run's or another service's. The reset
fixture also only ever deletes `mr:*` keys, **never** `FLUSHDB` - DB 0 is the
same Redis instance Celery uses (different DB, not a different container).

## The DNS-rebinding server and TLS

`rebinding-dns` (`tests/e2e/rebinding_dns/`) answers `rebind.test`: the fake
provider's IP on the first query, a decoy "internal" IP on every query after,
TTL 0. `unisage-agent`, `celery-worker`, and `test-runner` all point their
resolver at it (compose `dns:`) so `app/core/ssrf_guard.py`'s
`PinnedNetworkBackend` gets exercised against a real resolver, not a mocked
one - see `plan.md`'s "SSRF policy".

`fake-llm-provider` also serves TLS on `:8443` for `fake-provider.test`,
using a throwaway CA + leaf cert generated at container start by
`tests/e2e/tls/generate_certs.py` (written to the `fake_provider_certs`
volume, idempotent across restarts). Verified locally: the container
generates valid cert files and uvicorn binds the TLS port successfully. A
full TLS handshake through that mounted volume from a second container
worked in most local runs but was intermittently flaky under Docker Desktop
for Windows (a `docker exec` session sometimes saw an empty `/certs` on that
platform specifically) - worth re-checking in CI/Linux before relying on it
for `test_ssrf_rebinding.py`, but nothing in the cert-generation or
TLS-serving code itself was at fault.

## What's verifiable without Docker

Everything except the actual `docker compose up` needs Docker, but most of
the logic is unit-tested and runs in the default (non-`integration`) suite:

- `tests/e2e/fake_llm_provider/test_app.py` - every admin mode, via
  `TestClient`, no network.
- `tests/e2e/rebinding_dns/test_server.py` - the resolver's flip-after-first-query
  behavior, TTL, NXDOMAIN, thread-safety, over a real (loopback) UDP socket.
- `tests/e2e/tls/generate_certs.py` - run `python -m tests.e2e.tls.generate_certs
  <dir>` directly to inspect the CA/leaf cert it produces.
- `app/worker/celery_app.py`'s queue-prefix/heartbeat wiring - imports and
  configures cleanly (`python -c "from app.worker.celery_app import celery_app"`).

None of these need `pytest -m integration` or the compose stack.

## Marker

`integration` is registered in `pyproject.toml`
(`[tool.pytest.ini_options]`) and reasserted in `conftest.py`'s
`pytest_configure` as a safety net. The default `addopts` is `-m "not
integration"`, so `pytest` (no args) never touches this directory's
integration tests; `pytest -m integration` is the only way to select them,
and it only works from inside `test-runner` (the fixtures require env vars
that only exist there - see `conftest.py`'s `_required_env`).

## Beat's schedule in this profile

Celery Beat has exactly one schedule entry today, `beat_heartbeat`
(`app/worker/celery_app.py`), which writes a timestamp to
`mr:beat:last_tick` on an interval controlled by
`CELERY_BEAT_HEARTBEAT_INTERVAL_SECONDS` - `2` in this compose file (down
from the `15`s default) so the smoke test doesn't wait long for a tick. This
task exists only to give Beat something observable to do before Task 8 (the
real hot-reload verify-poll schedule) lands; it is not itself part of the
model registry's behavior.
