"""SSRF guard tests — syntax + resolved-IP range + PinnedNetworkBackend pinning.

`ssrf-url-vectors.json` is vendored by hand for now at contracts/vendor/ — Task
0.8's contracts:sync tool will take over keeping it in sync with backend-java.
"""

import json
from pathlib import Path

import httpcore
import pytest

from app.core.ssrf_guard import (
    PinnedNetworkBackend,
    SsrfBlockedError,
    SsrfSyntaxError,
    is_blocked,
    validate_syntax,
)

_VECTORS = json.loads(
    (Path(__file__).resolve().parents[2] / "contracts" / "vendor" / "ssrf-url-vectors.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("vector", _VECTORS["syntax"], ids=lambda v: v["url"])
def test_syntax_vectors(vector: dict) -> None:
    if vector["accept"]:
        validate_syntax(vector["url"])
    else:
        with pytest.raises(SsrfSyntaxError):
            validate_syntax(vector["url"])


def test_trailing_dot_stripped() -> None:
    assert validate_syntax("https://api.openai.com./v1") == "api.openai.com"


def test_idna_converts_to_a_label() -> None:
    assert validate_syntax("https://münchen.de/v1") == "xn--mnchen-3ya.de"


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "169.254.169.254",
        "10.0.0.1",
        "100.64.0.1",
        "::1",
        "::ffff:127.0.0.1",
        "fd00::1",
    ],
)
def test_is_blocked_true(ip: str) -> None:
    assert is_blocked(ip) is True


@pytest.mark.parametrize("ip", ["93.184.216.34", "1.1.1.1", "2606:4700:4700::1111"])
def test_is_blocked_false(ip: str) -> None:
    assert is_blocked(ip) is False


class _FakeDelegate(httpcore.AsyncNetworkBackend):
    def __init__(self) -> None:
        self.connect_calls: list[tuple[str, int]] = []

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.connect_calls.append((host, port))
        raise RuntimeError("no real socket in this test — call recorded, that's enough")


@pytest.mark.asyncio
async def test_pinned_backend_connects_to_resolved_ip_not_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.ssrf_guard.resolve_all", lambda host, port: ["93.184.216.34"])
    delegate = _FakeDelegate()
    backend = PinnedNetworkBackend(allowlist=frozenset())
    backend._delegate = delegate

    with pytest.raises(RuntimeError):
        await backend.connect_tcp("api.openai.com", 443)

    assert delegate.connect_calls == [("93.184.216.34", 443)]


@pytest.mark.asyncio
async def test_pinned_backend_rejects_blocked_resolved_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.ssrf_guard.resolve_all", lambda host, port: ["169.254.169.254"])
    delegate = _FakeDelegate()
    backend = PinnedNetworkBackend(allowlist=frozenset())
    backend._delegate = delegate

    with pytest.raises(SsrfBlockedError):
        await backend.connect_tcp("evil.test", 443)

    assert delegate.connect_calls == []


@pytest.mark.asyncio
async def test_pinned_backend_allowlisted_host_bypasses_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.ssrf_guard.resolve_all", lambda host, port: ["10.0.5.5"])
    delegate = _FakeDelegate()
    backend = PinnedNetworkBackend(allowlist=frozenset({"internal-llm.corp"}))
    backend._delegate = delegate

    with pytest.raises(RuntimeError):
        await backend.connect_tcp("internal-llm.corp", 443)

    assert delegate.connect_calls == [("10.0.5.5", 443)]


@pytest.mark.asyncio
async def test_pinned_backend_dns_rebinding_second_lookup_never_happens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates rebinding: the resolver used for validation and the one the real backend would
    use are different callables — this proves connect_tcp is given the literal IP, so the
    backend never gets a chance to re-resolve and land on a different (rebound) address."""
    monkeypatch.setattr("app.core.ssrf_guard.resolve_all", lambda host, port: ["93.184.216.34"])
    delegate = _FakeDelegate()
    backend = PinnedNetworkBackend(allowlist=frozenset())
    backend._delegate = delegate

    with pytest.raises(RuntimeError):
        await backend.connect_tcp("rebind.test", 443)

    host_passed_to_delegate = delegate.connect_calls[0][0]
    assert host_passed_to_delegate == "93.184.216.34"
    assert host_passed_to_delegate != "rebind.test"
