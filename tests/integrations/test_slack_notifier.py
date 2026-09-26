"""`send_slack_alert` — no live Slack, no real network call anywhere in this
file. Mocks `httpx.AsyncClient.post` directly since this client (unlike
`BackendJavaClient`) builds its own client with no transport injection
point."""

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core.config import settings
from app.integrations.slack_notifier import send_slack_alert


@pytest.mark.asyncio
async def test_send_slack_alert_posts_expected_json_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "SLACK_APIKEY_ALERT_WEBHOOK_URL", "https://hooks.slack.test/x")

    mock_response = httpx.Response(200, request=httpx.Request("POST", "https://hooks.slack.test/x"))
    with patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=mock_response)
    ) as mock_post:
        await send_slack_alert("something went wrong")

    args, kwargs = mock_post.call_args
    assert args[0] == "https://hooks.slack.test/x"
    assert kwargs["json"] == {"text": "something went wrong"}


@pytest.mark.asyncio
async def test_send_slack_alert_logs_and_returns_on_non_2xx(
    monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    monkeypatch.setattr(settings, "SLACK_APIKEY_ALERT_WEBHOOK_URL", "https://hooks.slack.test/x")

    mock_response = httpx.Response(
        400, text="invalid_payload", request=httpx.Request("POST", "https://hooks.slack.test/x")
    )
    with patch.object(httpx.AsyncClient, "post", new=AsyncMock(return_value=mock_response)):
        with caplog.at_level("WARNING"):
            result = await send_slack_alert("something went wrong")

    assert result is None
    assert "400" in caplog.text


@pytest.mark.asyncio
async def test_send_slack_alert_logs_and_returns_on_network_error(
    monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    monkeypatch.setattr(settings, "SLACK_APIKEY_ALERT_WEBHOOK_URL", "https://hooks.slack.test/x")

    with patch.object(
        httpx.AsyncClient,
        "post",
        new=AsyncMock(side_effect=httpx.ConnectError("connection refused")),
    ):
        with caplog.at_level("WARNING"):
            result = await send_slack_alert("something went wrong")

    assert result is None
    assert "network error" in caplog.text.lower()


@pytest.mark.asyncio
async def test_send_slack_alert_is_noop_when_webhook_url_unset(
    monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    monkeypatch.setattr(settings, "SLACK_APIKEY_ALERT_WEBHOOK_URL", "")

    with patch.object(httpx.AsyncClient, "post", new=AsyncMock()) as mock_post:
        with caplog.at_level("INFO"):
            result = await send_slack_alert("something went wrong")

    mock_post.assert_not_called()
    assert result is None
    assert "not configured" in caplog.text.lower()
