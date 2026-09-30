"""Slack Incoming Webhook client for operational alerts.

Posts a JSON payload to a Slack Incoming Webhook URL
(`settings.SLACK_APIKEY_ALERT_WEBHOOK_URL`). This is operator-configured
infrastructure config, the same trust tier as `BACKEND_JAVA_BASE_URL` or
`REDIS_URL` - not a URL a service admin controls dynamically - so this
builds its own plain `httpx.AsyncClient` and does not go through the
SSRF-guarded provider HTTP client factory.

Any failure here (missing/empty webhook URL, network error, non-2xx
response) is logged and swallowed - this must never raise, since an alert
that fails to send must not take down whatever main flow triggered it. A
missing webhook URL specifically is logged as a no-op, not an error: not
every environment has Slack configured, and that's expected.

What payload to send and when to call this is decided elsewhere
(`app.core.observability.alerting`) - this client only knows how to POST one.
"""

import logging
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


async def send_slack_alert(payload: dict[str, Any]) -> None:
    """POST `payload` (a Slack Incoming Webhook JSON body - `text` plus,
    typically, `blocks`/`attachments` for the rich Block Kit card) to the
    Slack webhook URL.

    Never raises. Logs and returns on any failure, including an unconfigured
    webhook URL.
    """

    webhook_url = settings.SLACK_APIKEY_ALERT_WEBHOOK_URL
    if not webhook_url:
        logger.info("Slack alerting is not configured - skipping alert")
        return

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(webhook_url, json=payload)
    except httpx.RequestError as exc:
        logger.warning("Slack alert failed - network error: %s", exc)
        return

    if response.status_code >= 300:
        logger.warning("Slack alert failed - HTTP %s: %s", response.status_code, response.text)
        return
