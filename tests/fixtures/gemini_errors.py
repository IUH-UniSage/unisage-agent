"""Gemini 429 error bodies as the API actually returns them - every free-tier limit hit says
"You exceeded your current quota", and only the `QuotaFailure.quotaId` names the window."""

from __future__ import annotations

from typing import Any

import google.genai.errors as google_errors
from pydantic_ai.exceptions import ModelHTTPError

PER_MINUTE_QUOTA_ID = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
PER_DAY_QUOTA_ID = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
QUOTA_MESSAGE = (
    "You exceeded your current quota, please check your plan and billing details. For more "
    "information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits."
)


def gemini_quota_body(*quota_ids: str, retry_delay: str | None = "39s") -> dict[str, Any]:
    details: list[dict[str, Any]] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
                {
                    "quotaMetric": (
                        "generativelanguage.googleapis.com/generate_content_free_tier_requests"
                    ),
                    "quotaId": quota_id,
                    "quotaDimensions": {"location": "global", "model": "gemini-3.1-flash-lite"},
                    "quotaValue": "15",
                }
                for quota_id in quota_ids
            ],
        },
        {
            "@type": "type.googleapis.com/google.rpc.Help",
            "links": [{"description": "Learn more", "url": "https://ai.google.dev/gemini-api"}],
        },
    ]
    if retry_delay is not None:
        details.append(
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}
        )
    return {
        "error": {
            "code": 429,
            "message": QUOTA_MESSAGE,
            "status": "RESOURCE_EXHAUSTED",
            "details": details,
        }
    }


def gemini_quota_http_error(*quota_ids: str, retry_delay: str | None = "39s") -> ModelHTTPError:
    """The PydanticAI-wrapped form - what the chat/extraction call sites actually catch."""

    return ModelHTTPError(
        status_code=429,
        model_name="gemini-3.1-flash-lite-preview",
        body=gemini_quota_body(*quota_ids, retry_delay=retry_delay),
    )


def gemini_quota_client_error(*quota_ids: str, retry_delay: str | None = "39s") -> Exception:
    """The raw `google-genai` SDK form."""

    return google_errors.ClientError(429, gemini_quota_body(*quota_ids, retry_delay=retry_delay))
