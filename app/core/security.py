import hmac

from fastapi import Header

from app.core.config import settings
from app.core.exceptions import InvalidInternalSecretException


async def verify_internal_secret(x_internal_secret: str | None = Header(default=None)) -> None:
    """Reject any request that doesn't carry the API Gateway's shared secret.

    Applied at router level to every client-facing route so this service is
    only reachable through the Gateway, never directly - the same
    shared-secret pattern already used in KLTN-Academic-Agent-AI's
    `app/security.py`.
    """

    if not x_internal_secret or not hmac.compare_digest(x_internal_secret, settings.APP_INTERNAL_SECRET_KEY):
        raise InvalidInternalSecretException()
