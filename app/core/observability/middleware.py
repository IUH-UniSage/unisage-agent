import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

logger = logging.getLogger("unisage.http")


async def request_logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Middleware logging HTTP request details and process time."""
    start_time = time.perf_counter()

    response = await call_next(request)

    process_time_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time"] = f"{process_time_ms:.2f}ms"

    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {process_time_ms:.2f}ms"
    )

    return response
