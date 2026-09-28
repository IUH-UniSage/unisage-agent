import math

from pydantic import BaseModel


class PageResponse[T](BaseModel):
    """Mirrors backend-java's `PageResponse<T>` record exactly (`data`, `page`,
    `totalPages`, `limit`, `totalItems`) - `page` is 1-indexed, matching
    Java's `PageResponse.fromPageData` (`page.getNumber() + 1`). Used as the
    `data` payload of an `ApiResponse` for any list endpoint, e.g.
    `ApiResponse[PageResponse[list[Chunk]]]`, exactly like Java wraps a
    `Page<T>` result in both types.
    """

    data: T
    page: int
    total_pages: int
    limit: int
    total_items: int

    @classmethod
    def of(cls, data: T, *, page: int, limit: int, total_items: int) -> "PageResponse[T]":
        total_pages = math.ceil(total_items / limit) if limit > 0 else 0
        return cls(
            data=data, page=page, total_pages=total_pages, limit=limit, total_items=total_items
        )


class ApiResponse[T](BaseModel):
    """Mirrors backend-java's `ApiResponse<T>` record exactly (`code`, `message`,
    `data`, `errors`), so the frontend can parse both services' responses with
    the same `readSuccessData`/`readApiResponse` utility. `code == 1000` means
    success, matching Java's `ApiResponse.success(...)`; any other value is an
    error code from `app/core/errors/error_codes.py`.
    """

    code: int
    message: str
    data: T | None = None
    errors: dict[str, str] | None = None

    @classmethod
    def success(cls, data: T, message: str = "Successful") -> "ApiResponse[T]":
        return cls(code=1000, message=message, data=data)

    @classmethod
    def success_without_data(cls, message: str = "Successful") -> "ApiResponse[None]":
        return ApiResponse[None](code=1000, message=message)
