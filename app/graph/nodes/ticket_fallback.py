"""Node 13: `TicketFallbackNode` (T1.12) — zero-hallucination fallback.

Activates when node 11 reports `has_valid_context = False`. Deterministic,
no LLM: never lets a model guess at a regulation it has no source for.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TicketFallbackResponse:
    message: str
    ui_buttons: list[dict[str, Any]] = field(default_factory=list)


def build_ticket_fallback_response(user_query: str) -> TicketFallbackResponse:
    return TicketFallbackResponse(
        message=(
            "Hệ thống chưa tìm thấy quy định chính thức cho câu hỏi này. "
            "Bạn có thể tạo phiếu hỗ trợ để được cán bộ phòng ban liên quan giải đáp trực tiếp."
        ),
        ui_buttons=[
            {
                "label": "Tạo Ticket Hỗ Trợ Học Vụ",
                "action": "OPEN_TICKET_MODAL",
                "prefill": {
                    "subject": user_query[:200],
                    "department": "PHONG_DAOTAO",
                },
            }
        ],
    )
