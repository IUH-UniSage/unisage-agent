from app.graph.nodes.ticket_fallback import build_ticket_fallback_response


def test_ticket_fallback_response_has_message_and_ticket_button() -> None:
    response = build_ticket_fallback_response("Câu hỏi không tìm thấy trong quy chế nào cả")

    assert "chưa tìm thấy" in response.message.lower()
    assert response.ui_buttons[0]["action"] == "OPEN_TICKET_MODAL"
    assert response.ui_buttons[0]["prefill"]["subject"].startswith("Câu hỏi không tìm thấy")
