import pytest

from app.graph.fence_redactor import FenceRedactor, take_bare_blocks

ANSWER = (
    "Điều kiện tốt nghiệp phụ thuộc vào khoá của bạn.\n\n"
    "Bạn thuộc khoá nào?\n\n"
    '```json\n{"type": "ask_user_form", "fields": [{"field": "cohort", "label": "Khoá", '
    '"options": [{"id": "k19", "label": "K19"}, {"id": "k20", "label": "K20"}]}]}\n```\n'
    '```JSON {"type": "confirmed_metadata", "fields": {"program": "cntt"}}```'
    "\nCảm ơn bạn."
)
VISIBLE = (
    "Điều kiện tốt nghiệp phụ thuộc vào khoá của bạn.\n\nBạn thuộc khoá nào?\n\n\n\nCảm ơn bạn."
)


def _run(chunks: list[str]) -> tuple[str, FenceRedactor]:
    redactor = FenceRedactor()
    shown = "".join(redactor.feed(chunk) for chunk in chunks) + redactor.finish()
    return shown, redactor


def test_blocks_are_removed_and_captured() -> None:
    shown, redactor = _run([ANSWER])
    assert shown == VISIBLE
    assert [block["type"] for block in redactor.captured] == ["ask_user_form", "confirmed_metadata"]


@pytest.mark.parametrize("cut", range(1, len(ANSWER)))
def test_any_two_chunk_split_gives_the_same_output(cut: int) -> None:
    shown, redactor = _run([ANSWER[:cut], ANSWER[cut:]])
    assert shown == VISIBLE
    assert len(redactor.captured) == 2


def test_one_character_per_chunk() -> None:
    shown, redactor = _run(list(ANSWER))
    assert shown == VISIBLE
    assert len(redactor.captured) == 2
    assert "ask_user_form" not in shown


def test_real_code_blocks_pass_through() -> None:
    text = "Ví dụ:\n```python\nprint('```')\n```\nvà ```json\n{\"a\": 1}\n``` xong."
    for chunks in ([text], list(text)):
        shown, redactor = _run(chunks)
        assert shown == text
        assert redactor.captured == []


def test_unclosed_form_fence_is_dropped() -> None:
    text = 'Trả lời.\n```json\n{"type": "ask_user_form", "fields": []}'
    shown, redactor = _run(list(text))
    assert shown == "Trả lời.\n"
    assert redactor.captured == [{"type": "ask_user_form", "fields": []}]


def test_unclosed_other_fence_is_shown() -> None:
    text = 'Trả lời.\n```json\n{"a": 1'
    shown, _ = _run([text])
    assert shown == text


def test_comment_inside_block_still_parses() -> None:
    text = '```json\n{"type": "ask_user_form", // ghi chú\n "fields": []}\n```'
    shown, redactor = _run([text])
    assert shown == ""
    assert redactor.captured[0]["type"] == "ask_user_form"


def test_backticks_that_are_not_a_fence_are_kept() -> None:
    text = "Dùng ` hoặc `` trong văn bản, cuối câu là ``"
    for chunks in ([text], list(text)):
        assert _run(chunks)[0] == text


def test_held_back_text_is_at_most_a_fence_prefix() -> None:
    redactor = FenceRedactor()
    assert redactor.feed("abc``") == "abc"
    assert redactor.feed("x") == "``x"


def test_an_unfenced_form_is_cut_out_and_captured() -> None:
    form = (
        '{"type":"ask_user_form","fields":[{"field":"bac_he","label":"Bậc/hệ?",'
        '"kind":"choice","options":[{"id":"a","label":"Đại học {chính quy}"}]}]}'
    )
    text, blocks = take_bare_blocks(f"Mình cần thêm thông tin:\n\n{form}\n\nCảm ơn bạn.")

    assert text == "Mình cần thêm thông tin:\n\nCảm ơn bạn."
    assert [block["fields"][0]["field"] for block in blocks] == ["bac_he"]


def test_other_json_and_broken_blocks_are_left_in_the_text() -> None:
    original = 'Ví dụ {"type": "other"} và {"type": "ask_user_form", "fields": ['
    assert take_bare_blocks(original) == (original, [])
