from app.services.chat_stream_service import _history_from_context


def test_history_reads_what_the_assistant_turn_retrieved_with() -> None:
    history = _history_from_context(
        [
            {"role": "USER", "content": "k21", "status": "COMPLETED"},
            {
                "role": "ASSISTANT",
                "content": "Học trước Tiếng Anh 1.",
                "status": "COMPLETED",
                "metadata": {"retrieval_questions": ["Tiếng Anh 2 khóa K21?", "", 3]},
            },
            {
                "role": "ASSISTANT",
                "content": "Học phí là ...",
                "status": "COMPLETED",
                "metadata": {"calculation": {"items": []}},
            },
            {"role": "ASSISTANT", "content": "x", "status": "COMPLETED", "metadata": None},
        ]
    )

    assert [m.retrieval_questions for m in history] == [[], ["Tiếng Anh 2 khóa K21?"], [], []]
    assert [m.calculation for m in history] == [False, False, True, False]


def test_history_reads_the_retrieval_slots_by_their_registry_names() -> None:
    (message,) = _history_from_context(
        [
            {
                "role": "ASSISTANT",
                "content": "3 tín chỉ.",
                "status": "COMPLETED",
                "metadata": {
                    "retrieval_questions": ["Cơ lưu chất khóa K21 mấy tín?"],
                    "retrieval_slots": {"hoc_phan": "Cơ lưu chất", "nganh": "Cơ khí", "x": 1},
                },
            }
        ]
    )

    assert message.retrieval_slots == {"hoc_phan": "Cơ lưu chất", "nganh_hoc": "Cơ khí"}
