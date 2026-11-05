"""A day/hour count whose marks only a document states is not asked of the student
(04/11/2026, q0055): the extractor sometimes picks `date_span` anyway, and the node
defers the task to the advisory answer, which reads the dates and computes in code."""

import pytest

from app.graph.nodes.calculation import CalculationDeps, Deferred, NeedsInput, plan_calculation
from tests.graph.test_profile_fields import _scripted


@pytest.mark.asyncio
async def test_a_date_span_without_dates_in_the_question_is_deferred() -> None:
    models, _ = _scripted({"formula_id": "date_span", "params": {}})
    deps = CalculationDeps(models=models, confirmed_metadata={})
    question = (
        "Theo hướng dẫn giao nhận đề cương luận văn thạc sĩ đợt 2 năm học 2018-2019, học viên"
        " có tổng cộng bao nhiêu ngày để nộp đề cương (tính cả ngày đầu và ngày cuối)?"
    )
    planned = await plan_calculation("T1", question, deps)
    assert isinstance(planned, Deferred)


@pytest.mark.asyncio
async def test_a_date_span_the_student_started_is_still_planned() -> None:
    # One date written: the other is asked, as before.
    models, _ = _scripted({"formula_id": "date_span", "params": {"tu_ngay": "13/08/2026"}})
    deps = CalculationDeps(models=models, confirmed_metadata={})
    planned = await plan_calculation("T1", "từ 13/08/2026 đến hạn chót là bao nhiêu ngày?", deps)
    assert not isinstance(planned, Deferred)
    assert not isinstance(planned, NeedsInput)


@pytest.mark.asyncio
async def test_a_question_asking_for_a_date_is_not_a_day_count() -> None:
    # Eval 04/11/2026 (d0335): "chậm nhất phải công bố vào ngày nào" was asked back.
    models, _ = _scripted({"formula_id": "date_span", "params": {"den_ngay": "20/6/2018"}})
    deps = CalculationDeps(models=models, confirmed_metadata={})
    question = (
        "Công bố chậm nhất 20 ngày trước ngày bảo vệ. Nếu bảo vệ ngày 20/6/2018 thì chậm nhất"
        " phải công bố vào ngày nào?"
    )
    assert isinstance(await plan_calculation("T1", question, deps), Deferred)
