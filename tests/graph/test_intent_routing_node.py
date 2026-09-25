import pytest

from app.graph.nodes.intent_routing import RoutePlan, plan_route
from app.schemas.intent import ClassifiedTask, IntentClassification, RoutingMode


def _task(
    intent: str, query: str = "câu hỏi", routing_mode: RoutingMode | None = None
) -> ClassifiedTask:
    return ClassifiedTask(intent=intent, query=query, routing_mode=routing_mode)


def _plan(*tasks: ClassifiedTask) -> RoutePlan:
    return plan_route(IntentClassification(tasks=list(tasks)))


@pytest.mark.parametrize(
    ("task", "expected_end"),
    [
        (_task("social_chat"), "SOCIAL_CHAT"),
        (_task("off_topic"), "OFF_TOPIC"),
    ],
)
def test_single_non_academic_task_ends_the_turn(task: ClassifiedTask, expected_end: str) -> None:
    plan = _plan(task)

    assert plan.end == expected_end
    assert plan.calculation_tasks == []
    assert plan.advisory_tasks == []


@pytest.mark.parametrize("routing_mode", ["SINGLE", "MULTI"])
def test_advisory_task_goes_to_query_transformation_with_its_routing_mode(
    routing_mode: RoutingMode,
) -> None:
    task = _task("academic_advisory", routing_mode=routing_mode)

    plan = _plan(task)

    assert plan.end is None
    assert plan.advisory_tasks == [(task, routing_mode)]
    assert plan.calculation_tasks == []


def test_advisory_task_without_routing_mode_defaults_to_single() -> None:
    task = _task("academic_advisory", routing_mode=None)

    assert _plan(task).advisory_tasks == [(task, "SINGLE")]


def test_calculation_task_goes_to_calculation_node() -> None:
    task = _task("academic_calculation")

    plan = _plan(task)

    assert plan.end is None
    assert plan.calculation_tasks == [task]
    assert plan.advisory_tasks == []


def test_calculation_plus_procedure_question_takes_both_branches() -> None:
    calculation = _task("academic_calculation", "Tính giúp điểm GPA cho mình")
    procedure = _task("academic_advisory", "Thủ tục đăng ký tốt nghiệp là gì?", "SINGLE")

    plan = _plan(calculation, procedure)

    assert plan.end is None
    assert plan.calculation_tasks == [calculation]
    assert plan.advisory_tasks == [(procedure, "SINGLE")]


def test_two_advisory_questions_keep_their_order() -> None:
    first = _task("academic_advisory", "Học phí ngành CNTT bao nhiêu?", "SINGLE")
    second = _task("academic_advisory", "Ngành CNTT và Kế toán khác gì?", "MULTI")

    assert _plan(first, second).advisory_tasks == [(first, "SINGLE"), (second, "MULTI")]


@pytest.mark.parametrize("non_academic", ["social_chat", "off_topic", "greeting"])
def test_non_academic_tasks_are_dropped_when_an_academic_task_exists(non_academic: str) -> None:
    academic = _task("academic_advisory", "Hạn đóng học phí kỳ 2?", "SINGLE")

    plan = _plan(_task(non_academic), academic)

    assert plan.end is None
    assert plan.advisory_tasks == [(academic, "SINGLE")]
    assert plan.calculation_tasks == []


def test_social_chat_plus_off_topic_ends_as_off_topic() -> None:
    assert _plan(_task("social_chat"), _task("off_topic")).end == "OFF_TOPIC"


def test_greeting_alone_falls_back_to_one_advisory_single_task() -> None:
    plan = _plan(_task("greeting", "Chào bot"))

    assert plan.end is None
    assert len(plan.advisory_tasks) == 1
    task, mode = plan.advisory_tasks[0]
    assert (task.intent, task.query, mode) == ("academic_advisory", "Chào bot", "SINGLE")
