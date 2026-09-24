from app.graph.nodes.intent_routing import route_intent


def test_social_chat_ends_with_template() -> None:
    assert route_intent("social_chat") == "END_SOCIAL_CHAT"


def test_off_topic_routes_to_off_topic_node() -> None:
    assert route_intent("off_topic") == "OffTopicRejectNode"


def test_academic_intents_route_to_query_transformation() -> None:
    for intent in (
        "academic_advisory",
        "academic_procedure",
        "academic_calendar",
        "academic_document",
    ):
        assert route_intent(intent) == "QueryTransformationNode"


def test_calculation_degrades_to_query_transformation() -> None:
    # Not implemented yet - documented fallback.
    assert route_intent("academic_calculation") == "QueryTransformationNode"


def test_unknown_intent_falls_back_safely() -> None:
    assert route_intent("something_unexpected") == "QueryTransformationNode"
