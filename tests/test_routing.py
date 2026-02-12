"""
Deterministic unit tests for the routing functions in `graph.routing`.

These tests exercise all branches of the pure routing functions without
performing any external API calls or invoking the full LangGraph.
"""

from graph.routing import (
    route_after_clarity,
    route_after_research,
    route_after_validation,
)


def test_route_after_clarity_needs_clarification():
    state = {"clarity_status": "needs_clarification"}
    assert route_after_clarity(state) == "interrupt"


def test_route_after_clarity_clear():
    state = {"clarity_status": "clear"}
    assert route_after_clarity(state) == "research"


def test_route_after_clarity_none():
    state = {"clarity_status": None}
    assert route_after_clarity(state) == "research"


def test_route_after_research_confidence_none():
    state = {"confidence_score": None}
    assert route_after_research(state) == "validator"


def test_route_after_research_confidence_low_values():
    assert route_after_research({"confidence_score": 3.0}) == "validator"
    assert route_after_research({"confidence_score": 5.9}) == "validator"


def test_route_after_research_confidence_high_values():
    assert route_after_research({"confidence_score": 6.0}) == "synthesis"
    assert route_after_research({"confidence_score": 9.0}) == "synthesis"


def test_route_after_validation_insufficient_under_cap():
    state = {"validation_result": "insufficient", "research_attempts": 0}
    assert route_after_validation(state) == "research"

    state = {"validation_result": "insufficient", "research_attempts": 2}
    assert route_after_validation(state) == "research"


def test_route_after_validation_insufficient_at_cap_or_sufficient():
    state = {"validation_result": "insufficient", "research_attempts": 3}
    assert route_after_validation(state) == "synthesis"

    state = {"validation_result": "sufficient", "research_attempts": 1}
    assert route_after_validation(state) == "synthesis"

    state = {"validation_result": None, "research_attempts": 0}
    assert route_after_validation(state) == "synthesis"

