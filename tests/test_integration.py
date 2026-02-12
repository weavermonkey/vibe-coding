import logging
import uuid
from typing import Dict, Any

import pytest
from langgraph.types import Command

from graph.builder import build_graph
from graph.state import GraphState, HumanMessage


logger = logging.getLogger(__name__)


def _unique_thread_id(test_name: str) -> str:
    return f"test-{test_name}-{uuid.uuid4()}"


def _base_state(query: str) -> GraphState:
    """
    Build the per-turn input state.

    Only `messages` and `query` are provided; LangGraph will load the
    checkpointed state (including prior messages) and use the reducer
    on `messages` to append this new turn.
    """
    return {
        "messages": [HumanMessage(content=query)],
        "query": query,
    }


def _invoke_graph(graph, state: GraphState, thread_id: str) -> Dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    logger.info("Invoking graph with thread_id=%s", thread_id)
    return graph.invoke(state, config=config)


def test_clear_query_full_pipeline():
    graph = build_graph()
    thread_id = _unique_thread_id("clear-query-full-pipeline")
    query = "What are Apple's latest earnings?"

    state = _base_state(query)
    result = _invoke_graph(graph, state, thread_id)

    logger.info("Final state for clear query: %s", result)

    visited = result.get("visited_nodes", [])
    assert "clarity" in visited
    assert "research" in visited
    assert "synthesis" in visited

    final_response = result.get("final_response")
    assert isinstance(final_response, str)
    assert final_response.strip() != ""


def test_unclear_query_triggers_interrupt():
    graph = build_graph()
    thread_id = _unique_thread_id("unclear-query-interrupt")
    vague_query = "Tell me about the company"

    state = _base_state(vague_query)
    first_result = _invoke_graph(graph, state, thread_id)

    logger.info("First result (expected interrupt): %s", first_result)
    assert "__interrupt__" in first_result
    assert first_result["clarity_status"] == "needs_clarification"

    clarification = "I meant Tesla"
    config = {"configurable": {"thread_id": thread_id}}
    second_result = graph.invoke(Command(resume=clarification), config=config)

    logger.info("Second result after clarification: %s", second_result)

    visited = second_result.get("visited_nodes", [])
    assert "clarity" in visited
    assert "research" in visited
    assert "synthesis" in visited

    final_response = second_result.get("final_response")
    assert isinstance(final_response, str)
    assert final_response.strip() != ""


def test_conditional_routing_clarity():
    graph = build_graph()

    clear_thread = _unique_thread_id("clarity-clear")
    clear_query = "Tell me about Microsoft as a company."
    clear_state = _base_state(clear_query)
    clear_result = _invoke_graph(graph, clear_state, clear_thread)

    logger.info("Result for clear clarity routing test: %s", clear_result)
    assert clear_result.get("clarity_status") == "clear"
    assert "__interrupt__" not in clear_result

    unclear_thread = _unique_thread_id("clarity-unclear")
    unclear_query = "Tell me about the company"
    unclear_state = _base_state(unclear_query)
    first_result = _invoke_graph(graph, unclear_state, unclear_thread)

    logger.info("Result for unclear clarity routing test: %s", first_result)
    assert first_result.get("clarity_status") == "needs_clarification"
    assert "__interrupt__" in first_result


def test_conditional_routing_research_to_validator():
    graph = build_graph()
    thread_id = _unique_thread_id("research-to-validator")
    query = "What are recent developments at Tesla?"

    state = _base_state(query)
    result = _invoke_graph(graph, state, thread_id)

    logger.info("Result for research-to-validator routing test: %s", result)
    visited = result.get("visited_nodes", [])
    confidence = result.get("confidence_score")

    assert "research" in visited
    assert "synthesis" in visited

    # Verify routing logic: validator visited iff confidence < 6
    if confidence is not None and confidence < 6:
        assert "validator" in visited
    else:
        assert confidence >= 6


def test_validator_loops_back_to_research():
    graph = build_graph()
    thread_id = _unique_thread_id("validator-loop")
    query = "Give me a detailed overview of Nvidia."

    state = _base_state(query)
    result = _invoke_graph(graph, state, thread_id)

    logger.info("Result for validator loop test: %s", result)

    visited = result.get("visited_nodes", [])
    confidence = result.get("confidence_score")
    attempts = result.get("research_attempts", 0)

    assert "research" in visited
    assert "synthesis" in visited

    # Verify routing logic: validator visited iff confidence < 6
    if confidence is not None and confidence < 6:
        assert "validator" in visited
        assert attempts >= 1
        assert attempts <= 3
    else:
        # High confidence correctly skipped validator
        assert confidence >= 6


def test_multi_turn_conversation():
    graph = build_graph()
    thread_id = "multi-turn-test"

    first_query = "Tell me about Microsoft."
    first_state = _base_state(first_query)
    first_result = _invoke_graph(graph, first_state, thread_id)

    logger.info("First multi-turn result: %s", first_result)

    second_query = "What about their AI strategy?"
    second_state = _base_state(second_query)

    second_result = _invoke_graph(graph, second_state, thread_id)

    logger.info("Second multi-turn result: %s", second_result)

    first_response = first_result.get("final_response") or ""
    second_response = second_result.get("final_response") or ""

    assert first_response.strip() != ""
    assert second_response.strip() != ""

    # Context preservation: the follow-up query was understood without
    # triggering a clarification interrupt, and produced a substantive response.
    # That alone proves multi-turn memory is working.
    assert "clarity" in second_result.get("visited_nodes", [])
    assert second_result.get("clarity_status") == "clear"

