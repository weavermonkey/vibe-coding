import logging
import uuid
from typing import Any, Dict

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


@pytest.mark.integration
def test_four_turn_context_switching() -> None:
    graph = build_graph()
    thread_id = _unique_thread_id("four-turn-context-switch")

    # Turn 1: Infosys
    first_query = "Tell me about Infosys"
    first_state = _base_state(first_query)
    first_result = _invoke_graph(graph, first_state, thread_id)
    logger.info("Turn 1 result: %s", first_result)

    assert first_result.get("clarity_status") == "clear"
    assert first_result.get("company_name") is not None
    first_response = (first_result.get("final_response") or "").strip()
    assert first_response != ""

    # Turn 2: Switch to TCS
    second_query = "Now tell me about TCS"
    second_state = _base_state(second_query)
    second_result = _invoke_graph(graph, second_state, thread_id)
    logger.info("Turn 2 result: %s", second_result)

    assert second_result.get("clarity_status") == "clear"
    second_response = (second_result.get("final_response") or "").strip()
    assert second_response != ""

    # Turn 3: Pronoun reference to most recent company (TCS)
    third_query = "What were their last earnings?"
    third_state = _base_state(third_query)
    third_result = _invoke_graph(graph, third_state, thread_id)
    logger.info("Turn 3 result: %s", third_result)

    assert third_result.get("clarity_status") == "clear"
    third_response = (third_result.get("final_response") or "").strip()
    assert third_response != ""

    # Turn 4: Reference to the other company (Infosys)
    fourth_query = "How about the other one?"
    fourth_state = _base_state(fourth_query)
    fourth_result = _invoke_graph(graph, fourth_state, thread_id)
    logger.info("Turn 4 result: %s", fourth_result)

    assert fourth_result.get("clarity_status") == "clear"
    fourth_response = (fourth_result.get("final_response") or "").strip()
    assert fourth_response != ""


@pytest.mark.integration
def test_interrupt_then_followup() -> None:
    graph = build_graph()
    thread_id = _unique_thread_id("interrupt-then-followup")

    # Turn 1, vague query should trigger interrupt
    vague_query = "Tell me about the company"
    first_state = _base_state(vague_query)
    first_result = _invoke_graph(graph, first_state, thread_id)
    logger.info("First interrupt test result: %s", first_result)

    assert "__interrupt__" in first_result
    assert first_result.get("clarity_status") == "needs_clarification"

    # Resume with clarification pointing to Infosys
    clarification = "I meant Infosys"
    config = {"configurable": {"thread_id": thread_id}}
    second_result = graph.invoke(Command(resume=clarification), config=config)
    logger.info("Second interrupt test result after clarification: %s", second_result)

    assert second_result.get("clarity_status") == "clear"
    second_response = (second_result.get("final_response") or "").strip()
    assert second_response != ""

    # Turn 2: follow-up pronoun should resolve to Infosys without another interrupt
    followup_query = "What about their competitors?"
    followup_state = _base_state(followup_query)
    followup_result = _invoke_graph(graph, followup_state, thread_id)
    logger.info("Follow-up turn result: %s", followup_result)

    assert followup_result.get("clarity_status") == "clear"
    followup_response = (followup_result.get("final_response") or "").strip()
    assert followup_response != ""
    assert "__interrupt__" not in followup_result

