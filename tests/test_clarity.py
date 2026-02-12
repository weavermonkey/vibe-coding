import logging
from unittest.mock import MagicMock, patch

from graph.state import GraphState, HumanMessage
from agents.clarity import ClarityAgent, ClarityResult


logger = logging.getLogger(__name__)


@patch("agents.clarity.ClarityAgent._get_model_structured")
def test_clear_query_extracts_company(mock_get_model) -> None:
    mock_model = MagicMock()
    mock_model.invoke.return_value = ClarityResult(
        clarity_status="clear",
        company_name="Tesla",
        clarification_question=None,
    )
    mock_get_model.return_value = mock_model

    state: GraphState = {
        "messages": [HumanMessage(content="Tell me about Tesla")],
        "query": "Tell me about Tesla",
    }

    agent = ClarityAgent()
    result = agent.run(state)
    logger.info("Clarity result for clear query: %s", result)

    assert result.get("clarity_status") == "clear"
    assert result.get("company_name") == "Tesla"
    assert "clarity" in result.get("visited_nodes", [])
    assert result.get("research_attempts") == 0


@patch("agents.clarity.ClarityAgent._get_model_structured")
def test_vague_query_needs_clarification(mock_get_model) -> None:
    mock_model = MagicMock()
    mock_model.invoke.return_value = ClarityResult(
        clarity_status="needs_clarification",
        company_name=None,
        clarification_question="Which company do you mean?",
    )
    mock_get_model.return_value = mock_model

    state: GraphState = {
        "messages": [HumanMessage(content="Tell me about the company")],
        "query": "Tell me about the company",
    }

    agent = ClarityAgent()
    result = agent.run(state)
    logger.info("Clarity result for vague query: %s", result)

    assert result.get("clarity_status") == "needs_clarification"
    assert result.get("company_name") is None
    assert result.get("clarification_question") is not None


@patch("agents.clarity.ClarityAgent._get_model_structured")
def test_clarity_resets_per_turn_state(mock_get_model) -> None:
    mock_model = MagicMock()
    mock_model.invoke.return_value = ClarityResult(
        clarity_status="clear",
        company_name="Apple",
        clarification_question=None,
    )
    mock_get_model.return_value = mock_model

    state: GraphState = {
        "messages": [HumanMessage(content="Tell me about Apple")],
        "query": "Tell me about Apple",
        # Stale values that should be reset by the clarity agent.
        "research_attempts": 2,
        "confidence_score": 7.0,
        "validation_result": "sufficient",
        "research_findings": "old stuff",
        "final_response": "old response",
    }

    agent = ClarityAgent()
    result = agent.run(state)
    logger.info("Clarity result with stale state: %s", result)

    assert result.get("clarity_status") == "clear"
    assert result.get("company_name") == "Apple"
    # Per-turn fields should be reset by clarity.
    assert result.get("research_attempts") == 0
    assert result.get("confidence_score") is None
    assert result.get("validation_result") is None
    assert result.get("research_findings") is None
    assert result.get("final_response") is None

