import logging
from unittest.mock import MagicMock, patch

from agents.validator import ValidationAssessment, ValidatorAgent
from graph.state import GraphState


logger = logging.getLogger(__name__)


@patch("agents.validator.ValidatorAgent._get_model_structured")
def test_validator_sufficient(mock_get_structured) -> None:
    mock_model = MagicMock()
    mock_model.invoke.return_value = ValidationAssessment(
        validation_result="sufficient",
        critique="Good",
        suggestions="",
    )
    mock_get_structured.return_value = mock_model

    state: GraphState = {
        "messages": [],
        "query": "Tesla query",
        "research_findings": "findings",
        "research_attempts": 0,
    }

    agent = ValidatorAgent()
    result = agent.run(state)
    logger.info("Validator sufficient result: %s", result)

    assert result.get("validation_result") == "sufficient"
    assert result.get("research_attempts") == 1
    assert "validator" in result.get("visited_nodes", [])


@patch("agents.validator.ValidatorAgent._get_model_structured")
def test_validator_insufficient_increments_attempts(mock_get_structured) -> None:
    mock_model = MagicMock()
    mock_model.invoke.return_value = ValidationAssessment(
        validation_result="insufficient",
        critique="Missing financials",
        suggestions="Add revenue data",
    )
    mock_get_structured.return_value = mock_model

    state: GraphState = {
        "messages": [],
        "query": "Tesla query",
        "research_findings": "findings",
        "research_attempts": 1,
    }

    agent = ValidatorAgent()
    result = agent.run(state)
    logger.info("Validator insufficient result: %s", result)

    assert result.get("validation_result") == "insufficient"
    assert result.get("research_attempts") == 2


@patch("agents.validator.ValidatorAgent._get_model_structured")
def test_validator_attempts_accumulate(mock_get_structured) -> None:
    mock_model = MagicMock()
    mock_model.invoke.return_value = ValidationAssessment(
        validation_result="insufficient",
        critique="Missing financials",
        suggestions="Add revenue data",
    )
    mock_get_structured.return_value = mock_model

    base_state: GraphState = {
        "messages": [],
        "query": "Tesla query",
        "research_findings": "findings",
        "research_attempts": 0,
    }

    agent = ValidatorAgent()
    first_result = agent.run(base_state)
    logger.info("First validator result: %s", first_result)
    assert first_result.get("research_attempts") == 1

    second_input: GraphState = {
        "messages": [],
        "query": base_state["query"],
        "research_findings": base_state["research_findings"],
        "research_attempts": first_result["research_attempts"],
    }

    second_result = agent.run(second_input)
    logger.info("Second validator result: %s", second_result)

    assert second_result.get("research_attempts") == 2

