import logging
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from agents.research import ConfidenceAssessment, ResearchAgent
from graph.state import GraphState


logger = logging.getLogger(__name__)


@patch("agents.research.ResearchAgent._get_confidence_assessor")
@patch("agents.research.ResearchAgent._get_research_model")
def test_research_returns_findings_and_confidence(
    mock_get_research_model, mock_get_confidence_assessor
) -> None:
    mock_research_model = MagicMock()
    mock_research_model.invoke.return_value = AIMessage(
        content="Tesla is doing well in Q4"
    )
    mock_get_research_model.return_value = mock_research_model

    mock_confidence = MagicMock()
    mock_confidence.invoke.return_value = ConfidenceAssessment(
        confidence_score=8.5,
        reasoning="Comprehensive",
    )
    mock_get_confidence_assessor.return_value = mock_confidence

    state: GraphState = {
        "messages": [],
        "query": "Tell me about Tesla",
        "company_name": "Tesla",
    }

    agent = ResearchAgent()
    result = agent.run(state)
    logger.info("Research agent result: %s", result)

    findings = result.get("research_findings")
    assert isinstance(findings, str)
    assert findings.strip() != ""
    assert result.get("confidence_score") == 8.5
    assert "research" in result.get("visited_nodes", [])

    messages = result.get("messages", [])
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)


@patch("agents.research.ResearchAgent._get_confidence_assessor")
@patch("agents.research.ResearchAgent._get_research_model")
def test_research_low_confidence(
    mock_get_research_model, mock_get_confidence_assessor
) -> None:
    mock_research_model = MagicMock()
    mock_research_model.invoke.return_value = AIMessage(
        content="Tesla is doing well in Q4"
    )
    mock_get_research_model.return_value = mock_research_model

    mock_confidence = MagicMock()
    mock_confidence.invoke.return_value = ConfidenceAssessment(
        confidence_score=3.0,
        reasoning="Limited coverage",
    )
    mock_get_confidence_assessor.return_value = mock_confidence

    state: GraphState = {
        "messages": [],
        "query": "Tell me about Tesla",
        "company_name": "Tesla",
    }

    agent = ResearchAgent()
    result = agent.run(state)
    logger.info("Research agent low-confidence result: %s", result)

    assert result.get("confidence_score") == 3.0
    findings = result.get("research_findings")
    assert isinstance(findings, str)
    assert findings.strip() != ""

