import logging
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.synthesis import SynthesisAgent
from graph.state import GraphState


logger = logging.getLogger(__name__)


@patch("agents.synthesis.SynthesisAgent._get_model")
def test_synthesis_produces_response(mock_get_model) -> None:
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(
        content="Here is a summary of Tesla's recent activity."
    )
    mock_get_model.return_value = mock_model

    state: GraphState = {
        "messages": [HumanMessage(content="Tell me about Tesla")],
        "query": "Tell me about Tesla",
        "research_findings": "Tesla data here",
    }

    agent = SynthesisAgent()
    result = agent.run(state)
    logger.info("Synthesis agent result: %s", result)

    final_response = result.get("final_response")
    assert isinstance(final_response, str)
    assert final_response.strip() != ""
    assert "synthesis" in result.get("visited_nodes", [])

    messages = result.get("messages", [])
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)


@patch("agents.synthesis.SynthesisAgent._get_model")
def test_synthesis_empty_response_raises(mock_get_model) -> None:
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(content="")
    mock_get_model.return_value = mock_model

    state: GraphState = {
        "messages": [HumanMessage(content="Tell me about Tesla")],
        "query": "Tell me about Tesla",
        "research_findings": "Tesla data here",
    }

    agent = SynthesisAgent()
    with pytest.raises(RuntimeError):
        agent.run(state)

