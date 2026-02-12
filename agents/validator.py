import logging
from typing import Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import get_gemini_api_key
from graph.state import GraphState


logger = logging.getLogger(__name__)


_validator_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=get_gemini_api_key(),
    temperature=0.1,
)


class ValidationAssessment(BaseModel):
    validation_result: Literal["sufficient", "insufficient"]
    critique: str = Field(description="Brief critique of the research quality")
    suggestions: str = Field(
        description="Suggestions for improvement if validation is insufficient"
    )


_validator_structured = _validator_model.with_structured_output(ValidationAssessment)


def _build_validator_messages(
    messages: list[BaseMessage],
    query: str,
    research_findings: Optional[str],
) -> list[BaseMessage]:
    system_message = SystemMessage(
        content=(
            "You are a research validator.\n"
            "Given the original user query and the research findings produced by a "
            "research agent, assess whether the research is sufficiently thorough, "
            "accurate, and directly relevant to the query.\n"
            "Respond with a short critique and suggestions for improvement if any.\n"
            "Your textual critique will be logged for observability but the routing "
            "logic will be handled by the system."
        )
    )
    context = f"User query: {query}\n\nResearch findings:\n{research_findings or '<<missing>>'}"
    analysis_request = AIMessage(
        content=context,
    )
    return [system_message] + messages + [analysis_request]


def run_validator_agent(state: GraphState) -> GraphState:
    """
    Validator Agent:
    - Uses Gemini to review research quality and completeness against the query.
    - Sets `validation_result` to \"sufficient\" or \"insufficient\" using structured
      LLM output.
    - Increments `research_attempts`.
    """
    logger.info("Running validator agent.")
    messages = list(state.get("messages", []))
    query = state.get("query") or ""
    research_findings = state.get("research_findings") or ""

    prompt_messages = _build_validator_messages(messages, query, research_findings)
    logger.debug("Validator prompt messages: %s", prompt_messages)

    assessment = _validator_structured.invoke(prompt_messages)
    logger.info(
        "Validator assessment: result=%s, critique=%s",
        assessment.validation_result,
        assessment.critique,
    )

    attempts = state.get("research_attempts", 0) + 1

    critique_message = AIMessage(
        content=f"Critique: {assessment.critique}\n\nSuggestions: {assessment.suggestions}"
    )

    updated_state: GraphState = {
        # Append the critique as a new message using the reducer.
        "messages": [critique_message],
        "validation_result": assessment.validation_result,
        "research_attempts": attempts,
        "visited_nodes": ["validator"],
    }

    return updated_state


__all__ = ["run_validator_agent"]

