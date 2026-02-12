import logging
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from config import get_gemini_api_key
from graph.state import GraphState


logger = logging.getLogger(__name__)


class ClarityResult(BaseModel):
    clarity_status: str
    company_name: Optional[str] = None
    clarification_question: Optional[str] = None


_clarity_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=get_gemini_api_key(),
    temperature=0.0,
)
_clarity_model_structured = _clarity_model.with_structured_output(ClarityResult)


def _build_clarity_messages(messages: list[BaseMessage], query: str) -> list[BaseMessage]:
    system_message = SystemMessage(
        content=(
            "You are a clarity assessment agent.\n"
            "Given the conversation so far and the latest user query, decide whether "
            "the query is clear and specific enough to begin company-focused research.\n"
            "You MUST respond using the structured schema provided to you.\n"
            "- If the query clearly refers to a specific company (e.g., Apple, Tesla, Microsoft), "
            'set clarity_status to \"clear\" and extract company_name.\n'
            "- If the query is vague (e.g., 'Tell me about the company') or does not clearly "
            "identify a single company, set clarity_status to \"needs_clarification\" and "
            "provide a short follow-up question asking the user to clarify what company "
            "or topic they mean.\n"
        )
    )
    latest_query_message = HumanMessage(
        content=f"Latest user query: {query}",
    )
    return [system_message] + messages + [latest_query_message]


def run_clarity_agent(state: GraphState) -> GraphState:
    """
    Clarity Agent:
    - Determines whether the query is clear or needs clarification.
    - Attempts to extract a company name when present.
    - Optionally sets a clarification question for the human-in-the-loop step.
    """
    logger.info("Running clarity agent.")
    messages = list(state.get("messages", []))
    query = state.get("query") or ""

    prompt_messages = _build_clarity_messages(messages, query)
    logger.debug("Clarity agent prompt messages: %s", prompt_messages)

    result: ClarityResult = _clarity_model_structured.invoke(prompt_messages)
    logger.info(
        "Clarity agent result: clarity_status=%s, company_name=%s",
        result.clarity_status,
        result.company_name,
    )

    # Per-turn resets FIRST, then clarity results overwrite the relevant fields.
    updated_state: GraphState = {
        # Resets for this turn
        "research_attempts": 0,
        "validation_result": None,
        "confidence_score": None,
        "research_findings": None,
        "final_response": None,
        # Clarity results (take precedence)
        "query": query,
        "company_name": result.company_name,
        "clarity_status": result.clarity_status,
        "clarification_question": result.clarification_question,
        # Delta for reducer
        "visited_nodes": ["clarity"],
    }

    return updated_state


__all__ = ["run_clarity_agent"]

