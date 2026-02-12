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


_clarity_model = None
_clarity_model_structured = None


def _get_clarity_model() -> ChatGoogleGenerativeAI:
    global _clarity_model
    if _clarity_model is None:
        _clarity_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=get_gemini_api_key(),
            temperature=0.0,
        )
    return _clarity_model


def _get_clarity_model_structured():
    global _clarity_model_structured
    if _clarity_model_structured is None:
        _clarity_model_structured = _get_clarity_model().with_structured_output(
            ClarityResult
        )
    return _clarity_model_structured


def _build_clarity_messages(messages: list[BaseMessage], query: str) -> list[BaseMessage]:
    system_message = SystemMessage(
        content=(
            "You are a clarity assessment agent.\n"
            "Given the full conversation history so far and the latest user query, decide whether "
            "the query is clear and specific enough to begin company-focused research.\n"
            "You MUST respond using the structured schema provided to you.\n"
            "\n"
            "Clarity rules:\n"
            "- Always use the entire conversation history (not just the latest query) to resolve what company "
            "the user is talking about.\n"
            "- If a specific company name is explicitly mentioned (e.g., Apple, Tesla, Microsoft), set "
            'clarity_status to \"clear\" and extract company_name.\n'
            "- If previous turns clearly established one or more specific companies, treat follow-up references "
            "that use pronouns or generic phrases (e.g., \"they\", \"their\", \"the company\", \"that company\", "
            "\"the other one\") as referring to those companies, even if the latest query does not repeat the "
            "company name.\n"
            "- When multiple companies have been discussed, default generic references like \"they\" or "
            "\"their\" to the most recently discussed company that makes sense in context.\n"
            "- Phrases like \"the other one\" should refer to the other clearly discussed company (for example, "
            "if the conversation first talked about Infosys and then TCS, \"the other one\" refers back to "
            "Infosys).\n"
            "- Only set clarity_status to \"needs_clarification\" if, after carefully checking the entire "
            "conversation history, you still cannot confidently resolve the company being referred to, or if "
            "no specific company has been mentioned at all (e.g., the very first query is just "
            "\"Tell me about the company\"). In that case, provide a short follow-up clarification question.\n"
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

    result: ClarityResult = _get_clarity_model_structured().invoke(prompt_messages)
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

