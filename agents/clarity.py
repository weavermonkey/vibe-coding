import logging
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from config import get_gemini_api_key
from graph.state import GraphState
from utils.prompts import (
    CLARITY_LAST_DISCUSSED_COMPANY_CONTEXT,
    CLARITY_SYSTEM_PROMPT,
)


logger = logging.getLogger(__name__)


class ClarityResult(BaseModel):
    clarity_status: str
    company_name: Optional[str] = None
    clarification_question: Optional[str] = None


class ClarityAgent:
    def __init__(self) -> None:
        self._model = None
        self._model_structured = None

    def _get_model(self) -> ChatGoogleGenerativeAI:
        if self._model is None:
            self._model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                api_key=get_gemini_api_key(),
                temperature=0.0,
            )
        return self._model

    def _get_model_structured(self):
        if self._model_structured is None:
            self._model_structured = self._get_model().with_structured_output(
                ClarityResult
            )
        return self._model_structured

    def _build_messages(
        self,
        messages: list[BaseMessage],
        query: str,
        last_discussed_company: Optional[str] = None,
    ) -> list[BaseMessage]:
        system_content = CLARITY_SYSTEM_PROMPT
        if last_discussed_company:
            system_content += "\n\n" + CLARITY_LAST_DISCUSSED_COMPANY_CONTEXT.format(
                last_discussed_company=last_discussed_company
            )
        system_message = SystemMessage(content=system_content)
        latest_query_message = HumanMessage(
            content=f"Latest user query: {query}",
        )
        return [system_message] + messages + [latest_query_message]

    def run(self, state: GraphState) -> GraphState:
        """
        Clarity Agent:
        - Determines whether the query is clear or needs clarification.
        - Attempts to extract a company name when present.
        - Optionally sets a clarification question for the human-in-the-loop step.
        """
        logger.info("Running clarity agent.")
        messages = list(state.get("messages", []))
        query = state.get("query") or ""

        prompt_messages = self._build_messages(
            messages, query, state.get("last_discussed_company")
        )
        logger.debug("Clarity agent prompt messages: %s", prompt_messages)

        result: ClarityResult = self._get_model_structured().invoke(prompt_messages)
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


__all__ = ["ClarityAgent", "ClarityResult"]
