import logging
from typing import Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import get_gemini_api_key
from graph.state import GraphState
from utils.prompts import VALIDATOR_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


class ValidationAssessment(BaseModel):
    validation_result: Literal["sufficient", "insufficient"]
    critique: str = Field(description="Brief critique of the research quality")
    suggestions: str = Field(
        description="Suggestions for improvement if validation is insufficient"
    )


class ValidatorAgent:
    def __init__(self) -> None:
        self._model = None
        self._model_structured = None

    def _get_model(self) -> ChatGoogleGenerativeAI:
        if self._model is None:
            self._model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                api_key=get_gemini_api_key(),
                temperature=0.1,
            )
        return self._model

    def _get_model_structured(self):
        if self._model_structured is None:
            self._model_structured = self._get_model().with_structured_output(
                ValidationAssessment
            )
        return self._model_structured

    def _build_messages(
        self,
        messages: list[BaseMessage],
        query: str,
        research_findings: Optional[str],
    ) -> list[BaseMessage]:
        system_message = SystemMessage(
            content=VALIDATOR_SYSTEM_PROMPT
        )
        context = f"User query: {query}\n\nResearch findings:\n{research_findings or '<<missing>>'}"
        analysis_request = HumanMessage(
            content=context,
        )
        return [system_message] + messages + [analysis_request]

    def run(self, state: GraphState) -> GraphState:
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

        prompt_messages = self._build_messages(messages, query, research_findings)
        logger.debug("Validator prompt messages: %s", prompt_messages)

        assessment = self._get_model_structured().invoke(prompt_messages)
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


__all__ = ["ValidatorAgent", "ValidationAssessment"]
