import logging
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_gemini_api_key
from graph.state import GraphState
from utils.prompts import SYNTHESIS_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


class SynthesisAgent:
    def __init__(self) -> None:
        self._model = None

    def _get_model(self) -> ChatGoogleGenerativeAI:
        if self._model is None:
            self._model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                api_key=get_gemini_api_key(),
                temperature=0.3,
            )
        return self._model

    def _build_messages(self, state: GraphState) -> List[BaseMessage]:
        system_message = SystemMessage(
            content=SYNTHESIS_SYSTEM_PROMPT
        )
        messages = list(state.get("messages", []))
        research_findings = state.get("research_findings") or ""
        latest_query = state.get("query") or ""

        synthesis_instruction = HumanMessage(
            content=(
                f"Latest user query: {latest_query}\n\n"
                f"Research findings to base your answer on:\n{research_findings}"
            )
        )
        return [system_message] + messages + [synthesis_instruction]

    def run(self, state: GraphState) -> GraphState:
        """
        Synthesis Agent:
        - Combines research findings with conversation history.
        - Produces a coherent, user-friendly final_response.
        """
        logger.info("Running synthesis agent.")
        prompt_messages = self._build_messages(state)
        logger.debug("Synthesis prompt messages: %s", prompt_messages)

        response = self._get_model().invoke(prompt_messages)
        if not isinstance(response, AIMessage):
            logger.error("Unexpected response type from synthesis model: %s", type(response))
            raise RuntimeError("Unexpected response type from synthesis model.")

        final_text = (response.content or "").strip()
        if not final_text:
            logger.error("Empty final response from synthesis agent.")
            raise RuntimeError("Synthesis agent produced an empty response.")

        logger.debug("Final synthesized response: %s", final_text)

        updated_state: GraphState = {
            # Append the final AI message via the messages reducer.
            "messages": [response],
            "final_response": final_text,
            "visited_nodes": ["synthesis"],
        }

        return updated_state


__all__ = ["SynthesisAgent"]
