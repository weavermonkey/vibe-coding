import logging
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_gemini_api_key
from graph.state import GraphState


logger = logging.getLogger(__name__)


_synthesis_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=get_gemini_api_key(),
    temperature=0.3,
)


def _build_synthesis_messages(state: GraphState) -> List[BaseMessage]:
    system_message = SystemMessage(
        content=(
            "You are a senior research analyst.\n"
            "Given the full conversation history and the latest research findings, "
            "produce a clear, user-friendly answer to the user's latest query.\n"
            "Requirements:\n"
            "- Maintain continuity with previous turns in the conversation.\n"
            "- Summarize key points, adding structure (sections, bullets) when helpful.\n"
            "- If the user asked for follow-ups (e.g., competitors or CEO), focus "
            "on that while still grounding in the earlier context.\n"
            "- Do not mention internal agent roles, routing, or system details."
        )
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


def run_synthesis_agent(state: GraphState) -> GraphState:
    """
    Synthesis Agent:
    - Combines research findings with conversation history.
    - Produces a coherent, user-friendly final_response.
    """
    logger.info("Running synthesis agent.")
    prompt_messages = _build_synthesis_messages(state)
    logger.debug("Synthesis prompt messages: %s", prompt_messages)

    response = _synthesis_model.invoke(prompt_messages)
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


__all__ = ["run_synthesis_agent"]

