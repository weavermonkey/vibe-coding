import json
import os
import re
import time
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import END, StateGraph


MODEL_NAME = "gemini-2.5-flash"


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


class AgentState(TypedDict, total=False):
    question: str
    search_results: List[str]
    agent_reasoning: str
    final_response: str
    iterations: int
    error: str
    needs_more_info: bool
    refined_query: str
    current_query: str
    failed_node: str
    retry_count: int


def _load_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it in your environment or .env file before running the agent."
        )
    return genai.Client(api_key=api_key)


GEMINI_CLIENT = _load_gemini_client()


def parse_llm_json(raw: str) -> Any:
    """
    Parse JSON from an LLM response safely.

    - Strips leading/trailing whitespace
    - Removes surrounding ``` / ```json fences if present
    - Uses json.loads for parsing
    - Raises on failure after logging the raw response
    """
    if raw is None:
        raise ValueError("LLM response text is None; cannot parse JSON.")

    original = raw
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json") :]
        else:
            cleaned = cleaned[len("```") :]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -len("```")]
        cleaned = cleaned.strip()

    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        print("Failed to parse LLM JSON response. Raw response:")
        print(original)
        raise ValueError(f"Failed to parse LLM JSON: {exc}") from exc


def reasoner(state: AgentState) -> AgentState:
    try:
        client = GEMINI_CLIENT
        question = state.get("current_query") or state.get("question") or ""

        if not question:
            raise ValueError("No question provided in state for reasoner node.")

        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(
            tools=[grounding_tool],
        )

        prompt = (
            "You are a research agent with access to Google Search.\n"
            "Analyze the user's question and perform any searches needed.\n"
            "Return a JSON object with the following fields:\n"
            '  - "search_summary": a concise, fact-based summary of the most\n'
            "    relevant information from your grounded search.\n"
            '  - "reasoning": a brief explanation of how you used the\n'
            "    information and whether you have enough to answer.\n"
            '  - "needs_more_info": a boolean. Use true if you believe another\n'
            "    focused grounded search would significantly improve the answer;\n"
            "    otherwise use false.\n"
            '  - "refined_query": if needs_more_info is true, provide an\n'
            "    improved, more specific search query; otherwise use an empty\n"
            "    string.\n"
            "Respond with JSON only, without any surrounding explanation."
        )

        contents = (
            f"{prompt}\n\n"
            f"User question: {state.get('question')}\n"
            f"Current search query: {question}\n"
            f"Previous search summaries (if any): {state.get('search_results')}"
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=config,
        )

        parsed = parse_llm_json(response.text)

        search_summary = parsed.get("search_summary", "")
        reasoning_text = parsed.get("reasoning", "")
        needs_more_info = bool(parsed.get("needs_more_info", False))
        refined_query = parsed.get("refined_query") or ""

        results = list(state.get("search_results") or [])
        if search_summary:
            results.append(search_summary)

        state["search_results"] = results
        state["agent_reasoning"] = reasoning_text
        state["needs_more_info"] = needs_more_info
        state["refined_query"] = refined_query
        if refined_query:
            state["current_query"] = refined_query

        state["error"] = ""
        return state
    except Exception as exc:
        state["error"] = f"reasoner node failed: {exc!r}"
        state["failed_node"] = "reasoner"
        return state


def evaluator(state: AgentState) -> AgentState:
    try:
        iterations = int(state.get("iterations") or 0) + 1
        state["iterations"] = iterations

        if state.get("needs_more_info") and iterations < 3:
            refined_query = state.get("refined_query") or state.get("question") or ""
            state["current_query"] = refined_query
        else:
            state["current_query"] = state.get("question") or ""

        state["error"] = ""
        return state
    except Exception as exc:
        state["error"] = f"evaluator node failed: {exc!r}"
        state["failed_node"] = "evaluator"
        return state


def responder(state: AgentState) -> AgentState:
    try:
        client = GEMINI_CLIENT
        question = state.get("question") or ""
        search_summaries = state.get("search_results") or []
        reasoning_text = state.get("agent_reasoning") or ""
        iterations = int(state.get("iterations") or 0)

        if not question:
            raise ValueError("No question provided in state for responder node.")

        summary_text = "\n\n".join(search_summaries)

        prompt = (
            "You are a helpful assistant.\n"
            "Use the grounded research summaries below to answer the user's\n"
            "question as accurately and clearly as possible.\n"
            "If the information seems incomplete or uncertain, be explicit\n"
            "about the uncertainty instead of guessing.\n\n"
            f"User question: {question}\n\n"
            f"Research iterations: {iterations}\n\n"
            "Grounded research summaries:\n"
            f"{summary_text}\n\n"
            "Agent reasoning so far:\n"
            f"{reasoning_text}\n\n"
            "Now provide a final, well-structured answer for the user."
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        state["final_response"] = response.text
        state["error"] = ""
        return state
    except Exception as exc:
        state["error"] = f"responder node failed: {exc!r}"
        state["failed_node"] = "responder"
        return state


def error_handler(state: AgentState) -> AgentState:
    try:
        retry_count = int(state.get("retry_count") or 0)
        last_error = state.get("error") or "Unknown error"

        if retry_count >= 2:
            question = state.get("question") or ""
            fallback = (
                "I was unable to complete your request after multiple attempts.\n\n"
                f"Question: {question}\n"
                f"Last error: {last_error}\n\n"
                "This is likely due to a temporary issue with the Gemini API or\n"
                "network connectivity. Please try again in a few minutes. If the\n"
                "problem persists, check your API key configuration and network\n"
                "access."
            )
            state["final_response"] = fallback
            return state

        delay_seconds = 2**retry_count
        print(
            f"Error handler: retry {retry_count + 1} after error: {last_error}. "
            f"Sleeping for {delay_seconds} seconds before retry."
        )
        time.sleep(delay_seconds)

        state["retry_count"] = retry_count + 1
        state["error"] = ""
        return state
    except Exception as exc:
        question = state.get("question") or ""
        fallback = (
            "An unexpected error occurred in the error handler while trying to\n"
            "recover from a previous failure.\n\n"
            f"Question: {question}\n"
            f"Original error: {state.get('error')}\n"
            f"Error handler failure: {exc!r}\n\n"
            "Please try again later or check the application logs for details."
        )
        state["final_response"] = fallback
        state["error"] = f"error_handler failed: {exc!r}"
        return state


def reasoner_router(state: AgentState) -> str:
    if state.get("error"):
        return "error_handler"
    return "evaluator"


def evaluator_router(state: AgentState) -> str:
    if state.get("error"):
        return "error_handler"

    iterations = int(state.get("iterations") or 0)
    needs_more_info = bool(state.get("needs_more_info") or False)

    if needs_more_info and iterations < 3:
        return "reasoner"

    return "responder"


def responder_router(state: AgentState) -> str:
    if state.get("error"):
        return "error_handler"
    return END


def error_router(state: AgentState) -> str:
    retry_count = int(state.get("retry_count") or 0)
    if retry_count >= 2:
        return END

    failed_node = state.get("failed_node") or ""
    if not failed_node:
        return END

    return failed_node


def build_graph() -> Any:
    builder = StateGraph(AgentState)

    builder.add_node("reasoner", reasoner)
    builder.add_node("evaluator", evaluator)
    builder.add_node("responder", responder)
    builder.add_node("error_handler", error_handler)

    builder.set_entry_point("reasoner")

    builder.add_conditional_edges("reasoner", reasoner_router)
    builder.add_conditional_edges("evaluator", evaluator_router)
    builder.add_conditional_edges("responder", responder_router)
    builder.add_conditional_edges("error_handler", error_router)

    return builder.compile()


GRAPH = build_graph()


def run_agent(question: str) -> AgentState:
    initial_state: AgentState = {
        "question": question,
        "iterations": 0,
        "search_results": [],
        "agent_reasoning": "",
        "final_response": "",
        "error": "",
        "needs_more_info": False,
        "refined_query": "",
        "current_query": "",
        "failed_node": "",
        "retry_count": 0,
    }
    final_state = GRAPH.invoke(initial_state)
    return final_state

