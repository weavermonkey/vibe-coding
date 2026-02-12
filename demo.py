"""
Demo script: runs 3 scenarios against the live graph and prints results.

Requires GEMINI_API_KEY in .env. Builds the graph once and runs:
  - Scenario 1: Clear query with follow-up (Apple → competitors)
  - Scenario 2: Vague query → interrupt → resume with "I meant Tesla"
  - Scenario 3: Company switching (Microsoft → Google)
"""

from __future__ import annotations

import uuid

from langgraph.types import Command

from graph.builder import build_graph
from graph.state import GraphState, HumanMessage


# ANSI colors
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

MAX_RESPONSE_LEN = 500


def _base_state(query: str) -> GraphState:
    """Per-turn input state; reducer appends messages for the thread."""
    return {
        "messages": [HumanMessage(content=query)],
        "query": query,
    }


def _invoke(graph, state: GraphState, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(state, config=config)


def _invoke_resume(graph, clarification: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(Command(resume=clarification), config=config)


def _format_response(final_response: str | None, interrupted: bool = False) -> str:
    if interrupted:
        return "(interrupt — no final response yet)"
    if not final_response:
        return "(none)"
    text = final_response.strip()
    if len(text) > MAX_RESPONSE_LEN:
        return text[:MAX_RESPONSE_LEN] + "..."
    return text


def _print_turn(
    user_query: str,
    result: dict,
    *,
    is_resume: bool = False,
) -> None:
    label = "Resume (user clarification)" if is_resume else "User query"
    print(f"{GREEN}{BOLD}{label}:{RESET} {GREEN}{user_query}{RESET}")

    visited = result.get("visited_nodes") or []
    clarity = result.get("clarity_status")
    company = result.get("company_name")
    confidence = result.get("confidence_score")

    meta = (
        f"visited_nodes={visited}, clarity_status={clarity}, "
        f"company_name={company}, confidence_score={confidence}"
    )
    print(f"{CYAN}{meta}{RESET}")

    interrupted = "__interrupt__" in result
    response_text = _format_response(result.get("final_response"), interrupted=interrupted)
    print(response_text)
    print("-" * 60)


def _run_scenario_1(graph) -> None:
    """Clear query with follow-up: Apple then competitors."""
    thread_id = f"demo-scenario-1-{uuid.uuid4()}"
    print(f"\n{BOLD}{YELLOW}=== Scenario 1: Clear query with follow-up ==={RESET}\n")

    turn1_query = "Tell me about Apple"
    state1 = _base_state(turn1_query)
    result1 = _invoke(graph, state1, thread_id)
    _print_turn(turn1_query, result1)

    turn2_query = "What about their competitors?"
    state2 = _base_state(turn2_query)
    result2 = _invoke(graph, state2, thread_id)
    _print_turn(turn2_query, result2)


def _run_scenario_2(graph) -> None:
    """Vague query → interrupt → resume with 'I meant Tesla'."""
    thread_id = f"demo-scenario-2-{uuid.uuid4()}"
    print(f"\n{BOLD}{YELLOW}=== Scenario 2: Vague query (interrupt) → resume ==={RESET}\n")

    turn1_query = "Tell me about the company"
    state1 = _base_state(turn1_query)
    result1 = _invoke(graph, state1, thread_id)
    _print_turn(turn1_query, result1)

    if "__interrupt__" not in result1:
        print(f"{YELLOW}Warning: expected __interrupt__ on vague query.{RESET}\n")

    clarification = "I meant Tesla"
    result2 = _invoke_resume(graph, clarification, thread_id)
    _print_turn(clarification, result2, is_resume=True)


def _run_scenario_3(graph) -> None:
    """Company switching: Microsoft then Google."""
    thread_id = f"demo-scenario-3-{uuid.uuid4()}"
    print(f"\n{BOLD}{YELLOW}=== Scenario 3: Company switching ==={RESET}\n")

    turn1_query = "Tell me about Microsoft"
    state1 = _base_state(turn1_query)
    result1 = _invoke(graph, state1, thread_id)
    _print_turn(turn1_query, result1)

    turn2_query = "Now tell me about Google"
    state2 = _base_state(turn2_query)
    result2 = _invoke(graph, state2, thread_id)
    _print_turn(turn2_query, result2)


def main() -> None:
    print(f"{BOLD}Building graph (once)...{RESET}")
    graph = build_graph()
    print(f"{BOLD}Graph built.{RESET}")

    scenarios = [
        ("Scenario 1 (clear + follow-up)", _run_scenario_1),
        ("Scenario 2 (vague → interrupt → resume)", _run_scenario_2),
        ("Scenario 3 (company switching)", _run_scenario_3),
    ]

    for name, run_fn in scenarios:
        try:
            run_fn(graph)
        except Exception as e:
            print(f"\n{YELLOW}Scenario failed: {name}{RESET}")
            print(f"Exception: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
