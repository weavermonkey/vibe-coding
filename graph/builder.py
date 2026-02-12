import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from agents.clarity import run_clarity_agent
from agents.research import run_research_agent
from agents.synthesis import run_synthesis_agent
from agents.validator import run_validator_agent
from graph.routing import (
    route_after_clarity,
    route_after_research,
    route_after_validation,
)
from graph.state import GraphState, HumanMessage


logger = logging.getLogger(__name__)


def clarity_interrupt_node(state: GraphState) -> Command:
    """
    Human-in-the-loop interrupt node for clarification.

    Behavior:
    - On first execution, calls `interrupt()` with a clarification question, which
      pauses the graph and surfaces the question to the caller under `__interrupt__`.
    - When resumed with `Command(resume=...)`, the resume value is treated as the
      user's clarification. The node updates `query` and `messages`, marks the
      query as clear, and returns control to the clarity node for re-evaluation.
    """
    question = state.get("clarification_question") or (
        "Your question about the company is ambiguous. "
        "Please clarify which company or topic you are interested in."
    )
    logger.info("Pausing for human clarification: %s", question)

    clarification = interrupt(question)

    logger.info("Received clarification from human: %s", clarification)

    updated_state: GraphState = {
        # Append the clarified user message using the reducer.
        "messages": [HumanMessage(content=str(clarification))],
        "query": str(clarification),
        "visited_nodes": ["clarity_interrupt"],
    }

    # After receiving clarification, route back to the clarity node so it can
    # re-run with the updated query and full message history.
    return Command(update=updated_state, goto="clarity")


def build_graph() -> CompiledStateGraph:
    """
    Construct and compile the LangGraph StateGraph for the multi-agent system.

    Nodes:
    - clarity
    - clarity_interrupt (human-in-the-loop)
    - research
    - validator
    - synthesis

    The graph uses a MemorySaver checkpointer so that:
    - Multi-turn conversations are preserved via `thread_id`.
    - Interrupt/resume flows work correctly.
    """
    logger.info("Building LangGraph StateGraph for multi-agent research system.")

    builder: StateGraph[GraphState] = StateGraph(GraphState)

    builder.add_node("clarity", run_clarity_agent)
    builder.add_node("clarity_interrupt", clarity_interrupt_node)
    builder.add_node("research", run_research_agent)
    builder.add_node("validator", run_validator_agent)
    builder.add_node("synthesis", run_synthesis_agent)

    builder.add_edge(START, "clarity")

    builder.add_conditional_edges(
        "clarity",
        route_after_clarity,
        {
            "interrupt": "clarity_interrupt",
            "research": "research",
        },
    )

    builder.add_conditional_edges(
        "research",
        route_after_research,
        {
            "validator": "validator",
            "synthesis": "synthesis",
        },
    )

    builder.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "research": "research",
            "synthesis": "synthesis",
        },
    )

    builder.add_edge("synthesis", END)

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("Graph compiled successfully with MemorySaver checkpointer.")
    return graph


__all__ = ["build_graph", "Command"]

