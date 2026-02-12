from __future__ import annotations

import operator
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class GraphState(TypedDict, total=False):
    """
    State carried through the LangGraph workflow.

    Notes:
    - `messages` holds the full conversation history across turns and uses an
      operator.add reducer so that messages from multiple invocations are
      appended instead of overwritten.
    - `research_attempts` should start at 0 for each turn and never exceed 3
      within a single turn.
    - Additional fields (like `visited_nodes` or `clarification_question`) are
      allowed beyond the minimum required by the user specification.
    """

    # Conversation and query
    messages: Annotated[List[BaseMessage], operator.add]
    query: str

    # Clarity and company information
    company_name: Optional[str]
    last_discussed_company: Optional[str]
    clarity_status: Optional[Literal["clear", "needs_clarification"]]

    # Research results
    research_findings: Optional[str]
    confidence_score: Optional[float]  # 0–10

    # Validation and attempts
    validation_result: Optional[Literal["sufficient", "insufficient"]]
    research_attempts: int

    # Final response for the user
    final_response: Optional[str]

    # Optional extras to support UX and testing
    clarification_question: Optional[str]
    visited_nodes: Annotated[List[str], operator.add]


__all__ = [
    "GraphState",
    "AIMessage",
    "HumanMessage",
    "BaseMessage",
]


