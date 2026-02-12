import logging
from typing import Literal

from .state import GraphState


logger = logging.getLogger(__name__)


def route_after_clarity(state: GraphState) -> Literal["interrupt", "research"]:
    """
    Route after the clarity agent.

    - If clarity_status == "needs_clarification" -> "interrupt" (human-in-the-loop)
    - Otherwise -> "research"
    """
    clarity_status = state.get("clarity_status")
    logger.info("Routing after clarity with clarity_status=%s", clarity_status)
    if clarity_status == "needs_clarification":
        return "interrupt"
    return "research"


def route_after_research(state: GraphState) -> Literal["validator", "synthesis"]:
    """
    Route after the research agent.

    - If confidence_score is None or < 6 -> "validator"
    - Otherwise -> "synthesis"
    """
    confidence_score = state.get("confidence_score")
    logger.info("Routing after research with confidence_score=%s", confidence_score)
    if confidence_score is None or confidence_score < 6:
        return "validator"
    return "synthesis"


def route_after_validation(state: GraphState) -> Literal["research", "synthesis"]:
    """
    Route after the validator agent.

    - If validation_result == "insufficient" AND research_attempts < 3 -> "research"
    - Otherwise -> "synthesis"
    """
    validation_result = state.get("validation_result")
    research_attempts = state.get("research_attempts", 0)
    logger.info(
        "Routing after validation with validation_result=%s, research_attempts=%s",
        validation_result,
        research_attempts,
    )
    if validation_result == "insufficient" and research_attempts < 3:
        return "research"
    return "synthesis"


__all__ = [
    "route_after_clarity",
    "route_after_research",
    "route_after_validation",
]

