import logging
from typing import Optional

from google import genai
from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import get_gemini_api_key
from graph.state import GraphState


logger = logging.getLogger(__name__)


_genai_client = genai.Client(api_key=get_gemini_api_key())
_google_search_tool = types.Tool(google_search=types.GoogleSearch())


class ConfidenceAssessment(BaseModel):
    confidence_score: float = Field(
        description=(
            "Confidence score from 0-10 on how well the research answers the query"
        )
    )
    reasoning: str = Field(
        description="Brief explanation of the confidence assessment"
    )


_confidence_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=get_gemini_api_key(),
    temperature=0.0,
)
_confidence_assessor = _confidence_model.with_structured_output(ConfidenceAssessment)


def _build_research_prompt(query: str, company_name: Optional[str]) -> str:
    parts: list[str] = []
    if company_name:
        parts.append(f"Company of interest: {company_name}.")
    if query:
        parts.append(f"User query: {query}")
    instruction = (
        "You are a research analyst using Gemini with Google Search Grounding.\n"
        "Use live Google Search to gather up-to-date information about the company, "
        "including:\n"
        "- Recent news and developments\n"
        "- Financial performance and key metrics (if available)\n"
        "- Products, services, and strategic initiatives\n"
        "- Any notable risks, controversies, or competitive pressures\n\n"
        "Synthesize this into a concise but detailed research brief suitable for an "
        "expert user. Include dates and specific figures when they are available.\n"
        "The response will be passed on to downstream agents, so avoid addressing "
        "the user directly.\n"
    )
    parts.append(instruction)
    return "\n\n".join(parts)


def run_research_agent(state: GraphState) -> GraphState:
    """
    Research Agent:
    - Uses Gemini with Search Grounding (Google Search tool) to gather company info.
    - Populates `research_findings`.
    - Assigns a confidence_score in the 0–10 range using a secondary LLM call.
    """
    logger.info("Running research agent.")
    query = state.get("query") or ""
    company_name = state.get("company_name")

    prompt = _build_research_prompt(query, company_name)
    logger.debug("Research prompt: %s", prompt)

    config = types.GenerateContentConfig(
        tools=[_google_search_tool],
    )
    response = _genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config,
    )

    text = (response.text or "").strip()
    if not text:
        logger.error("Empty research findings returned from Gemini.")
        raise RuntimeError("Gemini returned empty research findings.")

    logger.debug("Raw research findings text: %s", text)

    # Assess confidence using a structured-output Gemini model.
    assessment_prompt = [
        SystemMessage(
            content=(
                "You are a research quality assessor. Given a user query and "
                "research findings, rate your confidence from 0-10 that the "
                "research adequately answers the query. Consider completeness, "
                "relevance, specificity, and whether concrete data points are present."
            )
        ),
        HumanMessage(content=f"User query: {query}\n\nResearch findings:\n{text}"),
    ]
    assessment = _confidence_assessor.invoke(assessment_prompt)
    logger.info(
        "Research confidence assessment: score=%s, reasoning=%s",
        assessment.confidence_score,
        assessment.reasoning,
    )

    updated_state: GraphState = {
        # messages use an operator.add reducer, so we only return the delta
        "messages": [AIMessage(content=text)],
        "research_findings": text,
        "confidence_score": assessment.confidence_score,
        "visited_nodes": ["research"],
    }

    return updated_state


__all__ = ["run_research_agent"]

