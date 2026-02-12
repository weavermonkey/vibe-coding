import logging
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import get_gemini_api_key
from graph.state import GraphState
from utils.prompts import RESEARCH_CONFIDENCE_SYSTEM_PROMPT, RESEARCH_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


class ConfidenceAssessment(BaseModel):
    confidence_score: float = Field(
        description=(
            "Confidence score from 0-10 on how well the research answers the query"
        )
    )
    reasoning: str = Field(
        description="Brief explanation of the confidence assessment"
    )


class ResearchAgent:
    def __init__(self) -> None:
        self._research_model = None
        self._confidence_model = None
        self._confidence_assessor = None

    def _get_research_model(self) -> ChatGoogleGenerativeAI:
        if self._research_model is None:
            self._research_model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=get_gemini_api_key(),
                temperature=0.0,
            )
        return self._research_model

    def _get_confidence_model(self) -> ChatGoogleGenerativeAI:
        if self._confidence_model is None:
            self._confidence_model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                api_key=get_gemini_api_key(),
                temperature=0.0,
            )
        return self._confidence_model

    def _get_confidence_assessor(self):
        if self._confidence_assessor is None:
            self._confidence_assessor = self._get_confidence_model().with_structured_output(
                ConfidenceAssessment
            )
        return self._confidence_assessor

    def _build_research_prompt(
        self, query: str, company_name: Optional[str]
    ) -> str:
        parts: list[str] = []
        if company_name:
            parts.append(f"Company of interest: {company_name}.")
        if query:
            parts.append(f"User query: {query}")
        instruction = RESEARCH_SYSTEM_PROMPT
        parts.append(instruction)
        return "\n\n".join(parts)

    def run(self, state: GraphState) -> GraphState:
        """
        Research Agent:
        - Uses Gemini with Search Grounding (Google Search tool) to gather company info.
        - Populates `research_findings`.
        - Assigns a confidence_score in the 0–10 range using a secondary LLM call.
        """
        logger.info("Running research agent.")
        query = state.get("query") or ""
        company_name = state.get("company_name")

        prompt = self._build_research_prompt(query, company_name)
        logger.debug("Research prompt: %s", prompt)

        # Use Gemini via LangChain with built-in Google Search grounding.
        response = self._get_research_model().invoke(
            prompt,
            tools=[{"google_search": {}}],
        )

        if isinstance(response, AIMessage):
            content = response.content
            if isinstance(content, str):
                text = content.strip()
            else:
                text_blocks = []
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            value = block.get("text")
                            if isinstance(value, str):
                                text_blocks.append(value)
                text = "\n".join(text_blocks).strip() if text_blocks else ""
                if not text:
                    text_attr = getattr(response, "text", "")
                    if isinstance(text_attr, str):
                        text = text_attr.strip()
        else:
            logger.error("Unexpected response type from research model: %s", type(response))
            raise RuntimeError("Unexpected response type from research model.")

        if not text:
            logger.error("Empty research findings returned from Gemini.")
            raise RuntimeError("Gemini returned empty research findings.")

        logger.debug("Raw research findings text: %s", text)

        # Assess confidence using a structured-output Gemini model.
        assessment_prompt = [
            SystemMessage(
                content=RESEARCH_CONFIDENCE_SYSTEM_PROMPT
            ),
            HumanMessage(content=f"User query: {query}\n\nResearch findings:\n{text}"),
        ]
        assessment = self._get_confidence_assessor().invoke(assessment_prompt)
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
            "last_discussed_company": company_name,
            "visited_nodes": ["research"],
        }

        return updated_state


__all__ = ["ResearchAgent", "ConfidenceAssessment"]
