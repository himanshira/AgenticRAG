"""Pydantic models for structured outputs and flow state."""

from typing import Literal

from pydantic import BaseModel, Field

Route = Literal["pdf", "web", "direct"]


class RouteDecision(BaseModel):
    route: Route = Field(
        description=(
            "pdf = answer is in the static domain document; "
            "web = needs fresh or current information; "
            "direct = general knowledge, no retrieval needed"
        )
    )
    reason: str = Field(description="One sentence explaining the choice")


class RAGState(BaseModel):
    question: str = ""
    route: str = ""
    reason: str = ""
    answer: str = ""
