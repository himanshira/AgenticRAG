"""Task factories. Built per run because the question changes."""

from crewai import Agent, Task

from .schemas import RouteDecision

TOOL_NAMES = {"pdf": "PDF search tool", "web": "Tavily web search tool"}


def router_task(question: str, agent: Agent) -> Task:
    return Task(
        description=f"Decide the retrieval path for this question:\n\n{question}",
        expected_output="A JSON object with fields `route` and `reason`.",
        agent=agent,
        output_pydantic=RouteDecision,
    )


def retriever_task(question: str, source: str, agent: Agent) -> Task:
    tool_name = TOOL_NAMES[source]
    return Task(
        description=(
            f"Use ONLY the {tool_name} to find evidence for this question, "
            f"then write a concise, well-grounded answer.\n\nQuestion: {question}"
        ),
        expected_output=(
            "A clear answer in 1-3 paragraphs, followed by a short "
            "'Sources' section listing the passages or URLs used."
        ),
        agent=agent,
    )
