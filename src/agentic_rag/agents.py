"""Agent definitions."""

from crewai import Agent

from .config import DOMAIN_DESCRIPTION, get_llm
from .tools import pdf_tool, web_tool


def router_agent() -> Agent:
    return Agent(
        role="Query Router",
        goal="Classify each question into exactly one retrieval path: pdf, web or direct.",
        backstory=(
            f"You are a triage specialist. The static PDF covers: {DOMAIN_DESCRIPTION}.\n\n"
            "Apply these rules IN ORDER and stop at the first match:\n"
            "1. If the question is about the Transformer architecture, attention, or anything "
            "   else the PDF covers, route to 'pdf' — even if the question is basic or you "
            "   already know the answer. The PDF is the authoritative source for its topic.\n"
            "2. If the question needs current or time-sensitive information (news, prices, "
            "   recent releases, anything after 2017), route to 'web'.\n"
            "3. Only if the question is unrelated to the PDF's topic AND needs no fresh "
            "   information, route to 'direct'.\n\n"
            "Examples: 'What is a Transformer?' -> pdf. 'How does multi-head attention work?' "
            "-> pdf. 'What is the latest GPT model?' -> web. 'What is photosynthesis?' -> direct."
            ),
        llm=get_llm(),
        tools=[],  # the router only decides, it never retrieves
        allow_delegation=False,
        verbose=True,
    )


def retriever_agent() -> Agent:
    return Agent(
        role="Retriever and Answer Writer",
        goal="Retrieve evidence from the assigned source and write a grounded answer.",
        backstory=(
            "You are a meticulous researcher. You only use the tool you were told "
            "to use, cite the evidence you found, and say clearly when the source "
            "does not contain the answer instead of guessing."
        ),
        llm=get_llm(),
        tools=[pdf_tool, web_tool],
        allow_delegation=False,
        verbose=True,
    )
