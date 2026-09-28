import os
from dotenv import load_dotenv

# 1. Load .env BEFORE any CrewAI imports
load_dotenv()

# 2. Fix variable name and force telemetry flags
os.environ["CREWAI_TRACING_ENABLED"] = "true"
os.environ["CREWAI_TRACES_ENABLED"] = "true"
os.environ["PYTHONUTF8"] = "1"

# Automatically map CREW_API_KEY if that's what you named it in your .env
if os.getenv("CREW_API_KEY") and not os.getenv("CREWAI_API_KEY"):
    os.environ["CREWAI_API_KEY"] = os.getenv("CREW_API_KEY")

# 3. Now import CrewAI
from crewai import Crew, Process
from crewai.flow.flow import Flow, listen, or_, router, start

from .agents import retriever_agent, router_agent
from .config import get_llm
from .schemas import RAGState, RouteDecision
from .tasks import retriever_task, router_task


class AgenticRAGFlow(Flow[RAGState]):
    def __init__(self):
        super().__init__(tracing=True)  # Enable tracing for the flow

    @start()
    def receive_question(self):
        return self.state.question

    @router(receive_question)
    def decide_route(self):
        agent = router_agent()
        crew = Crew(
            agents=[agent],
            tasks=[router_task(self.state.question, agent)],
            process=Process.sequential,
            tracing=True
        )
        decision: RouteDecision = crew.kickoff().pydantic
        self.state.route = decision.route
        self.state.reason = decision.reason
        return decision.route

    def _retrieve(self, source: str) -> str:
        agent = retriever_agent()
        crew = Crew(
            agents=[agent],
            tasks=[retriever_task(self.state.question, source, agent)],
            process=Process.sequential,
            tracing=True
        )
        self.state.answer = crew.kickoff().raw
        return self.state.answer

    @listen("pdf")
    def retrieve_from_pdf(self):
        return self._retrieve("pdf")

    @listen("web")
    def retrieve_from_web(self):
        return self._retrieve("web")

    @listen("direct")
    def answer_directly(self):
        self.state.answer = get_llm().call(
            f"Answer this question clearly and concisely:\n\n{self.state.question}"
        )
        return self.state.answer

    @listen(or_(retrieve_from_pdf, retrieve_from_web, answer_directly))
    def finalize(self, answer: str) -> dict:
        return {
            "question": self.state.question,
            "route": self.state.route,
            "reason": self.state.reason,
            "answer": answer,
        }


def ask(question: str) -> dict:
    return AgenticRAGFlow().kickoff(inputs={"question": question})