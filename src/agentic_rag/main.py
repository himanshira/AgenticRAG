"""CLI entry point."""

import argparse
import json

from .flow import AgenticRAGFlow, ask


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG with CrewAI")
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--plot", action="store_true", help="Write agentic_rag_flow.html")
    args = parser.parse_args()

    if args.plot:
        AgenticRAGFlow().plot("agentic_rag_flow")
        print("Wrote agentic_rag_flow.html")

    if args.question:
        result = ask(args.question)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
