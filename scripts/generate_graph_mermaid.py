"""Script to generate Mermaid diagram from pydantic-graph state machine."""

import logging

from app.graph.graph import chat_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    mermaid_code = chat_graph.render(title="UniSage RAG Graph")
    print("```mermaid")
    print(mermaid_code)
    print("```")


if __name__ == "__main__":
    main()
