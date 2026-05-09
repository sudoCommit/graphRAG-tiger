from pathlib import Path

from llm.prompty_loader import load_system_prompt, render_prompt


PROMPTY_PATH = Path(__file__).with_name("basic_rag.prompty")
RAG_SYSTEM_PROMPT = load_system_prompt(str(PROMPTY_PATH))


def build_rag_prompt(question: str, context: str) -> list[dict]:
    """Build a RAG prompt with retrieved context injected."""
    return [
        {
            "role": "system",
            "content": render_prompt(
                RAG_SYSTEM_PROMPT,
                context=context,
                question=question,
            ),
        },
    ]
