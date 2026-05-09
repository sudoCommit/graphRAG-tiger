from pathlib import Path

from llm.prompty_loader import load_system_prompt, render_prompt


PROMPTY_PATH = Path(__file__).with_name("graph_rag.prompty")
GRAPH_RAG_SYSTEM_PROMPT = load_system_prompt(str(PROMPTY_PATH))


def render_graph_rag_prompt(
    question: str,
    retrieval_method: str,
    retrieved_context: str,
) -> str:
    return render_prompt(
        GRAPH_RAG_SYSTEM_PROMPT,
        question=question,
        retrieval_method=retrieval_method,
        retrieved_context=retrieved_context,
    )