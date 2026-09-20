from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from llm.rag_pipeline import run_rag
from tg.retrieval import RetrievalResult


class FakeRetriever:
    def __init__(self):
        self.expansions = 0

    def retrieve(self, question: str, top_k: int) -> RetrievalResult:
        document = Document(
            page_content="type: Person; id: 1; name: Alice",
            metadata={"vertex_type": "Person", "vertex_id": "1"},
        )
        return RetrievalResult([document], "seed context")

    def expand(
        self, documents, num_hops: int, num_seen_min: int
    ) -> RetrievalResult:
        self.expansions += 1
        return RetrievalResult(documents, "expanded graph context")


class FakeChatModel:
    def invoke(self, messages, config=None):
        return AIMessage(
            content="grounded answer",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
            },
            response_metadata={
                "model_name": "gpt-4.1-mini",
                "finish_reason": "stop",
            },
        )


def test_basic_rag_skips_graph_expansion():
    retriever = FakeRetriever()
    result = run_rag(
        "Who is Alice?",
        mode="basic",
        top_k=1,
        model="gpt-4.1-mini",
        retriever=retriever,
        chat_model=FakeChatModel(),
    )

    assert retriever.expansions == 0
    assert result.retrieved_context == "seed context"
    assert result.total_tokens == 14


def test_graph_rag_expands_neighbors():
    retriever = FakeRetriever()
    result = run_rag(
        "Who does Alice know?",
        mode="graph",
        top_k=1,
        model="gpt-4.1-mini",
        num_hops=2,
        retriever=retriever,
        chat_model=FakeChatModel(),
    )

    assert retriever.expansions == 1
    assert result.retrieved_context == "expanded graph context"
