from __future__ import annotations

from langchain_core.embeddings import Embeddings

from tg.retrieval import SavannaRetriever


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        return [1.0, 0.0] if "alice" in text.lower() else [0.0, 1.0]


class FakeSavannaConnection:
    def getSchema(self):
        return {"VertexTypes": [{"Name": "Person"}]}

    def getVertices(self, vertex_type, limit=None):
        assert vertex_type == "Person"
        return [
            {"v_id": "1", "attributes": {"name": "Alice", "embedding": [1.0]}},
            {"v_id": "2", "attributes": {"name": "Bob"}},
        ]

    def getEdges(self, source_type, source_id, **kwargs):
        if source_id == "1":
            return [{"e_type": "KNOWS", "to_type": "Person", "to_id": "2"}]
        return []

    def __getattr__(self, name):
        if name in {"gsql", "upsertVertices", "dropGraph", "ai"}:
            raise AssertionError(f"schema-mutating API accessed: {name}")
        raise AttributeError(name)


def test_retrieval_reads_existing_schema_and_expands_edges():
    retriever = SavannaRetriever(
        connection=FakeSavannaConnection(),
        embeddings=FakeEmbeddings(),
        candidate_limit=10,
    )

    seeds = retriever.retrieve("Alice", top_k=1)
    expanded = retriever.expand(seeds.documents, num_hops=1)

    assert seeds.documents[0].metadata["vertex_id"] == "1"
    assert "embedding" not in seeds.context
    assert "Person:1 -[KNOWS]-> Person:2" in expanded.context
    assert "name: Bob" in expanded.context
