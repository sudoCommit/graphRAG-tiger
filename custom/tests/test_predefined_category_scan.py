import pytest

from pre_defined import category_scan
from pre_defined.pipelines import _event_scan_retrieval, run_structured_graph


class FakeGraph:
    def __init__(self, documents: list[tuple[str, str, str]], missing_content: set[str] | None = None) -> None:
        self.documents = documents
        self.missing_content = missing_content or set()

    def _get_vertices(self, vertex_type: str, **kwargs: object) -> list[dict]:
        if vertex_type == "Document":
            return [{"v_id": doc_id.lower(), "attributes": {"id": doc_id.lower()}} for doc_id, _, _ in self.documents]
        assert vertex_type == "Content"
        return [
            {"v_id": doc_id.lower(), "attributes": {"id": doc_id.lower(), "text": text, "ctype": "single"}}
            for doc_id, _, text in self.documents
            if doc_id.upper() not in self.missing_content
        ]

    def _get_vertices_by_id(self, vertex_type: str, vertex_ids: list[str]) -> list[dict]:
        assert vertex_type == "Content"
        by_id = {doc_id.lower(): text for doc_id, _, text in self.documents}
        return [
            {"v_id": doc_id, "attributes": {"text": by_id[doc_id]}}
            for doc_id in vertex_ids
            if doc_id in by_id and doc_id.upper() not in self.missing_content
        ]

    def __getattr__(self, name: str):
        if name == "getVertices":
            return self._get_vertices
        if name == "getVertexCount":
            return lambda vertex_type: len(self.documents) if vertex_type == "Document" else 0
        raise AttributeError(name)


def _event(doc_id: str, title: str, competitors: int | None) -> tuple[str, str, str]:
    infobox = f"  event: {title.rsplit(' – ', 1)[-1]}\n  games: 2000 Summer\n"
    if competitors is not None:
        infobox += f"  competitors: {competitors}\n"
    return doc_id, title, f"Title: {title}\n\n[Infobox Olympic event]\n{infobox}\nDescription."


@pytest.fixture
def category_corpus() -> FakeGraph:
    docs = [
        _event("Q1", "Sailing at the 2000 Summer Olympics – Soling", 48),
        _event("Q2", "Sailing at the 2000 Summer Olympics – Star", 32),
        _event("Q3", "Sailing at the 2000 Summer Olympics – Women's Mistral", 29),
    ]
    category_scan._GRAPH_EVENT_CACHE.clear()
    return FakeGraph(docs)


def test_count_category_uses_complete_graph_records(category_corpus: FakeGraph) -> None:
    result = category_scan.scan_category(
        category_corpus,
        "According to the corpus, how many sailing events at the 2000 Summer Olympics had more than 30 competitors?",
    )

    assert result is not None and result.complete
    assert result.answer.startswith("Final answer: 2 ")
    assert len(result.doc_ids) == 3


def test_superlative_returns_title_with_max_competitors(category_corpus: FakeGraph) -> None:
    result = category_scan.scan_category(
        category_corpus,
        "Which sailing event at the 2000 Summer Olympics had the highest number of competitors?",
    )

    assert result is not None and result.complete
    assert result.answer == "Final answer: Sailing at the 2000 Summer Olympics – Soling [Q1]"


def test_incomplete_graph_category_is_not_counted(category_corpus: FakeGraph) -> None:
    category_corpus.missing_content.add("Q3")
    category_scan._GRAPH_EVENT_CACHE.clear()
    result = category_scan.scan_category(
        category_corpus,
        "How many sailing events at the 2000 Summer Olympics had more than 30 competitors?",
    )

    assert result is not None and not result.complete
    assert not result.answer
    assert "contents" in result.stop_reason


def test_lookup_question_does_not_take_category_route(category_corpus: FakeGraph) -> None:
    result = category_scan.scan_category(
        category_corpus,
        "Who won the gold medal in Sailing at the 2000 Summer Olympics – Soling?",
    )

    assert result is None


def test_event_scan_policy_detects_count_and_venue_date_questions() -> None:
    assert category_scan.requires_event_scan(
        "How many sailing events at the 2000 Summer Olympics had more than 30 competitors?"
    )
    assert category_scan.requires_event_scan(
        "Who won the event held at Olympic Weightlifting Gymnasium on 20 September 1988?"
    )
    assert not category_scan.requires_event_scan("Who won the gold medal in the 2012 men's pole vault?")


def test_structured_baseline_is_scoped_and_llm_free(category_corpus: FakeGraph) -> None:
    import asyncio

    category_scan.clear_graph_event_cache()
    result = asyncio.run(run_structured_graph(
        category_corpus,
        "How many sailing events at the 2000 Summer Olympics had more than 30 competitors?",
    ))
    not_applicable = asyncio.run(run_structured_graph(category_corpus, "Who won the 2012 men's pole vault?"))

    assert result.applicable
    assert result.answer.startswith("Final answer: 2 ")
    assert result.total_tokens == 0
    assert not not_applicable.applicable


def test_agentic_event_scan_returns_graph_evidence(category_corpus: FakeGraph) -> None:
    import asyncio

    category_scan.clear_graph_event_cache()
    retrieval = asyncio.run(_event_scan_retrieval(
        category_corpus,
        "How many sailing events at the 2000 Summer Olympics had more than 30 competitors?",
    ))

    assert retrieval is not None
    assert retrieval.method == "event_category_scan"
    assert len([chunk for chunk in retrieval.chunks if chunk.kind == "chunk"]) == 3
    assert any("Graph coverage: complete" in chunk.text for chunk in retrieval.chunks)


def test_unique_venue_date_lookup_uses_graph_gold() -> None:
    doc_id = "Q25239316"
    title = "Weightlifting at the 1988 Summer Olympics – Men's 60 kg"
    text = (
        f"Title: {title}\n\n[Infobox Olympic event]\n"
        "  games: 1988 Summer\n  venue: Olympic Weightlifting Gymnasium\n"
        "  date: 20 September 1988\n  gold: Naim Süleymanoğlu\n"
    )
    category_scan._GRAPH_EVENT_CACHE.clear()
    graph = FakeGraph([(doc_id, title, text)])

    result = category_scan.scan_venue_date(
        graph,
        "Who won the gold medal in the event held at Olympic Weightlifting Gymnasium on 20 September 1988?",
    )

    assert result is not None and result.complete
    assert result.answer == "Final answer: Naim Süleymanoğlu [Q25239316]"
