"""Retrieval against the GraphRAG server's Savanna-backed `WikipediaGraph`.

Documents are already chunked and embedded in Savanna, so every pipeline
retrieves through the server's retrieval-only `searchDocuments` endpoint and
generates the answer itself. That keeps every LLM call (and its usage/cost)
observable, which the server-side `answerQuestion` does not report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pre_defined.trace import count_tokens, truncate_tokens

METHODS = ("similarity", "hybrid", "community")
MAX_ENTITIES = 20
MAX_RELATIONSHIPS = 20
COMMUNITY_LEVEL = 2
DOCUMENT_BUDGET_SHARE = 0.85
# The infobox (competitors, venue, date, medallists) sits at the top of each page.
DOCUMENT_TOKEN_CAP = 350
MIN_DOCUMENT_TOKENS = 200
SEARCH_ATTEMPTS = 3


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str  # document id for chunks/entities, community id for community summaries
    text: str
    kind: str  # "chunk" | "entity" | "community"


@dataclass
class Retrieval:
    method: str
    query: str
    chunks: list[Chunk]
    relationships: list[str]
    latency_s: float


def _parameters(method: str, top_k: int, num_hops: int, num_seen_min: int) -> dict[str, Any]:
    if method == "similarity":
        return {"top_k": top_k, "index": "DocumentChunk", "withHyDE": False, "verbose": True}
    if method == "hybrid":
        return {
            "top_k": top_k,
            "indices": ["DocumentChunk", "Entity"],
            "num_hops": num_hops,
            "num_seen_min": num_seen_min,
            "withHyDE": False,
            "verbose": True,
        }
    if method == "community":
        return {"top_k": top_k, "community_level": COMMUNITY_LEVEL, "verbose": True}
    raise ValueError(f"unknown retrieval method: {method}")


def _texts(value: Any) -> list[str]:
    items = value if isinstance(value, list) else [value]
    return [text for item in items if (text := str(item).strip())]


def _chunk(key: str, text: str, method: str) -> Chunk:
    if method == "community":
        return Chunk(key, key, text, "community")
    # Savanna ids are lowercase; the corpus and eval gold ids are uppercase.
    source = key.split("_chunk_", 1)[0].upper()
    return Chunk(key, source, text, "entity" if text.startswith("Entity:") else "chunk")


def _parse(payload: Any, method: str) -> tuple[list[Chunk], list[str]]:
    chunks: list[Chunk] = []
    relationships: list[str] = []
    parts = [part for part in (payload if isinstance(payload, list) else [payload]) if isinstance(part, dict)]
    for part in parts:
        for key, value in (part.get("final_retrieval") or {}).items():
            chunks.extend(_chunk(str(key), text, method) for text in _texts(value))
        relationships.extend(
            f"{edge['s']} -> {edge['t']}"
            for edge in part.get("edges") or []
            if isinstance(edge, dict) and edge.get("s") and edge.get("t")
        )
    return chunks, relationships


def search(
    connection: Any, question: str, method: str, top_k: int, num_hops: int, num_seen_min: int
) -> Retrieval:
    parameters = _parameters(method, top_k, num_hops, num_seen_min)
    started = time.perf_counter()
    for attempt in range(SEARCH_ATTEMPTS):
        try:
            payload = connection.ai.searchDocuments(question, method=method, method_parameters=parameters)
            break
        except Exception:
            # Savanna intermittently aborts vector queries with a 500.
            if attempt == SEARCH_ATTEMPTS - 1:
                raise
            time.sleep(2 * (attempt + 1))
    latency = time.perf_counter() - started
    chunks, relationships = _parse(payload, method)
    return Retrieval(method, question, chunks, relationships, latency)


@dataclass
class Evidence:
    """Deduplicated evidence accumulated across one or more retrievals."""

    chunks: dict[tuple[str, str], Chunk] = field(default_factory=dict)
    relationships: dict[str, None] = field(default_factory=dict)
    shown_sources: set[str] | None = None

    def add(self, retrieval: Retrieval) -> int:
        """Merge a retrieval and return how many new document/community chunks it added."""
        new_chunks = 0
        for chunk in retrieval.chunks:
            key = (chunk.chunk_id, chunk.text)
            if key in self.chunks:
                continue
            self.chunks[key] = chunk
            new_chunks += chunk.kind != "entity"
        for relationship in retrieval.relationships:
            self.relationships.setdefault(relationship)
        return new_chunks

    def _of_kind(self, kind: str) -> list[Chunk]:
        return [chunk for chunk in self.chunks.values() if chunk.kind == kind]

    @property
    def chunk_count(self) -> int:
        return len(self._of_kind("chunk")) + len(self._of_kind("community"))

    @property
    def doc_ids(self) -> list[str]:
        """Documents the answer model actually saw once the answer context has been rendered."""
        sources = {chunk.source for chunk in self._of_kind("chunk")}
        return sorted(sources if self.shown_sources is None else sources & self.shown_sources)

    @property
    def source_ids(self) -> set[str]:
        return {chunk.source for chunk in self.chunks.values() if chunk.kind != "entity"}

    def render(self, model: str, max_tokens: int, track: bool = False) -> str:
        entities = self._of_kind("entity")[:MAX_ENTITIES]
        relationships = list(self.relationships)[:MAX_RELATIONSHIPS]
        # Reserve room for entities and relationships only when there are any.
        share = 1.0 if not (entities or relationships) else DOCUMENT_BUDGET_SHARE
        documents, per_item = self._select_documents(int(max_tokens * share), model)
        if track:
            self.shown_sources = {c.source for c in documents if c.kind == "chunk"}

        sections: list[str] = []
        for title, kind in (("Documents", "chunk"), ("Community summaries", "community")):
            items = [
                f"[{c.source}] {truncate_tokens(c.text, per_item, model)}"
                for c in documents if c.kind == kind
            ]
            if items:
                sections.append(f"{title}:\n" + "\n\n".join(items))
        if entities:
            sections.append("Entities:\n" + "\n".join(f"- [{c.source}] {c.text}" for c in entities))
        if relationships:
            sections.append("Relationships:\n" + "\n".join(f"- {line}" for line in relationships))
        return truncate_tokens("\n\n".join(sections), max_tokens, model)

    def _select_documents(self, budget: int, model: str) -> tuple[list[Chunk], int]:
        """Keep documents in retrieval order, packing full short records before truncating a long record."""
        documents = self._of_kind("chunk") + self._of_kind("community")
        selected: list[Chunk] = []
        used = 0
        per_item = DOCUMENT_TOKEN_CAP
        for document in documents:
            text = truncate_tokens(document.text, DOCUMENT_TOKEN_CAP, model)
            rendered = f"[{document.source}] {text}"
            cost = count_tokens(rendered, model) + 2
            if used + cost > budget:
                remaining = budget - used - count_tokens(f"[{document.source}] ", model)
                if remaining >= MIN_DOCUMENT_TOKENS:
                    selected.append(Chunk(document.chunk_id, document.source, truncate_tokens(document.text, remaining, model), document.kind))
                break
            selected.append(Chunk(document.chunk_id, document.source, text, document.kind))
            used += cost
        return selected, per_item
