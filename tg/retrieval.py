from __future__ import annotations

import asyncio
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pyTigerGraph import TigerGraphConnection

from tg.base import get_connection


@dataclass(frozen=True)
class RetrievalResult:
    documents: list[Document]
    context: str


class SavannaRetriever:
    """Read and rank vertices from an existing Savanna schema."""

    def __init__(
        self,
        connection: TigerGraphConnection | None = None,
        embeddings: Embeddings | None = None,
        candidate_limit: int | None = None,
        edge_limit: int | None = None,
    ) -> None:
        self.connection = connection or get_connection()
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_HOST_URL") or None,
        )
        self.edge_limit = edge_limit or int(
            os.getenv("TG_EDGES_PER_VERTEX", "25")
        )
        self._vertex_types_cache: list[str] | None = None
        self._documents_by_vertex: dict[tuple[str, str], Document] = {}
        self._retrieval_cache: dict[tuple[str, int], RetrievalResult] = {}
        self._expansion_cache: dict[
            tuple[tuple[tuple[str, str], ...], int, int], RetrievalResult
        ] = {}

    def retrieve(self, question: str, top_k: int) -> RetrievalResult:
        query_vector = self.embeddings.embed_query(question)
        return self._retrieve_vector_results(question, top_k, query_vector)

    async def aretrieve(self, question: str, top_k: int) -> RetrievalResult:
        query_vector = await self.embeddings.aembed_query(question)
        return await asyncio.to_thread(
            self._retrieve_vector_results,
            question,
            top_k,
            query_vector,
        )

    def _retrieve_vector_results(
        self,
        question: str,
        top_k: int,
        query_vector: list[float],
    ) -> RetrievalResult:
        cache_key = (" ".join(question.lower().split()), top_k)
        cached = self._retrieval_cache.get(cache_key)
        if cached is not None:
            return RetrievalResult(list(cached.documents), cached.context)

        candidates: list[tuple[float, str, str]] = []
        for vertex_type in self._vertex_types():
            query_name = f"search_{vertex_type.lower()}_vector"
            result = self.connection.runInstalledQuery(
                query_name,
                params={"query_vec": query_vector, "k": top_k},
            )
            candidates.extend(
                (score, vertex_type, vertex_id)
                for vertex_id, score in self._search_hits(result)
            )

        # vectorSearch returns distances; for cosine distance, lower is closer.
        candidates.sort(key=lambda item: item[0])
        documents = [
            document
            for _, vertex_type, vertex_id in candidates[:top_k]
            if (document := self._get_document(vertex_type, vertex_id)) is not None
        ]
        result = RetrievalResult(documents, self._format_documents(documents))
        self._cache_put(self._retrieval_cache, cache_key, result)
        return RetrievalResult(list(result.documents), result.context)

    async def aexpand(
        self,
        seeds: list[Document],
        num_hops: int,
        num_seen_min: int = 1,
    ) -> RetrievalResult:
        return await asyncio.to_thread(
            self.expand,
            seeds,
            num_hops,
            num_seen_min,
        )

    def expand(
        self,
        seeds: list[Document],
        num_hops: int,
        num_seen_min: int = 1,
    ) -> RetrievalResult:
        seed_key = tuple(sorted(
            (str(doc.metadata["vertex_type"]), str(doc.metadata["vertex_id"]))
            for doc in seeds
        ))
        cache_key = (seed_key, num_hops, num_seen_min)
        cached = self._expansion_cache.get(cache_key)
        if cached is not None:
            return RetrievalResult(list(cached.documents), cached.context)

        documents = list(seeds)
        seen_documents = {
            (str(doc.metadata["vertex_type"]), str(doc.metadata["vertex_id"]))
            for doc in seeds
        }
        frontier = set(seen_documents)
        relationships: list[str] = []

        for _ in range(max(0, num_hops)):
            next_frontier, hop_relationships = self._walk_one_hop(frontier)

            accepted = {
                key
                for key, count in next_frontier.items()
                if count >= max(1, num_seen_min) and key not in seen_documents
            }
            relationships.extend(
                line for key, line in hop_relationships if key in accepted
            )
            for key in accepted:
                document = self._get_document(*key)
                if document is not None:
                    documents.append(document)
            seen_documents.update(accepted)
            frontier = accepted
            if not frontier:
                break

        context = self._format_documents(documents)
        if relationships:
            context += "\n\nRelationships:\n" + "\n".join(relationships)
        result = RetrievalResult(documents, context)
        self._cache_put(self._expansion_cache, cache_key, result)
        return RetrievalResult(list(result.documents), result.context)

    @staticmethod
    def _cache_put(cache: dict, key: Any, value: RetrievalResult) -> None:
        if len(cache) >= 256:
            cache.pop(next(iter(cache)))
        cache[key] = value

    def _vertex_types(self) -> list[str]:
        if self._vertex_types_cache is not None:
            return self._vertex_types_cache
        configured = {
            item.strip()
            for item in os.getenv("TG_VERTEX_TYPES", "").split(",")
            if item.strip()
        }
        schema = self.connection.getSchema()
        self._vertex_types_cache = [
            str(vertex["Name"])
            for vertex in schema.get("VertexTypes", [])
            if vertex.get("Name") and (
                not configured or vertex["Name"] in configured
            )
        ]
        return self._vertex_types_cache

    def _get_document(self, vertex_type: str, vertex_id: str) -> Document | None:
        key = (str(vertex_type), str(vertex_id))
        cached = self._documents_by_vertex.get(key)
        if cached is not None:
            return cached
        rows = self.connection.getVerticesById(
            key[0], key[1], select="name,text_blob"
        )
        if not isinstance(rows, list) or not rows:
            return None
        document = self._vertex_document(key[0], rows[0])
        if document is not None:
            self._documents_by_vertex[key] = document
        return document

    @staticmethod
    def _search_hits(result: Any) -> list[tuple[str, float]]:
        rows: list[dict[str, Any]] = []
        distances: dict[str, float] = {}
        for payload in result if isinstance(result, list) else [result]:
            if not isinstance(payload, dict):
                continue
            for key in ("v", "results"):
                values = payload.get(key)
                if isinstance(values, list):
                    rows.extend(row for row in values if isinstance(row, dict))
            distance_data = payload.get("distances")
            if isinstance(distance_data, dict):
                distances.update({str(k): float(v) for k, v in distance_data.items()})
        hits = []
        for row in rows:
            vertex_id = row.get("v_id", row.get("id"))
            if vertex_id is None:
                continue
            score = distances.get(str(vertex_id), 0.0)
            hits.append((str(vertex_id), score))
        return hits

    def _walk_one_hop(
        self, frontier: set[tuple[str, str]]
    ) -> tuple[Counter[tuple[str, str]], list[tuple[tuple[str, str], str]]]:
        targets: Counter[tuple[str, str]] = Counter()
        relationships: list[tuple[tuple[str, str], str]] = []
        for source_type, source_id in frontier:
            edges = self.connection.getEdges(
                source_type,
                source_id,
                limit=self.edge_limit,
                withType=True,
            )
            for edge in edges if isinstance(edges, list) else []:
                target = self._edge_target(edge)
                if target is None:
                    continue
                target_type, target_id = target
                targets[target] += 1
                edge_type = str(
                    edge.get("e_type") or edge.get("type") or "related_to"
                )
                relationships.append(
                    (
                        target,
                        f"{source_type}:{source_id} -[{edge_type}]-> "
                        f"{target_type}:{target_id}",
                    )
                )
        return targets, relationships

    @staticmethod
    def _vertex_document(
        vertex_type: str, row: dict[str, Any]
    ) -> Document | None:
        vertex_id = row.get("v_id", row.get("id"))
        attributes = row.get("attributes", {})
        if vertex_id is None or not isinstance(attributes, dict):
            return None
        fields = [f"type: {vertex_type}", f"id: {vertex_id}"]
        for name, value in attributes.items():
            if name.lower() in {"embedding", "vector"} or value in (
                None,
                "",
                [],
                {},
            ):
                continue
            rendered = str(value)
            if len(rendered) <= 1000:
                fields.append(f"{name}: {rendered}")
        return Document(
            page_content="; ".join(fields),
            metadata={"vertex_type": vertex_type, "vertex_id": str(vertex_id)},
        )

    @staticmethod
    def _edge_target(edge: dict[str, Any]) -> tuple[str, str] | None:
        target_type = edge.get("to_type") or edge.get("target_type")
        target_id = edge.get("to_id") or edge.get("target_id")
        if target_type is None or target_id is None:
            return None
        return str(target_type), str(target_id)

    @staticmethod
    def _format_documents(documents: list[Document]) -> str:
        return "\n".join(
            f"- {document.page_content}" for document in documents
        )
