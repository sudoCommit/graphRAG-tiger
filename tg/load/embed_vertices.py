"""
Embedding pipeline for Hetionet TigerGraph vertices.

Steps executed by main:
    1. embed_all_vertex_types() - reads every nodes/VType.csv, builds a text
                                blob from meaningful columns, calls the OpenAI
                                embeddings API in batches, then upserts
                                text_blob + embedding to TigerGraph.
    2. wait_for_vector_indexes() - polls /vector/status until all indexes are
                                ready (or a timeout elapses).
"""

import csv
import os
import time
import argparse
import logging
from pathlib import Path
from typing import Iterator

import pyTigerGraph as tg
from dotenv import load_dotenv
from openai import OpenAI

from tg.load.base import get_connection
from tg.load.constants import (
    BATCH_SIZE,
    EMBED_DIM,
    EMBED_MODEL,
    GRAPH,
    MODEL_COST_PER_TOKEN,
    VERTEX_TYPES,
)


load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "nodes"
logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for batched document embeddings."""

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_HOST_URL") or None,
            timeout=60,
        )
        logger.info("Embedding client initialized (model=%s, dim=%s)", EMBED_MODEL, EMBED_DIM)

    def embed_batch(
        self,
        texts: list[str],
        model: str,
    ) -> tuple[list[list[float]], int]:
        try:
            response = self.client.embeddings.create(
                model=model,
                input=texts,
                dimensions=EMBED_DIM,
                encoding_format="float",
            )
        except Exception as exc:
            logger.exception(
                "Embedding request failed (%s: %s)",
                type(exc).__name__,
                str(exc),
            )
            raise
        tokens = int(getattr(response.usage, "total_tokens", 0) or 0)
        return [item.embedding for item in response.data], tokens


def _iter_batches(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def embed_vertex_type(
    embedding_client: EmbeddingClient,
    conn: tg.TigerGraphConnection,
    vtype: str,
    limit: int | None = 10,
) -> tuple[int, float]:
    """Generate embeddings for up to ``limit`` vertices of *vtype*."""
    total_tokens_used = 0
    total_cost = 0.0
    model_cost = MODEL_COST_PER_TOKEN.get(EMBED_MODEL)

    if model_cost is None:
        logger.warning("Embedding model not found: %s", EMBED_MODEL)
        model_cost = 0.0

    csv_path = DATA_DIR / f"{vtype}.csv"
    if not csv_path.exists():
        logger.warning("%s not found; skipping %s", csv_path, vtype)
        return 0, 0.0

    logger.info("Reading %s", csv_path)
    rows = _read_rows(csv_path)

    if limit is not None:
        rows = rows[:limit]

    if not rows:
        logger.warning("%s is empty; skipping %s", csv_path, vtype)
        return 0, 0.0

    batches = list(_iter_batches(rows, BATCH_SIZE))
    logger.info(
        "Processing %s %s vertices in %s batch(es) of up to %s",
        len(rows), vtype, len(batches), BATCH_SIZE,
    )
    total_accepted = 0

    for batch_num, batch in enumerate(batches, start=1):
        # Use the pre-built 'text_blob' column from json2csv.py as the text
        # to embed. It synthesizes name, type, and all meaningful fields.
        texts = [row["text_blob"] for row in batch]

        logger.info(
            "Embedding %s batch %s/%s (%s vertices)",
            vtype, batch_num, len(batches), len(batch),
        )
        embeddings, tokens_used = embedding_client.embed_batch(
            texts,
            model=EMBED_MODEL,
        )
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Embedding count mismatch for {vtype}: "
                f"expected {len(batch)}, got {len(embeddings)}"
            )
        cost_incurred = tokens_used * model_cost

        vertices = [
            (
                str(row["id"]),
                {"text_blob": text, "embedding": vec},
            )
            for row, text, vec in zip(batch, texts, embeddings)
        ]

        try:
            accepted = conn.upsertVertices(vtype, vertices)
        except Exception as exc:
            logger.warning("Could not upsert vertices for %s: %s", vtype, exc)
            accepted = 0

        logger.info(
            "Upserted %s/%s vertices for %s batch %s/%s (%s tokens, $%.6f)",
            accepted, len(batch), vtype, batch_num, len(batches), tokens_used, cost_incurred,
        )

        total_accepted += accepted
        total_tokens_used += tokens_used
        total_cost += cost_incurred

    logger.info("Upserted %s %s embeddings", total_accepted, vtype)
    logger.info("Tokens used: %s, cost: $%.6f", total_tokens_used, total_cost)
    return total_tokens_used, total_cost


def embed_all_vertex_types(
    embedding_client: EmbeddingClient,
    conn: tg.TigerGraphConnection,
    limit: int | None = 10,
) -> None:
    logger.info("Generating and upserting embeddings")
    tokens = 0
    cost = 0.0
    for vtype in VERTEX_TYPES:
        logger.info("Processing vertex type %s", vtype)
        v_tokens, v_cost = embed_vertex_type(
            embedding_client,
            conn,
            vtype,
            limit=limit,
        )
        tokens += v_tokens
        cost += v_cost

    logger.info("Total tokens used: %s, total cost: $%.6f", tokens, cost)


def wait_for_vector_indexes(
    conn: tg.TigerGraphConnection,
    timeout_s: int = 600,
) -> None:
    """Poll /vector/status until all embedding indexes are ready.

    On TigerGraph versions that do not support vector indexes the status check
    will raise an exception, which is caught and skipped.
    """
    logger.info("Waiting for vector indexes to be ready")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        all_ready = True
        for vtype in VERTEX_TYPES:
            try:
                status = conn.getVectorIndexStatus(
                    graphName=GRAPH, vertexType=vtype, vectorName="embedding"
                )
                if status.get("NeedRebuildServers", []):
                    all_ready = False
                    break
            except Exception:
                # Older TG or index not created – nothing to wait for
                logger.warning("Vector index status is unavailable; skipping wait")
                return
        if all_ready:
            logger.info("All vector indexes are ready")
            return
        remaining = int(deadline - time.time())
        logger.info("Vector indexes still building (%ss remaining)", remaining)
        time.sleep(15)
    logger.warning("Timed out waiting for vector indexes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate embeddings for loaded TigerGraph vertices.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum vertices to embed per type; use 0 for all vertices.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or greater")
    limit = args.limit or None

    logger.info(
        "Starting embedding pipeline for graph %s (limit=%s)",
        GRAPH, "all" if limit is None else limit,
    )

    conn = get_connection()
    logger.info("Connected to TigerGraph graph %s", GRAPH)
    embedding_client = EmbeddingClient()

    embed_all_vertex_types(embedding_client, conn, limit=limit)

    wait_for_vector_indexes(conn)

    logger.info("Embedding pipeline complete")


if __name__ == "__main__":
    main()
