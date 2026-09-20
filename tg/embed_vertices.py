"""
Embedding pipeline for Hetionet TigerGraph vertices.

Steps executed by main():
  1. add_embedding_schema()   – schema change job: adds text_blob STRING and
                                embedding LIST<DOUBLE> to every vertex type
                                (safe to re-run; skips if already present).
    2. TigerGraph manages the native vector indexes for those attributes.
  3. embed_all_vertex_types() – reads every nodes/VType.csv, builds a text
                                blob from meaningful columns, calls the OpenAI
                                embeddings API in batches, then upserts
                                text_blob + embedding to TigerGraph.
  4. wait_for_vector_indexes()– polls /vector/status until all indexes are
                                ready (or a timeout elapses).

Environment variables required:
  TG_API_KEY      – TigerGraph API token (same as used by tiger_graph.py)
  OPENAI_API_KEY  – OpenAI secret key for text-embedding-3-small

Usage:
  python -m tg.embed_vertices
"""

import csv
import os
import time
import argparse
from pathlib import Path
from typing import Iterator

import pyTigerGraph as tg
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


TG_HOST = os.getenv(
    "TG_HOST",
    "https://tg-02fe439b-55d1-49dd-b079-f3dccc2364d4.tg-2635877100.i.tgcloud.io",
)
GRAPH = "HetionetGraph"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 100  # OpenAI supports up to 2048 items per call; 100 is safe
DATA_DIR = Path(__file__).resolve().parent / "data" / "nodes"
# All vertex types in HetionetGraph
VERTEX_TYPES = [
    "Anatomy",
    "BiologicalProcess",
    "CellularComponent",
    "Compound",
    "Disease",
    "Gene",
    "MolecularFunction",
    "Pathway",
    "PharmacologicClass",
    "SideEffect",
    "Symptom",
]
_MODEL_COST_PER_TOKEN = {
    "text-embedding-ada-002": 0.000_000_1,
    "text-embedding-3-small": 0.000_000_02,
    "text-embedding-3-large": 0.000_000_13,
}


def _get_connection() -> tg.TigerGraphConnection:
    """Direct connection to TigerGraph (bypasses GraphRAG proxy).
    Required for gsql(), upsertVertices(), and getVectorStatus().
    """
    return tg.TigerGraphConnection(
        host=TG_HOST,
        graphname=GRAPH,
        apiToken=os.getenv("TG_API_KEY"),
    )

def _iter_batches(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def add_embedding_schema(conn: tg.TigerGraphConnection) -> None:
    """Add the native TigerGraph vector attribute to every vertex type.

    Runs as a single schema change job so all alterations are atomic.
    Safe to re-run: if the attributes already exist the exception is caught.
    """
    print("Step 1 – adding embedding schema attributes...")
    alter_stmts = "\n    ".join(
        f'ALTER VERTEX {vt} ADD VECTOR ATTRIBUTE embedding(DIMENSION={EMBED_DIM}, METRIC="COSINE");'
        for vt in VERTEX_TYPES
    )
    gsql = f"""
CREATE GLOBAL SCHEMA_CHANGE JOB add_hetionet_embeddings {{
    {alter_stmts}
}}
RUN GLOBAL SCHEMA_CHANGE JOB add_hetionet_embeddings
"""
    try:
        conn.gsql(gsql)
        print("  ✅ Schema attributes added")
    except Exception as exc:
        msg = str(exc).lower()
        if "already exists" in msg or "duplicate" in msg:
            print("  ⚠️  Attributes already present – skipping schema change")
        else:
            raise


def create_vector_indexes() -> None:
    """Native vector attributes create and maintain their own indexes."""
    print("Step 2 – native vector indexes are managed by TigerGraph")


def embed_vertex_type(
    conn: tg.TigerGraphConnection,
    openai_client: OpenAI,
    vtype: str,
    limit: int | None = 10,
) -> tuple[int, float]:
    """Generate embeddings for up to ``limit`` vertices of *vtype*."""
    total_tokens_used = 0
    total_cost = 0.0
    model_cost = _MODEL_COST_PER_TOKEN.get(EMBED_MODEL)

    if model_cost is None:
        print(f"Embedding model not found: {EMBED_MODEL}")
        model_cost = 0.0

    csv_path = DATA_DIR / f"{vtype}.csv"
    if not csv_path.exists():
        print(f"  ⚠️  {csv_path} not found – skipping {vtype}")
        return 0, 0.0

    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if limit is not None:
        rows = rows[:limit]

    if not rows:
        print(f"  ⚠️  {csv_path} is empty – skipping {vtype}")
        return 0, 0.0

    print(f"  Processing {len(rows):,} {vtype} vertices …")
    total_accepted = 0

    for batch in _iter_batches(rows, BATCH_SIZE):
        # Use the pre-built 'text_blob' column from json2csv.py as the text
        # to embed. It synthesizes name, type, and all meaningful fields.
        texts = [row["text_blob"] for row in batch]

        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts, dimensions=EMBED_DIM, encoding_format="float")
        embeddings = [item.embedding for item in resp.data]
        tokens_used = resp.usage.total_tokens
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
            print(f"  ⚠️  Could not upsert vertices for {vtype}: {exc}")
            accepted = 0

        total_accepted += accepted
        total_tokens_used += tokens_used
        total_cost += cost_incurred

    print(f"  ✅ Upserted {total_accepted:,} {vtype} embeddings")
    print(f"  💰 Tokens used: {total_tokens_used}, Cost: ${total_cost}")
    return total_tokens_used, total_cost


def embed_all_vertex_types(
    conn: tg.TigerGraphConnection,
    openai_client: OpenAI,
    limit: int | None = 10,
) -> None:
    print("Step 3 – generating and upserting embeddings …")
    tokens = 0
    cost = 0.0
    for vtype in VERTEX_TYPES:
        print(f"\n[{vtype}]")
        v_tokens, v_cost = embed_vertex_type(conn, openai_client, vtype, limit=limit)
        tokens += v_tokens
        cost += v_cost

    print(f"\n✅ --> Total tokens used: {tokens}, Total cost: ${cost}")



def wait_for_vector_indexes(
    conn: tg.TigerGraphConnection,
    timeout_s: int = 600,
) -> None:
    """Poll /vector/status until all embedding indexes are ready.

    On TigerGraph versions that do not support vector indexes the status check
    will raise an exception, which is caught and skipped.
    """
    print("\nStep 4 – waiting for vector indexes to be ready …")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        all_ready = True
        for vtype in VERTEX_TYPES:
            try:
                ready = conn.getVectorIndexStatus(
                    graphName=GRAPH, vertexType=vtype, vectorName="embedding"
                )
                if not ready:
                    all_ready = False
                    break
            except Exception:
                # Older TG or index not created – nothing to wait for
                print("  ℹ️  getVectorStatus not supported – skipping wait")
                return
        if all_ready:
            print("  ✅ All vector indexes are ready")
            return
        remaining = int(deadline - time.time())
        print(f"  ⏳ Still building … ({remaining}s remaining)")
        time.sleep(15)
    print("  ⚠️  Timed out waiting for vector indexes")


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

    conn = _get_connection()
    openai_client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_HOST_URL"),
        timeout=60,
    )

    # add_embedding_schema(conn)
    # time.sleep(3)  # let schema propagate

    create_vector_indexes()

    embed_all_vertex_types(conn, openai_client, limit=limit)

    wait_for_vector_indexes(conn)

    print("\n✅ Embedding pipeline complete")


if __name__ == "__main__":
    main()
