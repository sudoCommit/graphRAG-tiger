import logging
import time

import pyTigerGraph as tg

from custom.tg.load.constants import EMBED_DIM, GRAPH, VERTEX_TYPES

logger = logging.getLogger(__name__)


def run_gsql(conn: tg.TigerGraphConnection, query: str, description: str) -> object:
    logger.info(description)
    result = conn.gsql(query)
    if isinstance(result, str):
        message = result.lower()
        if (
            "failed: " in message and "failed: 0" not in message
            or "semantic check fails" in message
            or "parsing encountered" in message
            or "error" in message
            or "does not exist" in message
        ):
            raise RuntimeError(f"{description} failed: {result}")
    logger.info("%s complete", description)
    return result


def schema_names(
    conn: tg.TigerGraphConnection,
    query: str,
    pattern: str,
) -> list[str]:
    import re

    result = conn.gsql(query)
    if not isinstance(result, str):
        return []
    return re.findall(pattern, result, re.IGNORECASE)


def reset_savanna(conn: tg.TigerGraphConnection) -> None:
    logger.info("Resetting Savanna catalog")
    for graph in schema_names(conn, "SHOW GRAPH *", r"Graph\s+(\w+)\("):
        run_gsql(conn, f"DROP GRAPH {graph} CASCADE", f"Dropping graph {graph}")
    for edge in schema_names(conn, "SHOW EDGE *", r"EDGE\s+(\w+)\("):
        run_gsql(conn, f"DROP EDGE {edge}", f"Dropping edge type {edge}")
    for vertex in schema_names(conn, "SHOW VERTEX *", r"VERTEX\s+(\w+)\("):
        run_gsql(conn, f"DROP VERTEX {vertex}", f"Dropping vertex type {vertex}")
    logger.info("Savanna catalog reset complete")


def initialize(conn: tg.TigerGraphConnection) -> None:
    schema = conn.getSchema()
    vertex_types = [item.get("Name", "") for item in schema.get("VertexTypes", [])]
    logger.info("Connected to existing Savanna graph %s", GRAPH)
    logger.info("Vertex types: %s", ", ".join(name for name in vertex_types if name))


def add_vector_attributes(conn: tg.TigerGraphConnection) -> None:
    statements = "\n".join(
        f'ALTER VERTEX {vtype} ADD VECTOR ATTRIBUTE embedding(DIMENSION={EMBED_DIM}, METRIC="COSINE");'
        for vtype in sorted(VERTEX_TYPES)
    )
    job_name = f"add_hetionet_vectors_{int(time.time())}"
    job = f'''USE GRAPH {GRAPH}
CREATE SCHEMA_CHANGE JOB {job_name} FOR GRAPH {GRAPH} {{
{statements}
}}
RUN SCHEMA_CHANGE JOB {job_name} -N'''
    run_gsql(conn, job, "Adding embedding vector attributes")
    missing = [
        vtype
        for vtype in sorted(VERTEX_TYPES)
        if not conn.getVectorStatus(vtype, "embedding")
    ]
    if missing:
        raise RuntimeError(
            "Vector attribute embedding missing from vertex types: "
            + ", ".join(missing)
        )


def upload(conn: tg.TigerGraphConnection, data_dir, reset: bool = False) -> None:
    logger.info("Starting schema setup for graph %s", GRAPH)
    if conn.check_exist_graphs(GRAPH) and not reset:
        logger.info("Graph %s already exists; skipping schema creation", GRAPH)
        add_vector_attributes(conn)
        return
    if reset and conn.check_exist_graphs(GRAPH):
        run_gsql(conn, f"DROP GRAPH {GRAPH} CASCADE", "Dropping graph")

    schema_path = data_dir / "hetionet_schema.gsql"
    schema_query = schema_path.read_text(encoding="utf-8").replace(
        "HetionetGraph", GRAPH
    )
    run_gsql(conn, schema_query, "Creating local schema")
    if not conn.check_exist_graphs(GRAPH):
        raise RuntimeError(f"TigerGraph did not create graph {GRAPH}")
    add_vector_attributes(conn)
    logger.info("Schema setup complete for graph %s", GRAPH)
