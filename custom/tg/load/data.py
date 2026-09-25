import csv
import logging
import os
import tempfile
from pathlib import Path

import pyTigerGraph as tg

from custom.tg.load.constants import GRAPH, VERTEX_TYPES
from custom.tg.load.schema import run_gsql

logger = logging.getLogger(__name__)


def _limited_rows(path: Path, limit: int | None) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        next(reader, None)
        rows = (row for row in reader if len(row) >= 2)
        if limit is not None:
            rows = (row for _, row in zip(range(limit), rows))
        return [row[:2] for row in rows]


def _write_subset(header: list[str], rows: list[list[str]]) -> str:
    subset = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", newline="", delete=False
    )
    with subset:
        writer = csv.writer(subset)
        writer.writerow(header)
        writer.writerows(rows)
    return subset.name


def load_vertices(
    conn: tg.TigerGraphConnection,
    data_dir: Path,
    limit: int | None,
) -> dict[str, set[str]]:
    logger.info("Loading up to %s vertices per type", limit)
    loaded_ids: dict[str, set[str]] = {}
    for source_path in sorted((data_dir / "nodes").glob("*.csv")):
        vertex_type = source_path.stem
        if "_" in vertex_type:
            continue
        rows = _limited_rows(source_path, limit)
        loaded_ids[vertex_type] = {row[0] for row in rows}
        vertices = [
            (row[0], {"name": row[1], "text_blob": row[1]})
            for row in rows
        ]
        conn.upsertVertices(vertex_type, vertices)
        logger.info("Loaded %s %s vertices", len(vertices), vertex_type)
    return loaded_ids


def _matching_edge_rows(
    path: Path,
    loaded_ids: dict[str, set[str]],
    limit: int | None,
) -> list[list[str]]:
    from_type = next(
        (vertex_type for vertex_type in VERTEX_TYPES if path.stem.startswith(f"{vertex_type}_")),
        None,
    )
    to_type = next(
        (vertex_type for vertex_type in VERTEX_TYPES if path.stem.endswith(f"_{vertex_type}")),
        None,
    )
    if from_type is None or to_type is None:
        return []
    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        next(reader, None)
        for row in reader:
            if (
                len(row) >= 2
                and row[0] in loaded_ids.get(from_type, set())
                and row[1] in loaded_ids.get(to_type, set())
            ):
                rows.append(row[:2])
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def load_edges(
    conn: tg.TigerGraphConnection,
    data_dir: Path,
    loaded_ids: dict[str, set[str]],
    limit: int | None,
) -> None:
    logger.info("Loading up to %s edges per type", limit)
    for source_path in sorted((data_dir / "edges").glob("*.csv")):
        rows = _matching_edge_rows(source_path, loaded_ids, limit)
        edge_type = source_path.stem
        subset_path = _write_subset(["from_id", "to_id"], rows)
        job_name = f"load_{edge_type}"
        job = f'''USE GRAPH {GRAPH}
CREATE LOADING JOB {job_name} FOR GRAPH {GRAPH} {{
    DEFINE FILENAME f1;
    LOAD f1 TO EDGE {edge_type} VALUES ($0, $1) USING SEPARATOR=",", HEADER="true";
}}'''
        run_gsql(conn, job, f"Creating loading job {job_name}")
        try:
            conn.runLoadingJobWithFile(
                filePath=subset_path,
                fileTag="f1",
                jobName=job_name,
            )
            logger.info("Loaded %s %s edges", len(rows), edge_type)
        finally:
            os.unlink(subset_path)
