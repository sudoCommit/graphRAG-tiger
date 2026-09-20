"""Inspect an existing Savanna graph without modifying its schema or data."""

from __future__ import annotations

from tg.base import GRAPH, get_connection


def inspect_graph() -> dict[str, int]:
    connection = get_connection()
    schema = connection.getSchema()
    counts: dict[str, int] = {}
    for vertex in schema.get("VertexTypes", []):
        vertex_type = str(vertex.get("Name", ""))
        if vertex_type:
            counts[vertex_type] = int(connection.getVertexCount(vertex_type))
    return counts


def main() -> None:
    counts = inspect_graph()
    print(f"Connected to existing Savanna graph: {GRAPH}")
    for vertex_type, count in counts.items():
        print(f"{vertex_type}: {count}")


if __name__ == "__main__":
    main()
