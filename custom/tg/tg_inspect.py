"""Inspect an existing Savanna graph without modifying its schema or data."""

import logging

from custom.tg.load.base import get_connection
from custom.tg.load.constants import GRAPH

logger = logging.getLogger(__name__)


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
    logger.info("Connected to existing Savanna graph: %s", GRAPH)
    for vertex_type, count in counts.items():
        logger.info("%s: %s", vertex_type, count)


if __name__ == "__main__":
    main()
