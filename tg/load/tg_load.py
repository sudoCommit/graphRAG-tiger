import argparse
import logging
from pathlib import Path

import pyTigerGraph as tg

from tg.load import data, schema
from tg.load.base import get_connection
from tg.load.constants import GRAPH
from tg.load.vector_queries import install_vector_search_queries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class _LazyConnection:
    _connection: tg.TigerGraphConnection | None = None

    def _get(self) -> tg.TigerGraphConnection:
        if self._connection is None:
            self._connection = get_connection()
        return self._connection

    def __getattr__(self, name: str) -> object:
        return getattr(self._get(), name)


conn = _LazyConnection()


def _drop_configured_graph() -> None:
    if conn.check_exist_graphs(GRAPH):
        schema.run_gsql(conn, f"DROP GRAPH {GRAPH} CASCADE", f"Dropping graph {GRAPH}")
        logger.info("Graph %s dropped", GRAPH)
    else:
        logger.info("Graph %s does not exist; skipping drop", GRAPH)


def _load_sample_data(limit: int) -> None:
    loaded_ids = data.load_vertices(conn, DATA_DIR, limit)
    data.load_edges(conn, DATA_DIR, loaded_ids, limit)
    logger.info("Data load complete for count: %s", limit)


def _prepare_catalog(args: argparse.Namespace, has_load_action: bool) -> None:
    if args.reset_savanna:
        schema.reset_savanna(conn)
    if args.drop_graph:
        _drop_configured_graph()
    if args.schema or args.reset:
        schema.upload(conn, DATA_DIR, reset=args.reset)
    elif has_load_action:
        schema.initialize(conn)


def _execute_data_action(args: argparse.Namespace, has_load_action: bool) -> None:
    if has_load_action:
        _load_sample_data(args.limit)
    if args.install_vector_queries:
        install_vector_search_queries(conn)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage and load the Hetionet TigerGraph schema."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the configured graph, recreate the schema, and load sample data when --limit is set.",
    )
    parser.add_argument(
        "--drop-graph",
        action="store_true",
        help="Drop the configured TG_GRAPH and exit unless another action is requested.",
    )
    parser.add_argument(
        "--reset-savanna",
        action="store_true",
        help="Drop every graph, edge type, and vertex type from the Savanna instance.",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Create the Hetionet schema and vector attributes if needed.",
    )
    parser.add_argument(
        "--load-data",
        action="store_true",
        help="Load sample data into the existing graph; use --limit to choose rows per type.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Load up to this many rows per vertex and edge type.",
    )
    parser.add_argument(
        "--install-vector-queries",
        action="store_true",
        help="Install the vector search queries into the graph.",
    )
    args = parser.parse_args()
    if args.load_data and args.limit is None:
        parser.error("--load-data requires --limit")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be zero or greater")
    return args


def _run_requested_actions(args: argparse.Namespace) -> None:
    has_load_action = args.load_data or args.limit is not None
    has_setup_action = args.schema or args.reset or has_load_action
    _prepare_catalog(args, has_load_action)
    _execute_data_action(args, has_load_action)
    if not (args.reset_savanna or args.drop_graph or has_setup_action or args.install_vector_queries):
        schema.initialize(conn)


def main() -> None:
    args = _parse_args()
    logger.info("TG load started for graph %s", GRAPH)
    _run_requested_actions(args)
    logger.info("TG load completed for graph %s", GRAPH)


if __name__ == "__main__":
    main()
