import pyTigerGraph as tg
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("TG_HOST", "http://localhost:8000")
GRAPH = os.getenv("TG_GRAPH", "HetionetGraph")


def get_connection() -> tg.TigerGraphConnection:
    """Connect to an existing Savanna graph without changing its schema."""
    if not GRAPH:
        raise RuntimeError("TG_GRAPH must name an existing graph in Savanna")
    return tg.TigerGraphConnection(
        host=HOST,
        graphname=GRAPH,
        apiToken=os.getenv("TG_API_KEY", ""),
    )


def get_graphrag_connection() -> tg.TigerGraphConnection:
    """Backward-compatible alias for the direct, non-mutating connection."""
    return get_connection()
