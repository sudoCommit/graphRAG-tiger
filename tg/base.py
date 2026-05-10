import pyTigerGraph as tg
import os
from dotenv import load_dotenv

load_dotenv(override=True)

HOST = os.getenv("GRAPHRAG_HOST", "http://localhost:8000")
GRAPH = "HetionetGraph"


def _get_connection() -> tg.TigerGraphConnection:
    """Get a TigerGraph connection to the Savanna instance."""
    return tg.TigerGraphConnection(
        host=HOST,
        graphname=GRAPH,
        apiToken=os.getenv("TG_API_KEY", ""),
    )


def get_graphrag_connection() -> tg.TigerGraphConnection:
    """Get a TigerGraph connection configured for the GraphRAG service."""
    conn = _get_connection()
    conn.ai.configureGraphRAGHost(HOST)
    return conn
