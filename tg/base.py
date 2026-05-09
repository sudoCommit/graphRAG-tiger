import pyTigerGraph as tg
import os
from dotenv import load_dotenv

load_dotenv(override=True)

HOST = "https://tg-b07aa060-e3df-4207-94fc-971616eb7348.tg-2635877100.i.tgcloud.io"
GRAPH = "HetionetGraph"


def get_connection() -> tg.TigerGraphConnection:
    """Get a TigerGraph connection to the Savanna instance."""
    return tg.TigerGraphConnection(
        host=HOST,
        graphname=GRAPH,
        apiToken=os.getenv("TG_API_KEY"),
    )


def get_graphrag_connection() -> tg.TigerGraphConnection:
    """Get a TigerGraph connection configured for the GraphRAG service."""
    conn = get_connection()
    graphrag_host = os.getenv("GRAPHRAG_HOST", "http://localhost:8000")
    conn.ai.configureGraphRAGHost(graphrag_host)
    return conn
