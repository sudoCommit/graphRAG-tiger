import pyTigerGraph as tg

from custom.tg.load.constants import TG_HOST, GRAPH, API_KEY


def get_connection() -> tg.TigerGraphConnection:
    """Connect to an existing Savanna graph without changing its schema."""
    if not GRAPH:
        raise RuntimeError("TG_GRAPH must name an existing graph in Savanna")
    return tg.TigerGraphConnection(
        host=TG_HOST,
        graphname=GRAPH,
        apiToken=API_KEY,
    )
