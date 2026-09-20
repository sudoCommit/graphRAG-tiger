import logging

import pyTigerGraph as tg

from tg.load.constants import GRAPH, VERTEX_TYPES
from tg.load.schema import run_gsql

logger = logging.getLogger(__name__)


def vector_query_name(vertex_type: str) -> str:
    return f"search_{vertex_type.lower()}_vector"


def install_vector_search_queries(conn: tg.TigerGraphConnection) -> None:
    """Install one persistent native vector-search query per vertex type."""
    logger.info("Installing native vector-search queries")
    for vertex_type in sorted(VERTEX_TYPES):
        query_name = vector_query_name(vertex_type)
        try:
            conn.gsql(f"USE GRAPH {GRAPH}\nDROP QUERY {query_name}")
        except Exception:
            pass
        query = f'''USE GRAPH {GRAPH}
CREATE QUERY {query_name}(LIST<FLOAT> query_vec, INT k) FOR GRAPH {GRAPH} SYNTAX v3 {{
    MapAccum<VERTEX, FLOAT> @@distances;
    results = vectorSearch({{{vertex_type}.embedding}}, query_vec, k, {{ distance_map: @@distances }});
    PRINT results;
    PRINT @@distances AS distances;
}}
INSTALL QUERY {query_name}'''
        run_gsql(conn, query, f"Installing vector-search query {query_name}")
