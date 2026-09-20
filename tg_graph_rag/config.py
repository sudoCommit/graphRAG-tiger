import os
from dotenv import load_dotenv

load_dotenv(override=True)

GRAPHRAG_HOST = os.getenv("GRAPHRAG_HOST", "http://localhost:8000")

# Default retrieval parameters for hybrid search
DEFAULT_TOP_K = 5
DEFAULT_NUM_HOPS = 2
DEFAULT_NUM_SEEN_MIN = 1
DEFAULT_EDGE_LIMIT = 25
