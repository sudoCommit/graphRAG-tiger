import os

from dotenv import load_dotenv

load_dotenv(override=True)

# TigerGraph
TG_HOST = os.getenv(
    "TG_HOST",
    "https://tg-02fe439b-55d1-49dd-b079-f3dccc2364d4.tg-2635877100.i.tgcloud.io",
)
USERNAME = os.getenv("TG_USERNAME")
PASSWORD = os.getenv("TG_PASSWORD")
API_KEY = os.getenv("TG_API_KEY")
GRAPH = os.getenv("TG_GRAPH", "HetionetGraph")
VERTEX_TYPES = [
    "Anatomy",
    "BiologicalProcess",
    "CellularComponent",
    "Compound",
    "Disease",
    "Gene",
    "MolecularFunction",
    "Pathway",
    "PharmacologicClass",
    "SideEffect",
    "Symptom",
]

# Embedding
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 100  # OpenAI supports up to 2048 items per call

MODEL_COST_PER_TOKEN = {
    "text-embedding-ada-002": 0.000_000_1,
    "text-embedding-3-small": 0.000_000_02,
    "text-embedding-3-large": 0.000_000_13,
}
