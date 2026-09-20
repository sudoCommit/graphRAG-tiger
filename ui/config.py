# Configuration for the UI, including pipeline definitions, metadata, example questions, and model options.

PIPELINES = ("LLM-Only", "Basic RAG", "GraphRAG")

PIPELINE_META = {
    "LLM-Only": {"desc": "Parametric model knowledge", "color": "#8dc274", "icon": "🧠"},
    "Basic RAG": {"desc": "Native vector top-k retrieval", "color": "#5dd6e6", "icon": "🔎"},
    "GraphRAG": {"desc": "Vector seeds plus graph hops", "color": "#fda367", "icon": "🐯"},
}

EXAMPLE_QUESTIONS = [
    "What compounds treat epilepsy and what are their side effects?",
    "What anatomical structures express the gene BRCA1?",
    "What gene id is linked with protein phosphatase, Mg2+/Mn2+ dependent, 1A?",
]

MODEL_OPTIONS = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"]
DEFAULT_TOP_K = 5
DEFAULT_NUM_HOPS = 2
DEFAULT_NUM_SEEN_MIN = 1
DEFAULT_EDGE_LIMIT = 5
