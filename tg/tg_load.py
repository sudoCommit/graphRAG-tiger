import argparse
import csv
import os
import re
import sys
import tempfile
import time
import logging
from pathlib import Path

import pyTigerGraph as tg
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
	from tg.base import get_connection
except ModuleNotFoundError:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
	from tg.base import get_connection

load_dotenv(override=True)

HOST = os.getenv(
	"TG_HOST",
	"https://tg-02fe439b-55d1-49dd-b079-f3dccc2364d4.tg-2635877100.i.tgcloud.io",
)
GRAPH = os.getenv("TG_GRAPH", "HetionetGraph")
USERNAME = os.getenv("TG_USERNAME")
PASSWORD = os.getenv("TG_PASSWORD")
API_KEY = os.getenv("TG_API_KEY")
GRAPHRAG_HOST = os.getenv("GRAPHRAG_HOST", "http://localhost:8000")
DATA_DIR = Path(__file__).resolve().parent / "data"
DOCUMENT_PATH = DATA_DIR / "hetionet-docs.jsonl"
VERTEX_TYPES = {
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
}


class _LazyConnection:
	_connection: tg.TigerGraphConnection | None = None

	def _get(self) -> tg.TigerGraphConnection:
		if self._connection is None:
			self._connection = get_connection()
		return self._connection

	def __getattr__(self, name: str) -> object:
		return getattr(self._get(), name)


conn = _LazyConnection()


def _run_gsql(query: str, description: str) -> object:
	logger.info(description)
	result = conn.gsql(query)
	if isinstance(result, str):
		message = result.lower()
		failed_summary = re.search(r"failed:\s*[1-9]\d*", message)
		if (
			failed_summary
			or "semantic check fails" in message
			or "parsing encountered" in message
			or "error" in message
			or "does not exist" in message
		):
			raise RuntimeError(f"{description} failed: {result}")
	logger.info("%s complete", description)
	return result


def _schema_names(query: str, pattern: str) -> list[str]:
	result = conn.gsql(query)
	if not isinstance(result, str):
		return []
	return re.findall(pattern, result, re.IGNORECASE)


def reset_savanna() -> None:
	"""Drop every graph and global schema object from the Savanna instance."""
	logger.info("Resetting Savanna catalog")
	graphs = _schema_names("SHOW GRAPH *", r"Graph\s+(\w+)\(")
	if graphs:
		logger.info(f"Dropping graphs: {', '.join(graphs)}")
	for graph in graphs:
		_run_gsql(f"DROP GRAPH {graph} CASCADE", f"Dropping graph {graph}")

	edges = _schema_names("SHOW EDGE *", r"EDGE\s+(\w+)\(")
	if edges:
		logger.info(f"Dropping edge types: {', '.join(edges)}")
	for edge in edges:
		_run_gsql(f"DROP EDGE {edge}", f"Dropping edge type {edge}")

	vertices = _schema_names("SHOW VERTEX *", r"VERTEX\s+(\w+)\(")
	if vertices:
		logger.info(f"Dropping vertex types: {', '.join(vertices)}")
	for vertex in vertices:
		_run_gsql(f"DROP VERTEX {vertex}", f"Dropping vertex type {vertex}")
	logger.info("Savanna catalog reset complete")


def create_graph() -> None:
	logger.info(f"Creating graph {GRAPH}...")
	_run_gsql(f"CREATE GRAPH {GRAPH}()", "Creating graph")
	if not conn.check_exist_graphs(GRAPH):
		raise RuntimeError(f"TigerGraph did not create graph {GRAPH}")
	logger.info(f"Graph {GRAPH} created and ready")


def initialize_graphrag() -> None:
	"""Validate access to the existing graph without changing its schema."""
	schema = conn.getSchema()
	vertex_types = [item.get("Name", "") for item in schema.get("VertexTypes", [])]
	logger.info(f"Connected to existing Savanna graph {GRAPH}")
	logger.info(f"Vertex types: {', '.join(name for name in vertex_types if name)}")


# def ingest_documents() -> None:
# 	if not DOCUMENT_PATH.exists():
# 		raise FileNotFoundError(f"Document source not found: {DOCUMENT_PATH}")
# 	logger.info(f"Creating document ingest for {DOCUMENT_PATH}...")
# 	load_job_info = conn.ai.createDocumentIngest(
# 		data_source="local",
# 		data_source_config={"data_path": str(DOCUMENT_PATH)},
# 		file_format="json",
# 	)
# 	logger.info("Running document ingest...")
# 	result = conn.ai.runDocumentIngest(
# 		load_job_info["load_job_id"],
# 		load_job_info["data_source_id"],
# 		load_job_info["data_path"],
# 		data_source="local",
# 	)
# 	logger.info("Document ingest complete")
# 	logger.info(result)


# def force_consistency_update() -> None:
# 	logger.info("Forcing GraphRAG consistency update...")
# 	result = conn.ai.forceConsistencyUpdate("graphrag")
# 	logger.info("GraphRAG consistency update requested")
# 	logger.info(result)


def setup_graphrag(reset: bool = False) -> None:
	if reset:
		raise ValueError("Schema reset is disabled; manage the graph schema in Savanna")
	initialize_graphrag()


def add_vector_attributes() -> None:
	logger.info("Adding embedding vector attributes to %s vertex types", len(VERTEX_TYPES))
	statements = "\n".join(
		f'ALTER VERTEX {vtype} ADD VECTOR ATTRIBUTE embedding(DIMENSION=1536, METRIC="COSINE");'
		for vtype in sorted(VERTEX_TYPES)
	)
	job_name = f"add_hetionet_vectors_{int(time.time())}"
	logger.info("Running vector schema job %s", job_name)
	job = f'''USE GRAPH {GRAPH}
CREATE SCHEMA_CHANGE JOB {job_name} FOR GRAPH {GRAPH} {{
{statements}
}}
RUN SCHEMA_CHANGE JOB {job_name} -N'''
	result = conn.gsql(job)
	try:
		_verify_vector_attributes()
	except RuntimeError:
		if _gsql_failed(result):
			raise RuntimeError(f"Adding vector attributes failed: {result}") from None
		raise
	logger.info("Embedding vector attributes verified")


def _gsql_failed(result: object) -> bool:
	return isinstance(result, str) and any(
		marker in result.lower()
		for marker in ("encountered", "semantic check fails", "failed", "error")
	)


def _verify_vector_attributes() -> None:
	logger.info("Verifying embedding vector attributes")
	missing = []
	for vtype in sorted(VERTEX_TYPES):
		if not conn.getVectorStatus(vtype, "embedding"):
			missing.append(vtype)
	if missing:
		raise RuntimeError(f"Vector attribute embedding missing from vertex types: {', '.join(sorted(missing))}")


def vector_query_name(vertex_type: str) -> str:
	return f"search_{vertex_type.lower()}_vector"


def install_vector_search_queries() -> None:
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
		_run_gsql(query, f"Installing vector-search query {query_name}")


def schema_upload(reset: bool = False) -> None:
	logger.info("Starting schema setup for graph %s", GRAPH)
	if conn.check_exist_graphs(GRAPH) and not reset:
		logger.info(f"Graph {GRAPH} already exists; skipping schema creation")
		add_vector_attributes()
		install_vector_search_queries()
		return
	if reset:
		if conn.check_exist_graphs(GRAPH):
			logger.info("Dropping graph...")
			_run_gsql(f"DROP GRAPH {GRAPH} CASCADE", "Dropping graph")
		else:
			logger.info("Graph does not exist; skipping drop")

	schema_path = DATA_DIR / "hetionet_schema.gsql"
	schema_query = schema_path.read_text(encoding="utf-8").replace("HetionetGraph", GRAPH)
	logger.info(f"Creating local schema in graph {GRAPH}...")
	_run_gsql(schema_query, "Creating local schema")
	if not conn.check_exist_graphs(GRAPH):
		raise RuntimeError(f"TigerGraph did not create graph {GRAPH}")
	logger.info(f"Graph {GRAPH} created and ready")
	add_vector_attributes()
	install_vector_search_queries()
	logger.info("Schema setup complete for graph %s", GRAPH)


def _limited_rows(path: Path, limit: int) -> list[list[str]]:
	with path.open(newline="", encoding="utf-8") as source:
		reader = csv.reader(source)
		next(reader, None)
		return [row[:2] for _, row in zip(range(limit), reader) if len(row) >= 2]


def _write_subset(header: list[str], rows: list[list[str]]) -> str:
	subset = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", newline="", delete=False)
	with subset:
		writer = csv.writer(subset)
		writer.writerow(header)
		writer.writerows(rows)
	return subset.name


def create_and_run_vertex_jobs(limit: int) -> dict[str, set[str]]:
	"""Load at most limit rows per vertex type and return loaded IDs."""
	logger.info("Loading up to %s vertices per type", limit)
	loaded_ids: dict[str, set[str]] = {}
	for source_path in sorted((DATA_DIR / "nodes").glob("*.csv")):
		vtype = source_path.stem
		if "_" in vtype:
			continue

		rows = _limited_rows(source_path, limit)
		loaded_ids[vtype] = {row[0] for row in rows}
		vertices = [
			(
				row[0],
				{"name": row[1], "text_blob": row[1]},
			)
			for row in rows
		]
		conn.upsertVertices(vtype, vertices)
		logger.info("Loaded %s %s vertices", len(vertices), vtype)
	return loaded_ids


def _matching_edge_rows(path: Path, loaded_ids: dict[str, set[str]], limit: int) -> list[list[str]]:
	from_type = next((vtype for vtype in VERTEX_TYPES if path.stem.startswith(f"{vtype}_")), None)
	to_type = next((vtype for vtype in VERTEX_TYPES if path.stem.endswith(f"_{vtype}")), None)
	if from_type is None or to_type is None:
		return []
	rows: list[list[str]] = []
	with path.open(newline="", encoding="utf-8") as source:
		reader = csv.reader(source)
		next(reader, None)
		for row in reader:
			if (
				len(row) >= 2
				and row[0] in loaded_ids.get(from_type, set())
				and row[1] in loaded_ids.get(to_type, set())
			):
				rows.append(row[:2])
				if len(rows) >= limit:
					break
	return rows


def create_and_run_edge_jobs(loaded_ids: dict[str, set[str]], limit: int) -> None:
	"""Load only edges whose endpoints were included in the vertex sample."""
	logger.info("Loading up to %s edges per type", limit)
	for source_path in sorted((DATA_DIR / "edges").glob("*.csv")):
		rows = _matching_edge_rows(source_path, loaded_ids, limit)
		etype = source_path.stem
		subset_path = _write_subset(["from_id", "to_id"], rows)
		job_name = f"load_{etype}"
		job = f'''USE GRAPH {GRAPH}
CREATE LOADING JOB {job_name} FOR GRAPH {GRAPH} {{
	DEFINE FILENAME f1;
	LOAD f1 TO EDGE {etype} VALUES ($0, $1) USING SEPARATOR=",", HEADER="true";
}}'''
		_run_gsql(job, f"Creating loading job {job_name}")
		try:
			conn.runLoadingJobWithFile(filePath=subset_path, fileTag="f1", jobName=job_name)
			logger.info("Loaded %s %s edges", len(rows), etype)
		finally:
			os.unlink(subset_path)


def _has_schema_action(args: argparse.Namespace) -> bool:
	return args.schema or args.reset


def _has_load_action(args: argparse.Namespace) -> bool:
	return args.load_data or args.limit is not None


def _drop_configured_graph() -> None:
	if conn.check_exist_graphs(GRAPH):
		_run_gsql(f"DROP GRAPH {GRAPH} CASCADE", f"Dropping graph {GRAPH}")
		logger.info(f"Graph {GRAPH} dropped")
		return
	logger.info(f"Graph {GRAPH} does not exist; skipping drop")


def _load_sample_data(limit: int) -> None:
	logger.info("Starting sample data load with limit %s", limit)
	install_vector_search_queries()
	loaded_ids = create_and_run_vertex_jobs(limit)
	create_and_run_edge_jobs(loaded_ids, limit)
	logger.info(f"Data load complete for count: {limit}")
	

def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Manage and load the Hetionet TigerGraph schema.")
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
	has_schema_action = _has_schema_action(args)
	has_load_action = _has_load_action(args)

	if args.reset_savanna:
		reset_savanna()
		if not (has_schema_action or has_load_action):
			return

	if args.drop_graph:
		_drop_configured_graph()
		if not (has_schema_action or has_load_action):
			return

	if has_schema_action:
		schema_upload(reset=args.reset)
		if has_load_action:
			_load_sample_data(args.limit)
		return

	if has_load_action:
		_load_sample_data(args.limit)
		return

	if args.install_vector_queries:
		install_vector_search_queries()

	setup_graphrag()


def main() -> None:
	args = _parse_args()
	logger.info("tg_load started for graph %s", GRAPH)
	_run_requested_actions(args)


if __name__ == "__main__":
	main()
