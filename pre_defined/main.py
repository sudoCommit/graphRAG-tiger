import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Iterator

import pyTigerGraph as tg
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "data" / "corpus" / "corpus.jsonl"
FORBIDDEN_GRAPH = "HetionetGraph"

load_dotenv(ROOT.parent / ".env", override=True)


def _graph_name() -> str:
    graph = os.getenv("PREDEFINED_GRAPH", "WikipediaGraph")
    if graph == FORBIDDEN_GRAPH:
        raise RuntimeError(
            f"Refusing to modify {FORBIDDEN_GRAPH}; set PREDEFINED_GRAPH to a new graph"
        )
    return graph


def _connection() -> tg.TigerGraphConnection:
    graph = _graph_name()
    host = os.getenv("TG_HOST")
    token = os.getenv("TG_API_KEY")
    username = os.getenv("TG_USERNAME")
    password = os.getenv("TG_PASSWORD")
    auth_mode = os.getenv("TG_AUTH_MODE", "token").lower()
    if not host or (auth_mode == "token" and not token) or (
        auth_mode == "basic" and not (username and password)
    ):
        raise RuntimeError(
            "TG_HOST and either TG_API_KEY or TG_USERNAME/TG_PASSWORD must be set"
        )

    connection_args = {
        "host": host,
        "graphname": graph,
        "restppPort": os.getenv("TG_RESTPP_PORT", "443"),
        "gsPort": os.getenv("TG_GS_PORT", "443"),
    }
    if auth_mode == "basic":
        connection_args.update(username=username, password=password)
    else:
        connection_args["apiToken"] = token
    connection = tg.TigerGraphConnection(**connection_args)
    connection.ai.configureGraphRAGHost(
        os.getenv("GRAPHRAG_HOST", "http://localhost:8000")
    )
    return connection


def _read_documents(limit: int | None = None, offset: int = 0) -> Iterator[dict]:
    with CORPUS_PATH.open(encoding="utf-8") as corpus:
        for line_number, line in enumerate(corpus, start=1):
            if line_number <= offset:
                continue
            if limit is not None and line_number > offset + limit:
                break
            document = json.loads(line)
            for field in ("doc_id", "title", "url", "text"):
                if not document.get(field):
                    raise ValueError(f"Line {line_number} is missing {field}")
            yield document


def _normalized_documents(limit: int | None, offset: int = 0) -> tuple[Path, int]:
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".jsonl", delete=False
    )
    output_path = Path(handle.name)
    count = 0
    try:
        with handle:
            for document in _read_documents(limit, offset):
                metadata = [
                    f"Title: {document['title']}",
                    f"Source URL: {document['url']}",
                    f"Wikidata ID: {document.get('wikidata_qid', '')}",
                    f"Wikipedia page ID: {document.get('wikipedia_pageid', '')}",
                ]
                normalized = {
                    "doc_id": document["doc_id"],
                    "doc_type": "single",
                    "content": "\n".join(metadata) + "\n\n" + document["text"],
                }
                handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                count += 1
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return output_path, count


def initialize_graph(connection: tg.TigerGraphConnection) -> None:
    graph = _graph_name()
    if not connection.check_exist_graphs(graph):
        connection.gsql(f"CREATE GRAPH {graph}()")
    connection._req(
        "POST",
        f"{connection.ai.nlqs_host}/{graph}/graphrag/initialize",
        authMode="token",
        resKey=None,
    )
    print(f"GraphRAG initialized on {graph}")


def load_corpus(connection: tg.TigerGraphConnection, limit: int | None, offset: int = 0) -> None:
    upload_path, count = _normalized_documents(limit, offset)
    try:
        ingest = connection.ai.createDocumentIngest(
            data_source="local",
            data_source_config={"data_path": str(upload_path)},
            file_format="json",
        )
        connection.ai.runDocumentIngest(
            ingest["load_job_id"],
            ingest["data_source_id"],
            ingest["data_path"],
            data_source="local",
        )
    finally:
        upload_path.unlink(missing_ok=True)
    print(f"Submitted {count} document(s) to GraphRAG")


def refresh_graph(connection: tg.TigerGraphConnection) -> None:
    connection.ai.forceConsistencyUpdate("graphrag")
    print(f"GraphRAG refresh completed on {_graph_name()}")


def query_graph(
    connection: tg.TigerGraphConnection,
    question: str,
    method: str,
    top_k: int,
    num_hops: int,
    num_seen_min: int,
) -> str:
    return query_graph_response(
        connection,
        question,
        method,
        top_k,
        num_hops,
        num_seen_min,
    )["response"]


def query_graph_response(
    connection: tg.TigerGraphConnection,
    question: str,
    method: str,
    top_k: int,
    num_hops: int,
    num_seen_min: int,
) -> dict:
    parameters = {"top_k": top_k, "verbose": True}
    if method == "hybrid":
        parameters.update(
            {
                "indices": ["DocumentChunk", "Entity"],
                "num_hops": num_hops,
                "num_seen_min": num_seen_min,
                "withHyDE": False,
            }
        )
    elif method == "similarity":
        parameters.update({"index": "DocumentChunk", "withHyDE": False})
    else:
        parameters["community_level"] = 2

    response = connection.ai.answerQuestion(
        question,
        method=method,
        method_parameters=parameters,
    )
    if not isinstance(response, dict) or "response" not in response:
        raise RuntimeError("GraphRAG returned an invalid response payload")
    return response


def retrieval_diagnostics(
    response: dict,
    gold_doc_ids: list[str] | None = None,
) -> dict[str, object]:
    retrieved_ids = _document_ids(response.get("retrieved", []))
    retrieved_ids.update(_document_ids(response.get("verbose", {})))
    gold_ids = {str(doc_id) for doc_id in gold_doc_ids or []}
    matched_ids = sorted(gold_ids.intersection(retrieved_ids))
    return {
        "Retrieved documents": len(retrieved_ids),
        "Retrieved document IDs": ", ".join(sorted(retrieved_ids)),
        "Gold documents": len(gold_ids) or None,
        "Gold IDs retrieved": len(matched_ids) if gold_ids else None,
        "Recall": len(matched_ids) / len(gold_ids) if gold_ids else None,
        "Verbose retrieval": response.get("verbose", {}),
    }


def _document_ids(value: object) -> set[str]:
    ids: set[str] = set()
    if isinstance(value, list):
        for item in value:
            ids.update(_document_ids(item))
    elif isinstance(value, dict):
        for key in ("doc_id", "document_id", "id", "docId", "v"):
            identifier = value.get(key)
            if identifier is not None:
                identifier = str(identifier)
                ids.add(identifier.split("_chunk_", 1)[0])
        for item in value.values():
            if isinstance(item, (dict, list)):
                ids.update(_document_ids(item))
    return ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and query the pre-defined corpus")
    parser.add_argument("--init", action="store_true", help="Create and initialize the dedicated graph")
    parser.add_argument("--load", action="store_true", help="Load corpus documents into GraphRAG")
    parser.add_argument("--refresh", action="store_true", help="Refresh GraphRAG indexes and communities")
    parser.add_argument("--limit", type=int, help="Load only this many documents")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many documents before loading")
    parser.add_argument("--query", help="Ask a question using GraphRAG")
    parser.add_argument(
        "--method",
        choices=("hybrid", "similarity", "community"),
        default="hybrid",
        help="GraphRAG retrieval method for --query",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-hops", type=int, default=2)
    parser.add_argument("--num-seen-min", type=int, default=1)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero")
    if args.offset < 0:
        parser.error("--offset cannot be negative")
    if args.top_k < 1 or args.num_hops < 0 or args.num_seen_min < 1:
        parser.error("--top-k and --num-seen-min must be greater than zero; --num-hops cannot be negative")
    if not any((args.init, args.load, args.refresh, args.query)):
        parser.error("choose at least one of --init, --load, --refresh, or --query")
    return args


def main() -> None:
    args = _parse_args()
    connection = _connection()
    if args.init or args.load:
        initialize_graph(connection)
    if args.load:
        load_corpus(connection, args.limit, args.offset)
    if args.refresh:
        refresh_graph(connection)
    if args.query:
        print(query_graph(
            connection,
            args.query,
            args.method,
            args.top_k,
            args.num_hops,
            args.num_seen_min,
        ))


if __name__ == "__main__":
    main()