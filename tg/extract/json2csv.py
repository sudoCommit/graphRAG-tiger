import bz2
import csv
import json
import os
import re
from collections import defaultdict


_DESC_SKIP = {"license", "source", "url", "unbiased"}


def normalize(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', name)


def make_uid(node_type, node_id):
    return f"{normalize(node_type)}_{node_id}"


def _serialize(value) -> str:
    """Normalize a field value to a plain string for CSV output.

    Lists/tuples are joined with '|'. Booleans become lowercase strings.
    Everything else is converted with str().
    """
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value) if value is not None else ""


def build_description(node: dict, ntype: str) -> str:
    """Synthesize a human-readable description from a node's available fields.

    Always starts with "<name> (<type>)".  Appends every meaningful field
    from the data dict that isn't pure provenance metadata.

    Examples:
      Gene    → "SERPINF2 (Gene): chromosome: 17. description: serpin peptidase..."
      Disease → "azoospermia (Disease)"
      Compound→ "Caffeine (Compound): inchikey: InChIKey=RYYVLZVUVIJVGH..."
      Anatomy → "subclavian artery (Anatomy): mesh_id: D013348"
    """
    name = node.get("name", "")
    parts = [f"{name} ({ntype})"]

    node_data = node.get("data", {})
    for key in sorted(node_data.keys()):
        if key in _DESC_SKIP:
            continue
        val = _serialize(node_data[key]).strip()
        if val:
            parts.append(f"{key}: {val}")

    return ". ".join(parts)


def extract_hetionet(
    path: str = "tg/data/hetionet-v1.0.json.bz2",
    out_dir: str = "tg/data",
) -> None:
    os.makedirs(os.path.join(out_dir, "nodes"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "edges"), exist_ok=True)

    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data["nodes"]
    edges = data["edges"]

    # NODES → per type
    node_groups = defaultdict(list)
    node_keys_by_type = defaultdict(set)

    for node in nodes:
        ntype = normalize(node["kind"])
        node_groups[ntype].append(node)

        for k in node.get("data", {}):
            node_keys_by_type[ntype].add(k)

    for ntype, group in node_groups.items():
        keys = sorted(node_keys_by_type[ntype])

        file_path = os.path.join(out_dir, f"nodes/{ntype}.csv")

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)

            header = ["id", "name", "text_blob"] + keys
            writer.writerow(header)

            for node in group:
                node_id = node["identifier"]
                name = node.get("name", "")
                node_data = node.get("data", {})
                text_blob = build_description(node, ntype)

                row = [node_id, name, text_blob] + [_serialize(node_data.get(k, "")) for k in keys]
                writer.writerow(row)

        print(f"Vertex CSV: {file_path}")

    # EDGES → per type
    edge_groups = defaultdict(list)
    edge_keys_by_type = defaultdict(set)

    for edge in edges:
        s_type, s_id = edge["source_id"]
        t_type, t_id = edge["target_id"]

        s_type_n = normalize(s_type)
        t_type_n = normalize(t_type)
        relation = normalize(edge["kind"])

        edge_type = f"{s_type_n}_{relation}_{t_type_n}"

        edge_keys_by_type[edge_type].update(edge.get("data", {}).keys())

        edge_groups[edge_type].append(edge)

        if edge.get("direction") == "both":
            reverse_type = f"{t_type_n}_{relation}_{s_type_n}"
            edge_groups[reverse_type].append(edge)
            edge_keys_by_type[reverse_type].update(edge.get("data", {}).keys())

    for etype, group in edge_groups.items():
        keys = sorted(edge_keys_by_type[etype])
        file_path = os.path.join(out_dir, f"edges/{etype}.csv")

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)

            header = ["source_id", "target_id"] + keys
            writer.writerow(header)

            for edge in group:
                s_type, s_id = edge["source_id"]
                t_type, t_id = edge["target_id"]

                # reverse handling
                if etype.startswith(normalize(t_type)):
                    s_id, t_id = t_id, s_id

                edge_data = edge.get("data", {})

                row = [s_id, t_id] + [_serialize(edge_data.get(k, "")) for k in keys]
                writer.writerow(row)

        print(f"Edge CSV: {file_path}")

    print("\n✅ Extraction complete (TigerGraph-ready)")


if __name__ == "__main__":
    extract_hetionet()
