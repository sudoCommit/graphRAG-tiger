import os
import json
import csv
import re
from collections import defaultdict


def normalize(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', name)


def make_uid(node_type, node_id):
    return f"{normalize(node_type)}_{node_id}"


def extract_hetionet(path="data/hetionet.json", out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)

    with open(path) as f:
        data = json.load(f)

    nodes = data["nodes"]
    edges = data["edges"]

    # -------------------------
    # NODES → per type
    # -------------------------
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

            header = ["id", "name"] + keys
            writer.writerow(header)

            for node in group:
                node_id = node["identifier"]
                name = node.get("name", "")
                data = node.get("data", {})

                row = [node_id, name] + [data.get(k, "") for k in keys]
                writer.writerow(row)

        print(f"Vertex CSV: {file_path}")

    # -------------------------
    # EDGES → per type
    # -------------------------
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

                data = edge.get("data", {})

                row = [s_id, t_id] + [data.get(k, "") for k in keys]
                writer.writerow(row)

        print(f"Edge CSV: {file_path}")

    print("\n✅ Extraction complete (TigerGraph-ready)")


if __name__ == "__main__":
    extract_hetionet()
