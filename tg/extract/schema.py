import json
import re


def normalize(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', name)


def generate_tigergraph_schema(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)

    metanodes = meta["metanode_kinds"]
    metaedges = meta["metaedge_tuples"]

    vertex_defs = []
    edge_defs = []

    # -------------------------
    # VERTICES
    # -------------------------
    for node in metanodes:
        vtype = normalize(node)

        stmt = f"""
CREATE VERTEX {vtype} (
    PRIMARY_ID id STRING,
    name STRING
) WITH primary_id_as_attribute="true";
"""
        vertex_defs.append(stmt.strip())

    # -------------------------
    # EDGES
    # -------------------------
    for src, tgt, relation, direction in metaedges:
        src_v = normalize(src)
        tgt_v = normalize(tgt)
        rel = normalize(relation)

        edge_name = f"{src_v}_{rel}_{tgt_v}"

        if direction == "both":
            stmt = f"""
CREATE UNDIRECTED EDGE {edge_name} (
    FROM {src_v},
    TO {tgt_v}
);
"""
        else:
            stmt = f"""
CREATE DIRECTED EDGE {edge_name} (
    FROM {src_v},
    TO {tgt_v}
);
"""

        edge_defs.append(stmt.strip())

    # -------------------------
    # FINAL GRAPH
    # -------------------------
    graph_parts = [normalize(n) for n in metanodes]
    graph_parts += [
        f"{normalize(s)}_{normalize(r)}_{normalize(t)}"
        for s, t, r, _ in metaedges
    ]

    graph_stmt = f"""
CREATE GRAPH HetionetGraph (
    {", ".join(graph_parts)}
);
""".strip()

    return "\n\n".join(vertex_defs + edge_defs + [graph_stmt])


def save_schema(meta_path, out_file="schema.gsql"):
    schema = generate_tigergraph_schema(meta_path)

    with open(out_file, "w") as f:
        f.write(schema)

    print(f"✅ Schema saved to {out_file}")


if __name__ == "__main__":
    save_schema(
        "data/hetionet-schema.json",
        "data/hetionet_schema.gsql"
    )
