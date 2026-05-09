import pyTigerGraph as tg
import os
from dotenv import load_dotenv
import time


load_dotenv(override=True)

# TigerGraph instance details
HOST = "https://tg-b07aa060-e3df-4207-94fc-971616eb7348.tg-2635877100.i.tgcloud.io"
GRAPH = "HetionetGraph"
API_KEY = os.getenv("TG_API_KEY")

conn = tg.TigerGraphConnection(
    host=HOST,
    graphname=GRAPH,
    apiToken=API_KEY
)

# print(conn.check_exist_graphs(GRAPH)) 
print("✅ Connected to TigerGraph")


def schema_upload():
    print("Dropping graph if exists...")
    conn.gsql(f"DROP GRAPH {GRAPH} IF EXISTS")

    with open("tg/data/hetionet_schema.gsql", "r") as f:
        schema_query = f.read()

    statements = [stmt.strip() for stmt in schema_query.split(";") if stmt.strip()]

    print("Executing schema statements...")
    for stmt in statements:
        print(f"Executing: {stmt[:60]}...")
        conn.gsql(stmt)

    print("Installing graph...")
    conn.gsql(f"INSTALL GRAPH {GRAPH}")

    print(conn.getSchema())
    print("✅ Schema executed")


def create_and_run_vertex_jobs():
    """Create loading jobs for all vertex CSVs and run them"""
    DATA_DIR = "/home/sidharth/Desktop/GraphRAG/graphRAG-tiger/tg/data/nodes"
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".csv") or "_" in file:
            continue

        vtype = file.replace(".csv", "")
        path = os.path.join(DATA_DIR, file)
        job_name = f"load_{vtype}"
        file_tag = "f1"  # single file tag for each job
        
        print(f"📥 Creating and loading vertex: {vtype}")

        # Generate GSQL for loading job
        gsql_job = f"""
        USE GRAPH {GRAPH}
        CREATE LOADING JOB {job_name} FOR GRAPH {GRAPH} {{
            DEFINE FILENAME {file_tag};
            LOAD {file_tag} TO VERTEX {vtype} VALUES ($0, $1) USING SEPARATOR=",";
        }}
        """
        try:
            conn.gsql(gsql_job)
        except Exception as e:
            print(f"⚠️ Could not create job {job_name}: {e}")

        conn.runLoadingJobWithFile(
            filePath=path,
            fileTag=file_tag,
            jobName=job_name
        )

def create_and_run_edge_jobs():
    """Create loading jobs for all edge CSVs and run them"""
    DATA_DIR = "/home/sidharth/Desktop/GraphRAG/graphRAG-tiger/tg/data/edges"
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".csv") or "_" not in file:
            continue

        etype = file.replace(".csv", "")
        path = os.path.join(DATA_DIR, file)
        job_name = f"load_{etype}"
        file_tag = "f1"
        
        print(f"🔗 Creating and loading edge: {etype}")

        # For simplicity, assume all edges have 2 columns: from_id, to_id
        gsql_job = f"""
        USE GRAPH {GRAPH}
        CREATE LOADING JOB {job_name} FOR GRAPH {GRAPH} {{
            DEFINE FILENAME {file_tag};
            LOAD {file_tag} TO EDGE {etype} VALUES ($0, $1) USING SEPARATOR=",";
        }}
        """
        try:
            conn.gsql(gsql_job)
        except Exception as e:
            print(f"⚠️ Could not create job {job_name}: {e}")

        conn.runLoadingJobWithFile(
            filePath=path,
            fileTag=file_tag,
            jobName=job_name
        )


def main():
    schema_upload()
    time.sleep(5)
    create_and_run_vertex_jobs()
    create_and_run_edge_jobs()
    print("✅ All data loaded")

main()

# Verification
print(conn.getVertexCount("Gene"))
print(conn.getVertexCount("Disease"))

