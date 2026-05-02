import pyTigerGraph as tg
import os
from dotenv import load_dotenv


load_dotenv(override=True)

# Your TigerGraph instance details
HOST = "https://tg-b07aa060-e3df-4207-94fc-971616eb7348.tg-2635877100.i.tgcloud.io"
GRAPH = "HetionetGraph"
API_KEY = os.getenv("TIGER_GRAPH_KEY")

conn = tg.TigerGraphConnection(
    host=HOST,
    graphname=GRAPH,
    apiToken=API_KEY
)

# conn.getToken(conn.createSecret())
print(conn.check_exist_graphs(GRAPH)) 
print("✅ Connected to TigerGraph")


# def schema_upload():

#     # If graph does NOT exist yet, Run this once:
#     if not conn.graphExist(GRAPH):
#         print(f"Graph {GRAPH} dosn't exist. Creating...")
#         conn.gsql(
#             """
#                 DROP GRAPH HetionetGraph IF EXISTS;
#                 CREATE GRAPH HetionetGraph();
#             """
#         )

#     with open("data/hetionet_schema.gsql", "r") as f:
#         schema_query = f.read()

#     conn.gsql(schema_query)

#     print("✅ Schema executed")



# def load_nodes():
#     DATA_DIR = "data/nodes"
#     for file in os.listdir(DATA_DIR):
#         if not file.endswith(".csv"):
#             continue

#         # vertex files have no underscore
#         if "_" in file:
#             continue

#         vtype = file.replace(".csv", "")
#         path = os.path.join(DATA_DIR, file)

#         print(f"📥 Loading vertex: {vtype}")

#         conn.uploadFile(
#             filepath=path,
#             fileTag=vtype,
#             jobName=f"load_{vtype}"
#         )


# def load_edges(conn):
#     DATA_DIR = "data/edges"
#     for file in os.listdir(DATA_DIR):
#         if not file.endswith(".csv"):
#             continue

#         # edge files have underscore
#         if "_" not in file:
#             continue

#         etype = file.replace(".csv", "")
#         path = os.path.join(DATA_DIR, file)

#         print(f"🔗 Loading edge: {etype}")

#         conn.uploadFile(
#             filepath=path,
#             fileTag=etype,
#             jobName=f"load_{etype}"
#         )


# # Load all jobs
# jobs = conn.getLoadingJobs()

# for job in jobs:
#     print(f"▶ Running: {job}")
#     conn.runLoadingJob(job)

# print("✅ All data loaded")


# # Verification
# print(conn.getVertexCount("Gene"))
# print(conn.getVertexCount("Disease"))

