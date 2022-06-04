import networkx as nx
import pickle
import sys

from pathlib import Path
from pygenstability import run

server = True

# define root
if server:
    root = str(Path.cwd()) + "/adk_graph/"
else:
    root = str(Path.cwd()) + "/experiments/adk_graph/"

# Load graph in networkx, i.e. closed conformation of adk
G = nx.read_gpickle(root + "2RGX.gpickle")

# Get sparse adajacency matrix
A = nx.adjacency_matrix(G)

# set MS parameters
min_time = 0
max_time = 5
n_time = 150
constructor = "continuous_combinatorial"
n_workers = 50

# run MS analysis
ms_results = run(
    graph=A,
    min_time=min_time,
    max_time=max_time,
    n_time=n_time,
    constructor=constructor,
    n_workers=n_workers,
    with_ttprime=True,
)

# store MS results
with open(root + "ms_adg_2rgx.pkl", "wb") as handle:
    pickle.dump(
        ms_results, handle, protocol=pickle.HIGHEST_PROTOCOL,
    )

