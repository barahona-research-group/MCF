import networkx as nx
import pickle

import sys
from pathlib import Path

server = True

if server:
    module_path = str(Path.cwd().parents[0])

else:
    module_path = str(Path.cwd().parents[0]) + "/MSFiltration"


if module_path not in sys.path:
    sys.path.append(module_path)

from msfiltration import MSF


# define root
if server:
    root = str(Path.cwd()) + "/adk_graph/"
else:
    root = str(Path.cwd()) + "/experiments/adk_graph/"

# Load graph in networkx, i.e. closed conformation of adk
G = nx.read_gpickle(root + "2RGX.gpickle")

# Get sparse adajacency matrix
A = nx.adjacency_matrix(G)

# initialise MSF object
msf = MSF()

# set MS parameters
min_time = 0
max_time = 5
n_time = 300
constructor = "continuous_combinatorial"
n_workers = 100

# run MS analysis
msf.markov_stability_analysis(
    A,
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
        msf.ms_results, handle, protocol=pickle.HIGHEST_PROTOCOL,
    )

