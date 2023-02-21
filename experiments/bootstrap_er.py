import datetime
import gudhi as gd
import gudhi.wasserstein
import networkx as nx
import numpy as np
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

from msfiltration import MCF
from msfiltration.graph_sampling import multiscale_sbm
from msfiltration.msf_bootstrapping import msf_bootstrap

if __name__ == "__main__":

    # Set MS parameters
    n_time = 200
    min_time = -1.5
    max_time = 0.5
    n_workers = 15

    # initialise list of ms results
    ms_results = []

    # initialise list of persistences
    persistences = []

    ###################
    # MSF persistence #
    ###################

    n_realisations = 100
    n_nodes = 270
    n_edges = 3450

    # compute persistence for different realisations
    for i in range(n_realisations):

        print("Starting with realisation {} ...".format(i))

        # Sample ER network
        G = nx.gnm_random_graph(n_nodes, n_edges, seed=i)
        A = nx.adjacency_matrix(G).toarray()

        # initialise MSF object
        msf = MCF()

        # run MS analysis
        msf.markov_stability_analysis(
            A,
            min_time,
            max_time,
            n_time,
            n_workers,
            with_ttprime=True,
            with_optimal_scales=True,
        )

        # add MS results to list
        ms_results.append(msf.ms_results)

        # bootstrap MSF to compute persistent homology
        persistences_bootstrapped = msf_bootstrap(
            msf.partitions, msf.log_times, 100, 20, seed=i
        )

        # add persistence of different dimensions to list
        persistences.append(persistences_bootstrapped)

    # storing resutls
    print("Storing results ...")

    results = {}
    results["ms_results"] = ms_results
    results["persistence"] = persistences

    # get current time
    time = str(datetime.datetime.now().strftime("%m-%d_%H:%M"))

    # store distances
    if server:
        root = str(Path.cwd()) + "/results/"
    else:
        root = str(Path.cwd()) + "/experiments/results/"

    with open(root + "bootstrap_er_" + time + ".pkl", "wb") as handle:
        pickle.dump(
            results, handle, protocol=pickle.HIGHEST_PROTOCOL,
        )

