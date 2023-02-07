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

from msfiltration import MSF
from msfiltration.graph_sampling import multiscale_sbm

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
    sizes = [int(n / 3), int(n / 3), int(n / 3)]
    probs = [[0.25, 0.02, 0.01], [0.02, 0.25, 0.02], [0.01, 0.02, 0.3]]

    # compute persistence for different realisations
    for i in range(n_realisations):

        print("Starting with realisation {} ...".format(i))

        # Sample single scale SBM network
        G = nx.stochastic_block_model(sizes, probs, seed=0)
        A = nx.adjacency_matrix(G).toarray()

        # initialise MSF object
        msf = MSF()

        # run MS analysis, build filtration and compute persistence
        msf.fit_transform(
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

        # add persistence of different dimensions to list
        persistences.append(
            [msf.filtration.persistence_intervals_in_dimension(dim) for dim in range(4)]
        )

    ########################################
    # bottleneck and Wasserstein distances #
    ########################################

    print("Computing bottleneck and Wasserstein distance ...")

    # initialsise bottleneck and wasserstein distances for different dimensions
    bottleneck_distances = np.zeros((n_realisations, n_realisations, 4))
    wasserstein_distances = np.zeros((n_realisations, n_realisations, 4))

    for i in range(n_realisations):
        for j in range(i + 1, n_realisations):
            for dim in range(4):
                # compare two persistance diagrams for fixed dimension
                Dgm_i = persistences[i][dim]
                Dgm_j = persistences[j][dim]

                # compute bottleneck distance
                bottleneck_distances[i, j, dim] = gd.bottleneck_distance(Dgm_i, Dgm_j)
                bottleneck_distances[j, i, dim] = bottleneck_distances[i, j, dim]

                # compute 2-Wasserstein distance
                wasserstein_distances[i, j, dim] = gd.wasserstein.wasserstein_distance(
                    Dgm_i, Dgm_j, order=2, internal_p=2, keep_essential_parts=False
                )
                wasserstein_distances[j, i, dim] = wasserstein_distances[i, j, dim]

    # storing resutls
    print("Storing results ...")

    results = {}
    results["ms_results"] = ms_results
    results["persistence"] = persistences
    results["bottleneck"] = bottleneck_distances
    results["wasserstein"] = wasserstein_distances

    # get current time
    time = str(datetime.datetime.now().strftime("%m-%d_%H:%M"))

    # store distances
    if server:
        root = str(Path.cwd()) + "/"
    else:
        root = str(Path.cwd()) + "/experiments/"

    with open(root + "stability_singlescale_sbm_" + time + ".pkl", "wb") as handle:
        pickle.dump(
            results, handle, protocol=pickle.HIGHEST_PROTOCOL,
        )

