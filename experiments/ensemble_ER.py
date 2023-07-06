import datetime
import gudhi as gd
import gudhi.wasserstein
import networkx as nx
import numpy as np
import pickle
import pygenstability as pgs
from scipy.linalg import block_diag

import sys
from pathlib import Path

server = True

if server:
    root = str(Path.cwd()) + "/"
else:
    root = str(Path.cwd()) + "/experiments/"

from mcf import MCF
from mcf.measures import (
    compute_partition_size,
    compute_birth_count,
    compute_death_count,
)


def block(n, th, rng):
    A = rng.uniform(0, 1, (n, n))
    A[A < th] = 0.0
    A[A > th] = 1.0
    A = (A + A.T) / 2
    return A


if __name__ == "__main__":

    n_realisations = 100

    # Set MS parameters
    n_scale = 200
    min_scale = -1.5
    max_scale = 0.5
    n_workers = 15

    # initialise lists of results
    ms_results = []
    persistences = []
    bettis = []
    size_partitions = []
    persistent_hierarchy = []
    persistent_conflict = []
    average_persistent_hierarchy = []

    # network model parameters
    n = 270
    m = 3473

    # compute persistence for different realisations
    for i in range(n_realisations):

        print("Starting with realisation {} ...".format(i + 1))

        # construct adjacency matrix
        G = nx.gnm_random_graph(n, m, seed=i)
        A = nx.adjacency_matrix(G).toarray()

        # Run MS analysis
        MS_results = pgs.run(
            A,
            min_scale=min_scale,
            max_scale=max_scale,
            n_scale=n_scale,
            n_workers=4,
            constructor="continuous_normalized",
        )

        partitions = MS_results["community_id"]
        log_scales = np.log10(MS_results["scales"])

        # add MS results to list
        ms_results.append(MS_results)

        # initialise MCF object
        mcf = MCF()
        mcf.load_data(partitions, log_scales)

        # build filtration and compute PH
        mcf.build_filtration()
        mcf.compute_persistence()

        # add persistence of different dimensions to list
        persistences.append(
            [
                mcf.filtration_gudhi.persistence_intervals_in_dimension(dim)
                for dim in range(4)
            ]
        )

        # compute persistent hierarchy and conflict
        betti_numbers = np.zeros((n_scale, 3))
        for m in range((n_scale)):
            betti_numbers[m] = mcf.filtration_gudhi.persistent_betti_numbers(
                log_scales[m], log_scales[m]
            )
        betti_0 = betti_numbers[:, 0]
        betti_1 = betti_numbers[:, 1]
        betti_2 = betti_numbers[:, 2]

        s_partitions = compute_partition_size(mcf)

        h = betti_0 / s_partitions

        h_bar = np.mean(h[:-1])

        total_1 = np.cumsum(compute_birth_count(mcf, 1))
        resolved_1 = np.cumsum(compute_death_count(mcf, 1)[:-1])

        b_1 = compute_birth_count(mcf, 1)
        d_1 = compute_death_count(mcf, 1)[:-1]

        total_2 = np.cumsum(compute_birth_count(mcf, 2))
        resolved_2 = np.cumsum(compute_death_count(mcf, 2)[:-1])

        b_2 = compute_birth_count(mcf, 2)
        d_2 = compute_death_count(mcf, 2)[:-1]

        c_1 = b_1 - d_1
        c_2 = b_2 - d_2
        c = c_1 + c_2

        # store results
        bettis.append([betti_0, betti_1, betti_2])
        size_partitions.append(s_partitions)
        persistent_hierarchy.append(h)
        persistent_conflict.append(c)
        average_persistent_hierarchy.append(h_bar)

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
    results["bettis"] = bettis
    results["size_partitions"] = size_partitions
    results["persistent_hierarchy"] = persistent_hierarchy
    results["persistent_conflict"] = persistent_conflict
    results["average_persistent_hierarchy"] = average_persistent_hierarchy

    # get current time
    time_str = str(datetime.datetime.now().strftime("%m-%d_%H:%M"))
    time_str = time_str.replace(":", "_")

    with open(root + "ensemble_ER_" + time_str + ".pkl", "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
