import datetime
import gudhi as gd
import gudhi.wasserstein
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
    n_workers = 10

    # initialise list of persistences
    persistences = []

    ###################
    # MSF persistence #
    ###################

    n_realisations = 100

    # compute persistence for different realisations
    for i in range(n_realisations):

        # Sample multiscale SBM network
        A = multiscale_sbm(seed=i)

        # initialise MSF object
        msf = MSF()

        # run MS analysis, build filtration and compute persistence
        msf.fit_transform(A, min_time, max_time, n_time, n_workers)

        # add persistence of different dimensions to list
        persistences.append(
            [msf.filtration.persistence_intervals_in_dimension(dim) for dim in range(4)]
        )

        print("Finished with realisation {} ...".format(i))

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

                # compute Wasserstein distance
                wasserstein_distances[i, j, dim] = gd.wasserstein.wasserstein_distance(
                    Dgm_i, Dgm_j, keep_essential_parts=False
                )
                wasserstein_distances[j, i, dim] = wasserstein_distances[i, j, dim]

    # storing results
    print("Storing results ...")

    distances = {}
    distances["bottleneck"] = bottleneck_distances
    distances["wasserstein"] = wasserstein_distances

    # get current time
    time = str(datetime.datetime.now().strftime("%m-%d_%H:%M"))

    # store distances
    if server:
        root = str(Path.cwd()) + "/"
    else:
        root = str(Path.cwd()) + "/experiments/"

    with open(root + "stability_sbm_" + time + ".pkl", "wb") as handle:
        pickle.dump(
            distances, handle, protocol=pickle.HIGHEST_PROTOCOL,
        )
