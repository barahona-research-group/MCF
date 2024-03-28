"""Code for MS+MCF ensemble experiments."""

import scipy
import pickle

import gudhi as gd
import gudhi.wasserstein
import numpy as np
import pygenstability as pgs

from mcf import MCF, SBM
from tqdm import tqdm


def run_ensemble_experiment(model=SBM, n_realisations=300, n_ms_workers=100):
    """Run MS+MCF analysis on ensemble of realsations from SBM model."""

    # Set MS parameters
    n_scale = 200
    min_scale = -1.5
    max_scale = 0.5

    # initialise lists of results
    adjacencies = []
    ms_results = []
    persistences = []
    bettis = []
    size_partitions = []
    persistent_hierarchy = []
    persistent_conflict = []
    average_persistent_hierarchy = []

    print("Running MS+MCF on ensemble ...")

    # compute persistence for different realisations
    for i in tqdm(range(n_realisations)):

        # sample adjacency matrix from model
        A, _ = model.sample()
        adjacencies.append(scipy.sparse.csr_matrix(A))

        # Run MS analysis
        ms_result = pgs.run(
            A,
            min_scale=min_scale,
            max_scale=max_scale,
            n_scale=n_scale,
            n_workers=n_ms_workers,
            constructor="continuous_normalized",
            tqdm_disable=True,
        )

        partitions = ms_result["community_id"]
        log_scales = np.log10(ms_result["scales"])

        # add MS results to list
        ms_results.append(ms_result)

        # initialise MCF object
        mcf = MCF()
        mcf.load_data(partitions, log_scales)

        # build filtration and compute PH
        mcf.build_filtration(tqdm_disable=True)
        mcf.compute_persistence()

        # add persistence of different dimensions to list
        persistences.append(
            [
                mcf.filtration_gudhi.persistence_intervals_in_dimension(dim)
                for dim in range(4)
            ]
        )

        # compute Betti numbers
        betti_0, betti_1, betti_2 = mcf.compute_bettis()

        # compute size of partitions
        s_partitions = mcf.compute_partition_size()

        # compute persistent hierarchy
        h, h_bar = mcf.compute_persistent_hierarchy()

        # compute persistent conflict
        c_1, c_2, c = mcf.compute_persistent_conflict()

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

    for i in tqdm(range(n_realisations)):
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
    results = {}
    results["adjacencies"] = adjacencies
    results["ms_results"] = ms_results
    results["persistence"] = persistences
    results["bottleneck"] = bottleneck_distances
    results["wasserstein"] = wasserstein_distances
    results["bettis"] = bettis
    results["size_partitions"] = size_partitions
    results["persistent_hierarchy"] = persistent_hierarchy
    results["persistent_conflict"] = persistent_conflict
    results["average_persistent_hierarchy"] = average_persistent_hierarchy

    return results


if __name__ == "__main__":

    # define shared experiment parameters
    N_REALISATIONS = 200
    N_WORKERS = 60

    # define shared model parameters
    N = 270
    P_OUT = 0.028

    ######
    # ER #
    ######

    print("### ER ###")

    # construct ER model
    p_in_er = 0.09638
    er = SBM(N, seed=0)
    er.add_level(n_blocks=1, p_in=p_in_er, p_out=0, weight=1)

    # run ensemble experiment
    results_er = run_ensemble_experiment(
        model=er, n_realisations=N_REALISATIONS, n_ms_workers=N_WORKERS
    )

    # store results
    with open("results_ensemble_er.pkl", "wb") as handle:
        pickle.dump(results_er, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ########
    # sSBM #
    ########

    print("### sSBM ###")

    # construct sSBM model
    p_in_ssbm = 0.23465
    ssbm = SBM(N, seed=1)
    ssbm.add_level(n_blocks=3, p_in=p_in_ssbm, p_out=P_OUT, weight=3)

    # run ensemble experiment
    results_ssbm = run_ensemble_experiment(
        model=ssbm, n_realisations=N_REALISATIONS, n_ms_workers=N_WORKERS
    )

    # store results
    with open("results_ensemble_ssbm.pkl", "wb") as handle:
        pickle.dump(results_ssbm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ########
    # mSBM #
    ########

    print("### mSBM ###")

    # construct mSBM model
    p_in_msbm = 0.95839
    msbm = SBM(N, seed=2)
    msbm.add_level(n_blocks=3, p_in=p_in_msbm, p_out=P_OUT, weight=3)
    msbm.add_level(n_blocks=9, p_in=p_in_msbm, p_out=P_OUT, weight=9)
    msbm.add_level(n_blocks=27, p_in=p_in_msbm, p_out=P_OUT, weight=27)

    # run ensemble experiment
    results_msbm = run_ensemble_experiment(
        model=msbm, n_realisations=N_REALISATIONS, n_ms_workers=N_WORKERS
    )

    # store results
    with open("results_ensemble_msbm.pkl", "wb") as handle:
        pickle.dump(results_msbm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ###########
    # nh-mSBM #
    ###########

    print("### nh-mSBM ###")

    # construct nh-mSBM model
    p_in_nhmsbm = 0.85877
    nhmsbm = SBM(N, seed=3)
    nhmsbm.add_level(n_blocks=3, p_in=p_in_nhmsbm, p_out=P_OUT, weight=3)
    nhmsbm.add_level(n_blocks=5, p_in=p_in_nhmsbm, p_out=P_OUT, weight=5)
    nhmsbm.add_level(n_blocks=27, p_in=p_in_nhmsbm, p_out=P_OUT, weight=27)

    # run ensemble experiment
    results_nhmsbm = run_ensemble_experiment(
        model=nhmsbm, n_realisations=N_REALISATIONS, n_ms_workers=N_WORKERS
    )

    # store results
    with open("results_ensemble_nhmsbm.pkl", "wb") as handle:
        pickle.dump(results_nhmsbm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ##########################
    # WASSERSTEIN COMPARISON #
    ##########################

    print("### Wasserstein comparison ###")

    # combine results
    n_realisations_total = 4 * N_REALISATIONS
    combined_results = {
        0: results_er,
        1: results_ssbm,
        2: results_msbm,
        3: results_nhmsbm,
    }
    # initialsise wasserstein distances for different dimensions
    wasserstein_distances = np.zeros((n_realisations_total, n_realisations_total, 4))

    for i in tqdm(range(n_realisations_total)):
        for j in range(i + 1, n_realisations_total):

            if i // N_REALISATIONS == j // N_REALISATIONS:
                # we have already done the comparison within the same model
                wasserstein_distances[i, j] = combined_results[i // N_REALISATIONS][
                    "wasserstein"
                ][i % N_REALISATIONS, j % N_REALISATIONS]
                wasserstein_distances[j, i] = wasserstein_distances[i, j]

            for dim in range(4):
                # compare two persistance diagrams for fixed dimension
                Dgm_i = combined_results[i // N_REALISATIONS]["persistence"][
                    i % N_REALISATIONS
                ][dim]
                Dgm_j = combined_results[j // N_REALISATIONS]["persistence"][
                    j % N_REALISATIONS
                ][dim]

                # compute 2-Wasserstein distance
                wasserstein_distances[i, j, dim] = gd.wasserstein.wasserstein_distance(
                    Dgm_i, Dgm_j, order=2, internal_p=2, keep_essential_parts=False
                )
                wasserstein_distances[j, i, dim] = wasserstein_distances[i, j, dim]

    # store results
    np.save("results_comparison_wasserstein", wasserstein_distances, allow_pickle=False)

    ########################
    # FROBENIUS COMPARISON #
    ########################

    print("### Frobenius comparison ###")

    # compute Frobenius distances
    Frobenius_distances = np.zeros((n_realisations_total, n_realisations_total))

    for i in tqdm(range(n_realisations_total)):
        A_i = combined_results[i // N_REALISATIONS]["adjacencies"][
            i % N_REALISATIONS
        ].todense()
        for j in range(i + 1, n_realisations_total):
            A_j = combined_results[j // N_REALISATIONS]["adjacencies"][
                j % N_REALISATIONS
            ].todense()
            Frobenius_distances[i, j] = np.linalg.norm(A_i - A_j)
            Frobenius_distances[j, i] = Frobenius_distances[i, j]

    # store results
    np.save("results_comparison_frobenius", Frobenius_distances, allow_pickle=False)

    # reshuffle graph to not use ground truth labels
    rng = np.random.default_rng(N_REALISATIONS)
    adjacencies_rewired = []

    for i in range(n_realisations_total):
        A = combined_results[i // N_REALISATIONS]["adjacencies"][i % N_REALISATIONS]
        permutation = np.arange(270)
        rng.shuffle(permutation)

        B = A.todense()[permutation]

        C = (B.T)[permutation]
        C = C.T

        adjacencies_rewired.append(C)

    # compute Frobenius distances
    Frobenius_distances_random = np.zeros((n_realisations_total, n_realisations_total))

    for i in tqdm(range(n_realisations_total)):
        for j in range(i + 1, n_realisations_total):

            Frobenius_distances_random[i, j] = np.linalg.norm(
                adjacencies_rewired[i] - adjacencies_rewired[j]
            )
            Frobenius_distances_random[j, i] = Frobenius_distances_random[i, j]

    # store results
    np.save(
        "results_comparison_frobenius_random", Frobenius_distances, allow_pickle=False
    )
