"""Compute various measures from MCF persistent homology."""

import numpy as np


def _compute_death_count(mcf, dim):
    """Count the number of deaths at each scale."""

    # get all deaths for dimension
    all_deaths = mcf.filtration_gudhi.persistence_intervals_in_dimension(dim)[:, 1]

    # initialise death count array where the last entry corresponds to inf
    death_count = np.zeros(mcf.n_partitions + 1)

    # count deaths for each scale
    for i in range(mcf.n_partitions):
        death_count[i] = np.sum(all_deaths == mcf.filtration_indices[i])

    # count inf
    death_count[mcf.n_partitions] = np.sum(all_deaths == np.Inf)

    return death_count


def _compute_birth_count(mcf, dim):
    """Count the number of births at each scale."""

    # get all births for dimension
    all_births = mcf.filtration_gudhi.persistence_intervals_in_dimension(dim)[:, 0]

    # initialise birth count array
    birth_count = np.zeros(mcf.n_partitions)

    # count deaths for each scale
    for i in range(mcf.n_partitions):
        birth_count[i] = np.sum(all_births == mcf.filtration_indices[i])

    return birth_count


def compute_partition_size(mcf):
    """Compute partition sizes at all scales."""
    return np.asarray(
        [len(np.unique(mcf.partitions[i])) for i in range(mcf.n_partitions)]
    )


def compute_bettis(mcf):
    """Compute Betti curves."""
    betti_numbers = np.zeros((mcf.n_partitions, 3))
    n_dim = mcf.filtration_gudhi.dimension()

    for m in range((mcf.n_partitions)):
        betti_numbers[m][0:n_dim] = mcf.filtration_gudhi.persistent_betti_numbers(
            mcf.filtration_indices[m], mcf.filtration_indices[m]
        )

    betti_0 = betti_numbers[:, 0]
    betti_1 = betti_numbers[:, 1]
    betti_2 = betti_numbers[:, 2]

    return betti_0, betti_1, betti_2


def compute_persistent_hierarchy(mcf):
    """Compute persistent hierarchy of MCF."""

    # compute 0-dim. Betti number and partition sizes
    betti_0, _, _ = compute_bettis(mcf)
    s = compute_partition_size(mcf)

    # compute persistent hierarchy
    h = betti_0 / s

    # compute average persistent hierarchy
    # TODO: take into account non-equidistant filtration indices
    h_bar = np.mean(h[:-1])

    return h, h_bar


def compute_persistent_conflict(mcf):
    """Compute persistent conflict of MCF."""

    # count births
    b_1 = _compute_birth_count(mcf, 1)
    b_2 = _compute_birth_count(mcf, 2)

    # count deaths
    d_1 = _compute_death_count(mcf, 1)[:-1]
    d_2 = _compute_death_count(mcf, 2)[:-1]

    # compute persistent conflict
    c_1 = b_1 - d_1
    c_2 = b_2 - d_2

    # compute total persistent conflict
    c = c_1 + c_2

    return c_1, c_2, c
