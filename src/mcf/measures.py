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
    death_count[mcf.n_partitions] = np.sum(all_deaths == np.inf)

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


def compute_bettis(mcf):
    """Compute Betti curves from the persistence intervals."""
    filtration_indices = np.asarray(mcf.filtration_indices, dtype=float)
    bettis = []

    for dim in range(3):
        dgm = mcf.filtration_gudhi.persistence_intervals_in_dimension(dim)
        if len(dgm) == 0:
            bettis.append(np.zeros(mcf.n_partitions))
            continue
        # count intervals alive at scale t, i.e. birth <= t < death
        alive = (dgm[:, 0] <= filtration_indices[:, None]) & (
            filtration_indices[:, None] < dgm[:, 1]
        )
        bettis.append(alive.sum(axis=1).astype(float))

    return bettis[0], bettis[1], bettis[2]


def _average_over_scales(values, filtration_indices):
    """Average of a piecewise-constant function of scale, weighted by the
    lengths of the scale intervals (see average conflict measures in our
    paper), where the interval at the last scale is the average gap. Reduces
    to the arithmetic mean for equidistant filtration indices."""
    values = np.asarray(values, dtype=float)
    filtration_indices = np.asarray(filtration_indices, dtype=float)

    if len(filtration_indices) == 1:
        return values[0]

    last_gap = (filtration_indices[-1] - filtration_indices[0]) / (
        len(filtration_indices) - 1
    )
    gaps = np.append(np.diff(filtration_indices), last_gap)

    return np.sum(values * gaps) / np.sum(gaps)


def compute_k_conflict_difference(mcf):
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
