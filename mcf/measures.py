import gudhi as gd
import numpy as np


def compute_death_count(mcf, dim):

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


def compute_birth_count(mcf, dim):

    # get all births for dimension
    all_births = mcf.filtration_gudhi.persistence_intervals_in_dimension(dim)[:, 0]

    # initialise birth count array
    birth_count = np.zeros(mcf.n_partitions)

    # count deaths for each scale
    for i in range(mcf.n_partitions):
        birth_count[i] = np.sum(all_births == mcf.filtration_indices[i])

    return birth_count


def compute_combined_death_count(mcf, dimensions):

    # initialise combined death density array where the last entry corresponds to inf
    combined_death_count = np.zeros(mcf.n_partitions + 1)

    # sum of densities for different dimensions
    for dim in dimensions:

        combined_death_count += compute_death_count(mcf, dim)

    return combined_death_count


def compute_combined_birth_count(mcf, dimensions):

    # initialise combined death density array where the last entry corresponds to inf
    combined_birth_count = np.zeros(mcf.n_partitions + 1)

    # sum of densities for different dimensions
    for dim in dimensions:

        combined_birth_count += compute_birth_count(mcf, dim)

    return combined_birth_count


def compute_partition_size(mcf):
    # compute rank of partition matrix
    return np.asarray(
        [len(np.unique(mcf.partitions[i])) for i in range(mcf.n_partitions)]
    )


def compute_beta_0(mcf):
    # compute beta_0
    death_count_0 = compute_death_count(mcf, 0)
    return np.sum(death_count_0) - np.cumsum(death_count_0[:-1])  # CORRECT?


def compute_persistent_hierarchy(mcf):
    return compute_beta_0(mcf) / compute_partition_size(mcf)


def compute_persistent_conflict(mcf, dim):
    n_conflicts = np.cumsum(compute_birth_count(mcf, dim))
    n_resolved_conflicts = np.cumsum(compute_death_count(mcf, dim)[:-1])
    return (n_conflicts - n_resolved_conflicts) / np.maximum(n_conflicts, 1)

