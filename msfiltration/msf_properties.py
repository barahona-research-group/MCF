import gudhi as gd
import numpy as np


def compute_death_count(msf, dim):

    # get all deaths for dimension
    all_deaths = msf.filtration.persistence_intervals_in_dimension(dim)[:, 1]

    # initialise death density array where the last entry corresponds to inf
    death_count = np.zeros(msf.n_times + 1)

    # count deaths for each scale
    for i in range(msf.n_times):
        death_count[i] = np.sum(all_deaths == msf.log_times[i])

    # count inf
    death_count[msf.n_times] = np.sum(all_deaths == np.Inf)

    return death_count


def compute_birth_count(msf, dim):

    # get all births for dimension
    all_births = msf.filtration.persistence_intervals_in_dimension(dim)[:, 0]

    # initialise death density array where the last entry corresponds to inf
    birth_count = np.zeros(msf.n_times + 1)

    # count deaths for each scale
    for i in range(msf.n_times):
        birth_count[i] = np.sum(all_births == msf.log_times[i])

    # count inf
    birth_count[msf.n_times] = np.sum(all_births == np.Inf)

    return birth_count


def compute_combined_death_count(msf, dimensions):

    # initialise combined death density array where the last entry corresponds to inf
    combined_death_count = np.zeros(msf.n_times + 1)

    # sum of densities for different dimensions
    for dim in dimensions:

        combined_death_count += compute_death_count(msf, dim)

    return combined_death_count


def compute_combined_birth_count(msf, dimensions):

    # initialise combined death density array where the last entry corresponds to inf
    combined_birth_count = np.zeros(msf.n_times + 1)

    # sum of densities for different dimensions
    for dim in dimensions:

        combined_birth_count += compute_birth_count(msf, dim)

    return combined_birth_count


def compute_rank(msf):
    # compute rank of partition matrix
    return np.asarray(
        [len(np.unique(msf.community_ids[i])) for i in range(msf.n_times)]
    )


def compute_beta_0(msf):
    # compute beta_0
    death_count_0 = compute_death_count(msf, 0)
    return np.sum(death_count_0) - np.cumsum(death_count_0)  # CORRECT?

