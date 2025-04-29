"""Utils to process partition data."""

import numpy as np


def node_id_to_dict(node_id):
    """Obtains communities from partition encoded as array.
    Input: Array of node_id's
    Output: Dictionary that maps community number to node_keys
    """
    node_id = np.asarray(node_id)
    node_keys = np.arange(len(node_id))
    n_communities = np.max(node_id)
    community_dict = {}
    for i in range(n_communities + 1):
        c = np.argwhere(node_id == i).flatten()
        c_set = frozenset(node_keys[j] for j in c)
        if len(c) > 0:
            community_dict[i] = c_set
    return community_dict


def _get_partition_clusters(partitions):
    """Transform list of node ids to list of list of clusters as frozensets."""
    partitions_clusters = []

    for partition in partitions:
        clusters = []
        # get all cluster indices in partition
        cluster_values = np.unique(partition)
        # iterate through all clusters in partitions
        for i in cluster_values:
            # define cluster as a set
            c = frozenset(np.argwhere(partition == i).flatten())
            clusters.append(c)
        partitions_clusters.append(clusters)

    return partitions_clusters


def _cluster_id_preprocessing(partitions):
    """Obtain ids for distinct clusters in sequence and store their first
    occurence. This is required to compute the ids for the points in the
    nerve-based MCF."""
    # we store cluster indices of new clusters per partition
    partitions_c_ind = []
    # and a dictionary that maps cluster indices to sets
    ind_to_c = {}

    # store clusters seen already
    all_clusters = set()
    n_clusters_total = 0

    # iterate through all partitions in sequence
    for partition in partitions:
        n_clusters_before = n_clusters_total
        # get all cluster indices in partition
        cluster_values = np.unique(partition)
        # iterate through all clusters in partitions
        for i in cluster_values:
            # define cluster as a set
            c = frozenset(np.argwhere(partition == i).flatten())
            # check if cluster is new
            if c in all_clusters:
                continue
            else:
                # add cluster if new
                all_clusters.add(c)
                ind_to_c[n_clusters_total] = c
                n_clusters_total += 1
        # store all indices of new clusters per partition
        partitions_c_ind.append(np.arange(n_clusters_before, n_clusters_total))

    return partitions_c_ind, ind_to_c
