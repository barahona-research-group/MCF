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
