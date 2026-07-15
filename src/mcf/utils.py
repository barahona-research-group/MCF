"""Utils to process partition data."""

import numpy as np
import pandas as pd


def _partition_to_clusters(partition):
    """Return the clusters of a partition as arrays of member indices, one per
    cluster label, in ascending label order. Single O(N log N) pass."""
    partition = np.asarray(partition)
    if partition.size == 0:
        return []
    _, inverse = np.unique(partition, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    return np.split(order, np.cumsum(np.bincount(inverse))[:-1])


def node_id_to_dict(node_id):
    """Obtains communities from partition encoded as array.
    Input: Array of node_id's
    Output: Dictionary that maps community number to node_keys
    """
    node_id = np.asarray(node_id)
    return {
        int(label): frozenset(members.tolist())
        for label, members in zip(np.unique(node_id), _partition_to_clusters(node_id))
    }


def _get_partition_clusters(partitions):
    """Transform list of node ids to list of list of clusters as frozensets."""
    return [
        [frozenset(members.tolist()) for members in _partition_to_clusters(partition)]
        for partition in partitions
    ]


def _cluster_id_preprocessing(partitions):
    """Obtain ids for distinct clusters in sequence and store their first
    occurence. This is required to compute the ids for the points in the
    nerve-based MCF."""
    # we store cluster indices of new clusters per partition
    partitions_c_ind = []
    # and a dictionary that maps cluster indices to sets
    ind_to_c = {}

    # store ids of clusters seen already
    c_to_ind = {}

    # iterate through all partitions in sequence
    for partition in partitions:
        n_clusters_before = len(c_to_ind)
        for members in _partition_to_clusters(partition):
            c = frozenset(members.tolist())
            # add cluster if new
            if c not in c_to_ind:
                c_to_ind[c] = len(c_to_ind)
                ind_to_c[c_to_ind[c]] = c
        # store all indices of new clusters per partition
        partitions_c_ind.append(np.arange(n_clusters_before, len(c_to_ind)))

    return partitions_c_ind, ind_to_c


def compute_true_overlaps(partitions):
    """Compute the set of true overlaps (Definition 21 in our paper): points
    that lie in the intersection of two non-nested clusters of the sequence.

    All distinct clusters containing a point x intersect pairwise (in x), so x
    is a true overlap iff these clusters, sorted by size, do not form a chain
    under inclusion.
    """
    partitions = [np.asarray(p) for p in partitions]
    n_points = len(partitions[0])
    n_partitions = len(partitions)

    # assign global ids to distinct clusters and record membership per point
    clusters = []
    c_to_ind = {}
    point_cluster_ids = np.empty((n_partitions, n_points), dtype=np.int64)
    for m, partition in enumerate(partitions):
        for members in _partition_to_clusters(partition):
            c = frozenset(members.tolist())
            if c not in c_to_ind:
                c_to_ind[c] = len(clusters)
                clusters.append(c)
            point_cluster_ids[m, members] = c_to_ind[c]

    sizes = np.array([len(c) for c in clusters])

    # memoised inclusion checks between clusters
    nested = {}

    def is_nested(a, b):
        if (a, b) not in nested:
            nested[(a, b)] = clusters[a] <= clusters[b]
        return nested[(a, b)]

    true_overlaps = []
    for x in range(n_points):
        gids = np.unique(point_cluster_ids[:, x])
        gids = gids[np.argsort(sizes[gids], kind="stable")]
        if any(not is_nested(int(a), int(b)) for a, b in zip(gids[:-1], gids[1:])):
            true_overlaps.append(x)

    return np.array(true_overlaps, dtype=np.int64)

def _moving_average(x, w):
    """Computes moving average for given window size (from MCF repo)."""
    return np.roll(
        np.asarray(pd.Series(x).rolling(window=w, win_type="triang").mean()),
        -int(w / 2),
    )
