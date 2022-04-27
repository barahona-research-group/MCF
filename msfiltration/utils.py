import numpy as np


def node_id_to_dict(node_id):
    """
    Input: Array of node_id's
    Output: Dictionary that maps community number to node_keys
    """
    node_id = np.asarray(node_id)
    node_keys = np.arange(len(node_id))
    n_communities = np.max(node_id)
    community_dict = {}
    for i in range(n_communities + 1):
        c = np.argwhere(node_id == i).flatten()
        c_set = set(node_keys[j] for j in c)
        if len(c) > 0:
            community_dict[i] = c_set
    return community_dict
