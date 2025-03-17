"""Code to sample from multiscale SBM."""

import numpy as np


class SBM:
    """Class to sample from multiscale SBM."""

    def __init__(self, N, seed=42):
        """Initialise multiscale SBM object.

        Parameters:
            N (int): Number of nodes.
            seed (float): Seed for random number generator.
        """
        self.levels = []
        self.P = np.zeros((N, N))
        self.n_nodes = N
        self.rng = np.random.default_rng(seed)

    @property
    def n_expected_edges(self):
        """Computes number of expected edges."""
        return self.P.sum() / 2

    @property
    def expected_sparsity(self):
        """Computes expected sparsity."""
        return 2 * self.n_expected_edges / (self.n_nodes * (self.n_nodes - 1))

    @property
    def n_levels(self):
        """Returns number of levels."""
        return len(self.levels)

    @property
    def labels(self):
        """Compute cluster labels at different levels."""
        # obtain labels
        labels = []
        for partition in self.levels:
            # label is given by column of F matrix
            y = partition["F"] @ np.arange(partition["F"].shape[1])
            labels.append(y)
        return labels

    def add_level(self, n_blocks=1, p_in=0.1, p_out=0, weight=1):
        """Add partition to multiscale SBM.

        Parameters:
            n_blocks (int): Number of blocks, equally divided (up to remainders).
            p_in (float): Probability (from 0 to 1) of connection for two nodes
                within the same block.
            p_out (float): Probability (from 0 to 1) of connection for two nodes
                across different blocks.
            weight (float): Weight for this level in the multiscale SBM.
        """

        # for simplicty all blocks have the same size
        s_blocks = self.n_nodes / n_blocks

        # define partition indicator matrix F
        F = np.zeros((self.n_nodes, n_blocks), dtype=int)
        for i in range(self.n_nodes):
            F[i, int(i // s_blocks)] = 1

        if n_blocks > 1:
            # define affinity matrix
            Omega = p_out * np.ones((n_blocks, n_blocks))
            Omega += np.diag((p_in - p_out) * np.ones(n_blocks))

            # define probability matrix
            P_level = F @ Omega @ F.T

        elif n_blocks == 1:
            # define probability matrix in case of single block (ER model)
            P_level = p_in * np.ones((self.n_nodes, self.n_nodes))

        # set diagonal zero to avoid self-loops
        np.fill_diagonal(P_level, 0)

        # add to levels
        self.levels.append({"P": P_level, "F": F, "w": weight})

        # recompute combined probability matrix
        self._combine_probabilities()
        return self.P

    def _combine_probabilities(self):
        """Compute probability matrix from different layers."""

        # obtain combined probability matrix as convex sum over levels
        self.P = np.zeros((self.n_nodes, self.n_nodes))
        total_weight = 0
        for partition in self.levels:
            self.P += partition["w"] * partition["P"]
            total_weight += partition["w"]
        self.P /= total_weight

        return self.P

    def sample(self, with_shuffle=True):
        """Sample from multiscale SBM.

        Parameters:
            with_shuffle (bool): Permutes node IDs if true.
        """

        # sample from Bernoulli(P)
        U = self.rng.uniform(size=(self.n_nodes, self.n_nodes))
        A_unsymmetric = np.asarray(U <= self.P, dtype=int)

        # make symmetric by copying upper triangular part
        A = np.tril(A_unsymmetric) + np.tril(A_unsymmetric).T

        # shuffle rows and columns to rewire graph
        permutation = np.arange(self.n_nodes)
        if with_shuffle:
            self.rng.shuffle(permutation)

        A = A[permutation]  # shuffle rows
        A = (A.T[permutation]).T  # shuffle columns consistently

        return A, permutation
