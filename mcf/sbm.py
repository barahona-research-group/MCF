"""Code to sample from multiscale SBM."""

import numpy as np


class SBM:
    """Class to sample from multiscale SBM."""

    def __init__(self, N, seed=42):
        self.level = []
        self.n_nodes = N
        self.rng = np.random.default_rng(seed)

    def add_level(self, n_blocks, p_in, p_out, weight):
        """ "Add partition to multiscale SBM."""

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
            P = F @ Omega @ F.T

        elif n_blocks == 1:
            P = p_in * np.ones((self.n_nodes, self.n_nodes))

        # set diagonal zero to avoid self-loops
        np.fill_diagonal(P, 0)

        self.level.append({"P": P, "H": F, "w": weight})

    def sample(self):
        """Sample from multiscale SBM."""

        # obtain combined probability matrix as convex sum
        P = np.zeros((self.n_nodes, self.n_nodes))

        total_weight = 0

        for partition in self.level:
            P += partition["w"] * partition["P"]
            total_weight += partition["w"]

        P /= total_weight

        # sample from Bernoulli(P)
        U = self.rng.uniform(size=(self.n_nodes, self.n_nodes))
        A_unsymmetric = np.asarray(U <= P, dtype=int)

        # make symmetric by copying upper triangular part
        A = np.tril(A_unsymmetric) + np.tril(A_unsymmetric).T

        return A, P
