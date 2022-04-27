import numpy as np

from scipy.linalg import block_diag


def block(n, th, rng):
    A = rng.uniform(0, 1, (n, n))
    A[A < th] = 0.0
    A[A > th] = 1.0
    A = (A + A.T) / 2
    return A


def multiscale_sbm(
    n0=270, n1=3, n2=9, n3=27, th0=0.995, th1=0.95, th2=0.8, th3=0.2, seed=42
):

    # construct adjacency matrix
    rng = np.random.RandomState(seed)
    A = block(n0, th0, rng)
    A += block_diag(*[block(int(n0 / n1), th1, rng) for i in range(n1)])
    A += block_diag(*[block(int(n0 / n2), th2, rng) for i in range(n2)])
    A += block_diag(*[block(int(n0 / n3), th3, rng) for i in range(n3)])

    # binarized
    A[A > 0] = 1

    # remove self-loops
    A -= np.diag(np.diag(A))

    return A
