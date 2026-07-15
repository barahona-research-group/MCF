"""Shared fixtures and helpers for the MCF test suite."""

import sys
from pathlib import Path

# always test the working tree in src/, not an installed copy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from mcf import MultiscaleClusteringFiltration

# toy example from the paper
TOY_PARTITIONS = [[0, 1, 2], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]]
TOY_FILTRATION_INDICES = [1, 2, 3, 4, 5]


@pytest.fixture
def toy_example():
    return TOY_PARTITIONS, TOY_FILTRATION_INDICES


def build_mcf(method, partitions, filtration_indices=None, max_dim=3, restrict=False):
    """Construct an MCF object and build its filtration."""
    mcf = MultiscaleClusteringFiltration(
        method=method, max_dim=max_dim, restrict_to_true_overlaps=restrict
    )
    mcf.load_data(partitions, filtration_indices)
    mcf.build_filtration(tqdm_disable=True)
    return mcf


def sorted_diagrams(mcf):
    """Compute persistence and return sorted diagrams per dimension."""
    mcf.compute_persistence()
    return [np.array(sorted(map(tuple, dgm))).reshape(-1, 2) for dgm in mcf.persistence]


def assert_diagrams_equal(dgms_a, dgms_b, dims):
    """Assert that two lists of sorted persistence diagrams agree in dims."""
    for dim in dims:
        assert dgms_a[dim].shape == dgms_b[dim].shape, f"dim {dim}: different size"
        np.testing.assert_allclose(
            dgms_a[dim], dgms_b[dim], err_msg=f"dim {dim}: diagrams differ"
        )


def random_sequence(n_points, n_partitions, n_clusters_start, n_clusters_end, seed=0):
    """Non-hierarchical sequence: independent random partitions with a
    decreasing number of clusters (coarse-graining ordering)."""
    rng = np.random.default_rng(seed)
    n_clusters = np.linspace(n_clusters_start, n_clusters_end, n_partitions)
    partitions = [rng.integers(0, max(1, round(k)), n_points) for k in n_clusters]
    return partitions, np.arange(1, n_partitions + 1)


def hierarchical_sequence(n_points, n_partitions, max_cluster_size=None, seed=0):
    """Strictly hierarchical sequence obtained by random successive merges,
    starting from singletons."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n_points)
    partitions = [labels.copy()]

    for _ in range(n_partitions - 1):
        values = np.unique(labels)
        sizes = {v: np.sum(labels == v) for v in values}
        # merge a quarter of the clusters in random disjoint pairs
        shuffled = rng.permutation(values)
        n_merges = max(1, len(values) // 4)
        for a, b in zip(shuffled[:n_merges], shuffled[n_merges : 2 * n_merges]):
            if max_cluster_size is not None and sizes[a] + sizes[b] > max_cluster_size:
                continue
            labels[labels == b] = a
            sizes[a] += sizes[b]
        partitions.append(labels.copy())

    return partitions, np.arange(1, n_partitions + 1)


def quasi_hierarchical_sequence(
    n_points, n_partitions, n_conflicts, max_cluster_size=None, seed=0
):
    """Hierarchical sequence perturbed by reassigning a few random points to a
    random other cluster of their partition."""
    rng = np.random.default_rng(seed)
    partitions, filtration_indices = hierarchical_sequence(
        n_points, n_partitions, max_cluster_size, seed=seed
    )

    for _ in range(n_conflicts):
        m = rng.integers(1, n_partitions)
        x = rng.integers(0, n_points)
        values = np.unique(partitions[m])
        if len(values) > 1:
            others = values[values != partitions[m][x]]
            partitions[m][x] = rng.choice(others)

    return partitions, filtration_indices
