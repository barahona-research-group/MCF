"""Tests for partition processing utils and measure helpers."""

import numpy as np
import numpy.testing as npt

from mcf.measures import _average_over_scales
from mcf.utils import (
    _cluster_id_preprocessing,
    _partition_to_clusters,
    compute_true_overlaps,
    node_id_to_dict,
)

from conftest import TOY_PARTITIONS, hierarchical_sequence


def test_partition_to_clusters_matches_brute_force():
    """Clusters from the vectorised helper match a per-label argwhere scan."""
    rng = np.random.default_rng(0)
    for labels in [
        rng.integers(0, 7, 30),
        np.array([10, 3, 10, 7, 3]),  # non-contiguous labels
        np.zeros(5, dtype=int),
        np.arange(6),  # singletons
    ]:
        clusters = _partition_to_clusters(labels)
        expected = [
            np.argwhere(labels == value).flatten() for value in np.unique(labels)
        ]
        assert len(clusters) == len(expected)
        for members, expected_members in zip(clusters, expected):
            npt.assert_array_equal(members, expected_members)


def test_partition_to_clusters_empty():
    """An empty partition has no clusters."""
    assert _partition_to_clusters(np.array([], dtype=int)) == []


def test_node_id_to_dict():
    """Communities are keyed by label and contain the right members."""
    communities = node_id_to_dict([10, 3, 10, 7, 3])
    assert communities == {
        3: frozenset({1, 4}),
        7: frozenset({3}),
        10: frozenset({0, 2}),
    }


def test_cluster_id_preprocessing():
    """Distinct clusters get ids in order of first occurrence."""
    partitions = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]]
    partitions_c_ind, ind_to_c = _cluster_id_preprocessing(partitions)

    # first partition introduces two clusters, the repeated partition none,
    # the last partition two new ones
    npt.assert_array_equal(partitions_c_ind[0], [0, 1])
    npt.assert_array_equal(partitions_c_ind[1], [])
    npt.assert_array_equal(partitions_c_ind[2], [2, 3])
    assert ind_to_c == {
        0: frozenset({0, 1}),
        1: frozenset({2, 3}),
        2: frozenset({0, 1, 2}),
        3: frozenset({3}),
    }


def test_compute_true_overlaps_toy_example():
    """All three points of the toy example lie in non-nested clusters."""
    npt.assert_array_equal(compute_true_overlaps(TOY_PARTITIONS), [0, 1, 2])


def test_compute_true_overlaps_hand_checked():
    """Only the point in the overlap of two non-nested clusters is found."""
    # clusters {0,1}, {2,3} and then {0,1,2}, {3}: point 2 lies in the
    # non-nested pair {2,3} and {0,1,2}, all other pairs are nested
    partitions = [[0, 0, 1, 1], [0, 0, 0, 1]]
    npt.assert_array_equal(compute_true_overlaps(partitions), [2])


def test_compute_true_overlaps_hierarchical():
    """A strictly hierarchical sequence has no true overlaps."""
    partitions, _ = hierarchical_sequence(50, 8, seed=5)
    assert len(compute_true_overlaps(partitions)) == 0


def test_average_over_scales_equidistant():
    """For equidistant indices the weighted average is the arithmetic mean."""
    values = np.array([0.0, 0.0, 0.5, 0.5, 0.0])
    npt.assert_allclose(_average_over_scales(values, [1, 2, 3, 4, 5]), 0.2)


def test_average_over_scales_non_equidistant():
    """Scale intervals weight the average, the last one by the average gap."""
    # gaps are [1, 3] and the extrapolated last gap is (4 - 0) / 2 = 2
    average = _average_over_scales([1.0, 3.0, 2.0], [0.0, 1.0, 4.0])
    npt.assert_allclose(average, (1 * 1 + 3 * 3 + 2 * 2) / 6)


def test_average_over_scales_single_scale():
    """A single scale returns its value."""
    npt.assert_allclose(_average_over_scales([0.7], [1.0]), 0.7)
