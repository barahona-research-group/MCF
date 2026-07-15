"""Equivalence tests: the standard and nerve-based MCF constructions must
always lead to the same persistent homology, and restricting to true overlaps
must preserve the persistent homology in dimensions >= 1."""

import numpy as np
import pytest

from mcf.utils import compute_true_overlaps

from conftest import (
    TOY_FILTRATION_INDICES,
    TOY_PARTITIONS,
    assert_diagrams_equal,
    build_mcf,
    hierarchical_sequence,
    quasi_hierarchical_sequence,
    random_sequence,
    sorted_diagrams,
)

# battery of test cases: name -> (partitions, filtration_indices, max_dim)
CASES = {
    "toy_example": (TOY_PARTITIONS, TOY_FILTRATION_INDICES, 3),
    "toy_example_arrays": (
        [np.array(p) for p in TOY_PARTITIONS],
        np.array(TOY_FILTRATION_INDICES),
        3,
    ),
    "random_seed_0": (*random_sequence(60, 8, 20, 4, seed=0), 3),
    "random_seed_1": (*random_sequence(60, 8, 20, 4, seed=1), 3),
    "random_seed_2": (*random_sequence(60, 8, 20, 4, seed=2), 3),
    "random_max_dim_2": (*random_sequence(40, 6, 15, 3, seed=3), 2),
    "random_max_dim_1": (*random_sequence(30, 5, 10, 3, seed=7), 1),
    "non_equidistant_indices": (
        random_sequence(50, 7, 18, 4, seed=4)[0],
        np.cumsum(np.random.default_rng(4).uniform(0.1, 2.0, 7)),
        3,
    ),
    "default_indices": (random_sequence(30, 5, 10, 3, seed=8)[0], None, 3),
    "hierarchical": (*hierarchical_sequence(50, 8, seed=5), 3),
    "quasi_hierarchical": (*quasi_hierarchical_sequence(60, 8, 5, seed=6), 3),
    "repeated_partitions": (
        [[0, 1, 2, 3], [0, 1, 2, 3], [0, 0, 1, 2], [0, 0, 1, 2], [0, 0, 0, 0]],
        [1, 2, 3, 4, 5],
        3,
    ),
    "single_partition": ([[0, 0, 1, 2]], [1], 3),
    "one_cluster": ([[0, 1, 0, 1, 2, 2, 0, 1], [0] * 8], [1, 2], 3),
    "from_singletons": (
        [list(range(6)), [0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1]],
        [1, 2, 3],
        3,
    ),
    "non_contiguous_labels": (
        [[10, 3, 10, 7], [3, 3, 10, 7], [5, 5, 5, 5]],
        [1, 2, 3],
        3,
    ),
    # only 2-point clusters whose edges form a cycle of points, so the
    # 1-dimensional homology lives in the top dimension of the complex
    "cycle_of_pairs": ([[0, 0, 1, 1], [0, 1, 1, 0]], [1, 2], 3),
}


@pytest.mark.parametrize("name", CASES)
def test_standard_and_nerve_have_same_ph(name):
    """Standard and nerve-based MCF lead to the same persistence diagrams."""
    partitions, filtration_indices, max_dim = CASES[name]

    standard = build_mcf("standard", partitions, filtration_indices, max_dim)
    nerve = build_mcf("nerve", partitions, filtration_indices, max_dim)

    assert_diagrams_equal(
        sorted_diagrams(standard), sorted_diagrams(nerve), range(max_dim)
    )


@pytest.mark.parametrize("name", CASES)
def test_standard_and_nerve_have_same_bettis(name):
    """Standard and nerve-based MCF lead to the same Betti curves."""
    partitions, filtration_indices, max_dim = CASES[name]

    standard = build_mcf("standard", partitions, filtration_indices, max_dim)
    nerve = build_mcf("nerve", partitions, filtration_indices, max_dim)
    standard.compute_persistence()
    nerve.compute_persistence()

    for betti_standard, betti_nerve in zip(
        standard.compute_bettis(), nerve.compute_bettis()
    ):
        np.testing.assert_allclose(betti_standard, betti_nerve)


@pytest.mark.parametrize("method", ["standard", "nerve"])
@pytest.mark.parametrize("name", CASES)
def test_restriction_preserves_higher_ph(name, method):
    """Restriction to true overlaps preserves the persistence diagrams in
    dimensions >= 1."""
    partitions, filtration_indices, max_dim = CASES[name]

    full = build_mcf(method, partitions, filtration_indices, max_dim)
    restricted = build_mcf(
        method, partitions, filtration_indices, max_dim, restrict=True
    )

    assert_diagrams_equal(
        sorted_diagrams(full), sorted_diagrams(restricted), range(1, max_dim)
    )


def test_hierarchical_has_trivial_higher_ph():
    """A strictly hierarchical sequence has no true overlaps and trivial
    persistent homology in dimensions >= 1."""
    partitions, filtration_indices = hierarchical_sequence(50, 8, seed=5)

    assert len(compute_true_overlaps(partitions)) == 0

    for method in ["standard", "nerve"]:
        mcf = build_mcf(method, partitions, filtration_indices, max_dim=3)
        dgms = sorted_diagrams(mcf)
        assert dgms[1].size == 0
        assert dgms[2].size == 0


@pytest.mark.parametrize("method", ["standard", "nerve"])
def test_toy_example_matches_paper(method):
    """The toy example reproduces the persistence diagrams from the paper."""
    mcf = build_mcf(method, TOY_PARTITIONS, TOY_FILTRATION_INDICES, max_dim=3)
    dgms = sorted_diagrams(mcf)

    np.testing.assert_allclose(
        dgms[0], np.array([[1.0, 2.0], [1.0, 3.0], [1.0, np.inf]])
    )
    np.testing.assert_allclose(dgms[1], np.array([[4.0, 5.0]]))
    assert dgms[2].size == 0


@pytest.mark.parametrize("method", ["standard", "nerve"])
def test_cycle_of_pairs_has_1_conflict(method):
    """The cycle of pairs produces a never-resolved 1-conflict even though the
    complex has dimension 1 (homology in the top dimension of the complex)."""
    partitions, filtration_indices, max_dim = CASES["cycle_of_pairs"]
    mcf = build_mcf(method, partitions, filtration_indices, max_dim)
    dgms = sorted_diagrams(mcf)

    np.testing.assert_allclose(dgms[1], np.array([[2.0, np.inf]]))

    _, betti_1, _ = mcf.compute_bettis()
    np.testing.assert_allclose(betti_1, np.array([0, 1]))
