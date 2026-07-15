"""Tests for MCF measures, run for both the standard and the nerve-based
construction: exact values on the toy example from the paper and structural
invariants on larger generated sequences."""

import numpy as np
import numpy.testing as npt
import pytest

from mcf import MultiscaleClusteringFiltration as MCF

from conftest import (
    TOY_FILTRATION_INDICES,
    TOY_PARTITIONS,
    hierarchical_sequence,
    quasi_hierarchical_sequence,
    random_sequence,
)


@pytest.fixture(params=["standard", "nerve"])
def toy_mcf(request):
    """MCF of the toy example with built filtration."""
    mcf = MCF(method=request.param)
    mcf.load_data(TOY_PARTITIONS, TOY_FILTRATION_INDICES)
    mcf.build_filtration(tqdm_disable=True)
    return mcf


def test_compute_partition_size(toy_mcf):
    """Test for computing partition sizes."""
    toy_mcf.compute_partition_size()
    npt.assert_allclose(toy_mcf.s_partitions_, np.array([3, 2, 2, 2, 1]))


def test_compute_persistence(toy_mcf):
    """Test for computing MCF persistent homology."""
    toy_mcf.compute_persistence()

    # check if persistence pairs match for 0-dim
    dgm_0 = np.array(sorted(map(tuple, toy_mcf.persistence[0])))
    npt.assert_allclose(dgm_0, np.array([[1.0, 2.0], [1.0, 3.0], [1.0, np.inf]]))

    # check if persistence pairs match for 1-dim
    assert len(toy_mcf.persistence[1]) == 1
    npt.assert_allclose(toy_mcf.persistence[1][0], np.array([4.0, 5.0]))

    # check if persistence pairs match for 2-dim
    assert len(toy_mcf.persistence[2]) == 0


def test_compute_bettis(toy_mcf):
    """Test for computing MCF Betti curves."""
    toy_mcf.compute_persistence()
    betti_0, betti_1, betti_2 = toy_mcf.compute_bettis()

    npt.assert_allclose(betti_0, np.array([3, 2, 1, 1, 1]))
    npt.assert_allclose(betti_1, np.array([0, 0, 0, 1, 0]))
    npt.assert_allclose(betti_2, np.array([0, 0, 0, 0, 0]))


def test_compute_conflict_measures(toy_mcf):
    """Test for computing MCF conflict measures."""
    toy_mcf.compute_persistence()
    toy_mcf.compute_conflict_measures()

    # check persistent 0-conflict and its scale-weighted average
    npt.assert_allclose(toy_mcf.conflict_0_, np.array([0.0, 0.0, 0.5, 0.5, 0.0]))
    npt.assert_allclose(toy_mcf.conflict_0_avg_, 0.2)

    # check average k-conflicts
    npt.assert_allclose(toy_mcf.conflict_1_avg_, 0.2)
    npt.assert_allclose(toy_mcf.conflict_2_avg_, 0.0)

    # check k-conflict differences
    npt.assert_allclose(toy_mcf.conflict_1_diff_, np.array([0.0, 0.0, 0.0, 1.0, -1.0]))
    npt.assert_allclose(toy_mcf.conflict_2_diff_, np.zeros(5))
    npt.assert_allclose(
        toy_mcf.conflict_total_diff_, np.array([0.0, 0.0, 0.0, 1.0, -1.0])
    )


def test_compute_landscapes(toy_mcf):
    """Smoke test for computing persistence landscapes."""
    toy_mcf.compute_persistence()
    toy_mcf.compute_landscapes(l_dims=[0, 1], l_k_max=2, l_resolution=50)

    assert toy_mcf.l_0_.shape == (2 * 50,)
    assert toy_mcf.l_1_.shape == (2 * 50,)
    assert toy_mcf.l_2_ is None


def test_compute_all_measures(toy_mcf):
    """Test for computing all MCF measures without writing to disk."""
    results = toy_mcf.compute_all_measures(file_path=None)

    npt.assert_allclose(results["filtration_indices"], TOY_FILTRATION_INDICES)
    assert results["max_dim"] == 3
    assert results["method"] == toy_mcf.method
    assert results["restrict_to_true_overlaps"] is False
    npt.assert_allclose(
        np.array(sorted(map(tuple, results["persistence"][0]))),
        np.array([[1.0, 2.0], [1.0, 3.0], [1.0, np.inf]]),
    )
    npt.assert_allclose(results["persistence"][1], np.array([[4.0, 5.0]]))
    assert len(results["persistence"][2]) == 0
    npt.assert_allclose(results["betti_0"], np.array([3, 2, 1, 1, 1]))
    npt.assert_allclose(results["betti_1"], np.array([0, 0, 0, 1, 0]))
    npt.assert_allclose(results["betti_2"], np.array([0, 0, 0, 0, 0]))
    npt.assert_allclose(results["s_partitions"], np.array([3, 2, 2, 2, 1]))
    npt.assert_allclose(results["conflict_0"], np.array([0.0, 0.0, 0.5, 0.5, 0.0]))
    npt.assert_allclose(results["conflict_0_avg"], 0.2)
    npt.assert_allclose(results["conflict_1_diff"], np.array([0.0, 0.0, 0.0, 1.0, -1.0]))
    npt.assert_allclose(results["conflict_2_diff"], np.zeros(5))
    npt.assert_allclose(
        results["conflict_total_diff"], np.array([0.0, 0.0, 0.0, 1.0, -1.0])
    )


# larger sequences where measures are checked via structural invariants
LARGER_SEQUENCES = {
    "random": random_sequence(60, 8, 20, 4, seed=0),
    "quasi_hierarchical": quasi_hierarchical_sequence(60, 8, 5, seed=6),
    "hierarchical": hierarchical_sequence(50, 8, seed=5),
}


@pytest.fixture(params=list(LARGER_SEQUENCES))
def larger_name(request):
    return request.param


@pytest.fixture(params=["standard", "nerve"])
def larger_results(request, larger_name):
    """All measures of a larger sequence for one construction method."""
    partitions, filtration_indices = LARGER_SEQUENCES[larger_name]
    mcf = MCF(method=request.param)
    mcf.load_data(partitions, filtration_indices)
    return mcf.compute_all_measures(file_path=None, tqdm_disable=True)


def test_betti_0_bounds(larger_results):
    """The 0-dim Betti curve is non-increasing, positive and bounded by the
    smallest number of clusters seen so far."""
    betti_0 = larger_results["betti_0"]

    assert np.all(np.diff(betti_0) <= 0)
    assert np.all(betti_0 >= 1)
    assert np.all(betti_0 <= np.minimum.accumulate(larger_results["s_partitions"]))


def test_conflict_0_range(larger_results):
    """The persistent 0-conflict starts at 0 and stays within [0, 1]."""
    conflict_0 = larger_results["conflict_0"]

    assert conflict_0[0] == 0
    assert np.all(conflict_0 >= 0)
    assert np.all(conflict_0 <= 1)
    assert 0 <= larger_results["conflict_0_avg"] <= 1


def test_k_conflict_difference_is_betti_derivative(larger_results):
    """The k-conflict differences accumulate to the k-dim Betti curves."""
    npt.assert_allclose(
        np.cumsum(larger_results["conflict_1_diff"]), larger_results["betti_1"]
    )
    npt.assert_allclose(
        np.cumsum(larger_results["conflict_2_diff"]), larger_results["betti_2"]
    )


def test_infinite_bars_match_final_bettis(larger_results):
    """The number of infinite bars per dimension equals the Betti number at
    the final scale."""
    for dim, betti in enumerate(["betti_0", "betti_1", "betti_2"]):
        dgm = larger_results["persistence"][dim]
        n_infinite = np.sum(np.isinf(dgm[:, 1])) if len(dgm) else 0
        assert n_infinite == larger_results[betti][-1]


def test_measures_agree_between_methods(larger_name):
    """Standard and nerve-based MCF lead to the same measures."""
    partitions, filtration_indices = LARGER_SEQUENCES[larger_name]

    results = {}
    for method in ["standard", "nerve"]:
        mcf = MCF(method=method)
        mcf.load_data(partitions, filtration_indices)
        results[method] = mcf.compute_all_measures(file_path=None, tqdm_disable=True)

    for key in [
        "betti_0",
        "betti_1",
        "betti_2",
        "conflict_0",
        "conflict_0_avg",
        "conflict_1_avg",
        "conflict_2_avg",
        "conflict_1_diff",
        "conflict_2_diff",
        "conflict_total_diff",
    ]:
        npt.assert_allclose(
            results["standard"][key], results["nerve"][key], err_msg=key
        )


def test_save_and_load_results(toy_mcf, tmp_path):
    """Test that results survive a save/load round trip."""
    file_path = str(tmp_path / "mcf_results.pkl")
    results = toy_mcf.compute_all_measures(file_path=file_path)

    loaded_mcf = MCF()
    loaded_mcf.load_data_from_file(file_path)

    npt.assert_allclose(loaded_mcf.filtration_indices, TOY_FILTRATION_INDICES)
    assert loaded_mcf.method == toy_mcf.method
    for dim in range(3):
        npt.assert_allclose(loaded_mcf.persistence[dim], results["persistence"][dim])
    npt.assert_allclose(loaded_mcf.betti_0_rank_, results["betti_0"])
    npt.assert_allclose(loaded_mcf.conflict_0_, results["conflict_0"])
    npt.assert_allclose(loaded_mcf.conflict_total_diff_, results["conflict_total_diff"])
