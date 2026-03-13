"""Tests for standard MCF"""

import numpy as np
import numpy.testing as npt

from mcf import MultiscaleClusteringFiltration as MCF

# data for toy example
partitions = [[0, 1, 2], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]]
filtration_indices = [1, 2, 3, 4, 5]


def test_compute_partition_size():
    """Test for computing partition sizes."""

    # initialise standard MCF
    n_mcf = MCF(method="standard")
    n_mcf.load_data(partitions, filtration_indices)

    # compute size of partitions
    n_mcf.compute_partition_size()
    s_partitions = n_mcf.s_partitions_

    # check if sizes match
    npt.assert_allclose(s_partitions, np.array([3, 2, 2, 2, 1]))


def test_compute_persistence():
    """Test for computing standard MCF persistent homology."""

    # initialise standard MCF
    n_mcf = MCF(method="standard")
    n_mcf.load_data(partitions, filtration_indices)

    # build filtration
    n_mcf.build_filtration()

    # compute persistent homology
    n_mcf.compute_persistence()

    # check if persistence pairs match for 0-dim
    assert len(n_mcf.persistence[0]) == 3
    npt.assert_allclose(n_mcf.persistence[0][0], np.array([1.0, 2.0]))
    npt.assert_allclose(n_mcf.persistence[0][1], np.array([1.0, 3.0]))
    npt.assert_allclose(n_mcf.persistence[0][2], np.array([1.0, np.inf]))

    # check if persistence pairs match for 1-dim
    assert len(n_mcf.persistence[1]) == 1
    npt.assert_allclose(n_mcf.persistence[1][0], np.array([4.0, 5.0]))

    # check if persistence pairs match for 2-dim
    assert len(n_mcf.persistence[2]) == 0


def test_compute_bettis():
    """Test for computing standard MCF Betti curves."""

    # initialise standard MCF
    n_mcf = MCF(method="standard")
    n_mcf.load_data(partitions, filtration_indices)

    # build filtration
    n_mcf.build_filtration()

    # compute persistent homology
    n_mcf.compute_persistence()

    # compute Betti curves
    betti_0, betti_1, betti_2 = n_mcf.compute_bettis()

    # check if Bett curves match
    npt.assert_allclose(betti_0, np.array([3, 2, 1, 1, 1]))
    npt.assert_allclose(betti_1, np.array([0, 0, 0, 1, 0]))
    npt.assert_allclose(betti_2, np.array([0, 0, 0, 0, 0]))


def test_compute_0_conflict_measures():
    """Test for computing standard MCF persistent hierarchy."""

    # initialise standard MCF
    n_mcf = MCF(method="standard")
    n_mcf.load_data(partitions, filtration_indices)

    # build filtration
    n_mcf.build_filtration()

    # compute persistent homology
    n_mcf.compute_persistence()

    # compute persistent hierarchy
    n_mcf.compute_0_conflict()
    h = n_mcf.conflict_0_
    h_bar = n_mcf.conflict_0_avg_

    # check if persistent hierachy matches
    assert h_bar == 0.25
    npt.assert_allclose(h, np.array([0.0, 0.0, 0.5, 0.5, 0.0]))


def test_compute_k_conflict_measures():
    """Test for computing standard MCF persistent conflict."""

    # initialise standard MCF
    n_mcf = MCF(method="standard")
    n_mcf.load_data(partitions, filtration_indices)

    # build filtration
    n_mcf.build_filtration()

    # compute persistent homology
    n_mcf.compute_persistence()

    # compute persistent conflict
    n_mcf.compute_k_conflict_difference()
    c_1 = n_mcf.conflict_1_diff_
    c_2 = n_mcf.conflict_2_diff_
    c = n_mcf.conflict_total_diff_

    # check if persistent conflict matches
    npt.assert_allclose(c_1, np.array([0.0, 0.0, 0.0, 1.0, -1.0]))
    npt.assert_allclose(c_2, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    npt.assert_allclose(c, np.array([0.0, 0.0, 0.0, 1.0, -1.0]))


def test_compute_all_measures():
    """Test for computing all standard MCF measures."""

    # initialise standard MCF
    n_mcf = MCF(method="standard")
    n_mcf.load_data(partitions, filtration_indices)

    # compute all standard MCF measures
    n_mcf_results = n_mcf.compute_all_measures()

    # check if results match
    npt.assert_allclose(n_mcf_results["filtration_indices"], filtration_indices)
    assert n_mcf_results["max_dim"] == 3
    npt.assert_allclose(
        n_mcf_results["persistence"][0],
        np.array([[1.0, 2.0], [1.0, 3.0], [1.0, np.inf]]),
    )
    npt.assert_allclose(n_mcf_results["persistence"][1], np.array([[4.0, 5.0]]))
    assert len(n_mcf_results["persistence"][2]) == 0
    npt.assert_allclose(n_mcf_results["betti_0"], np.array([3, 2, 1, 1, 1]))
    npt.assert_allclose(n_mcf_results["betti_1"], np.array([0, 0, 0, 1, 0]))
    npt.assert_allclose(n_mcf_results["betti_2"], np.array([0, 0, 0, 0, 0]))
    npt.assert_allclose(n_mcf_results["s_partitions"], np.array([3, 2, 2, 2, 1]))
    npt.assert_allclose(n_mcf_results["conflict_0"], np.array([0.0, 0.0, 0.5, 0.5, 0.0]))
    assert n_mcf_results["conflict_0_avg"] == 0.25
    npt.assert_allclose(n_mcf_results["conflict_1_diff"], np.array([0.0, 0.0, 0.0, 1.0, -1.0]))
    npt.assert_allclose(n_mcf_results["conflict_2_diff"], np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    npt.assert_allclose(n_mcf_results["conflict_total_diff"], np.array([0.0, 0.0, 0.0, 1.0, -1.0]))
