"""Tests for nerve-based MCF"""

import numpy as np
import numpy.testing as npt

from mcf import MCF

# data for toy example
partitions = [[0, 1, 2], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]]
filtration_indices = [1, 2, 3, 4, 5]


def test_compute_partition_size():
    """Test for computing partition sizes."""

    # initialise nerve-based MCF
    n_mcf = MCF(method="nerve")
    n_mcf.load_data(partitions, filtration_indices)

    # compute size of partitions
    s_partitions = n_mcf.compute_partition_size()

    # check if sizes match
    npt.assert_allclose(s_partitions, np.array([3, 2, 2, 2, 1]))


def test_compute_persistence():
    """Test for computing nerve-based MCF persistent homology."""

    # initialise nerve-based MCF
    n_mcf = MCF(method="nerve")
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
    """Test for computing nerve-based MCF Betti curves."""

    # initialise nerve-based MCF
    n_mcf = MCF(method="nerve")
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


def test_compute_persistent_hierarchy():
    """Test for computing nerve-based MCF persistent hierarchy."""

    # initialise nerve-based MCF
    n_mcf = MCF(method="nerve")
    n_mcf.load_data(partitions, filtration_indices)

    # build filtration
    n_mcf.build_filtration()

    # compute persistent homology
    n_mcf.compute_persistence()

    # compute persistent hierarchy
    h, h_bar = n_mcf.compute_persistent_hierarchy()

    # check if persistent hierachy matches
    assert h_bar == 0.75
    npt.assert_allclose(h, np.array([1.0, 1.0, 0.5, 0.5, 1.0]))


def test_compute_persistent_conflict():
    """Test for computing nerve-based MCF persistent conflict."""

    # initialise nerve-based MCF
    n_mcf = MCF(method="nerve")
    n_mcf.load_data(partitions, filtration_indices)

    # build filtration
    n_mcf.build_filtration()

    # compute persistent homology
    n_mcf.compute_persistence()

    # compute persistent conflict
    c_1, c_2, c = n_mcf.compute_persistent_conflict()

    # check if persistent conflict matches
    npt.assert_allclose(c_1, np.array([0.0, 0.0, 0.0, 1.0, -1.0]))
    npt.assert_allclose(c_2, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    npt.assert_allclose(c, np.array([0.0, 0.0, 0.0, 1.0, -1.0]))


def test_compute_all_measures():
    """Test for computing all nerve-based MCF measures."""

    # initialise nerve-based MCF
    n_mcf = MCF(method="nerve")
    n_mcf.load_data(partitions, filtration_indices)

    # compute all nerve-based MCF measures
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
    npt.assert_allclose(n_mcf_results["h"], np.array([1.0, 1.0, 0.5, 0.5, 1.0]))
    assert n_mcf_results["h_bar"] == 0.75
    npt.assert_allclose(n_mcf_results["c_1"], np.array([0.0, 0.0, 0.0, 1.0, -1.0]))
    npt.assert_allclose(n_mcf_results["c_2"], np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    npt.assert_allclose(n_mcf_results["c"], np.array([0.0, 0.0, 0.0, 1.0, -1.0]))
