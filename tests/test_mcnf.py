"""Tests for MCNF"""

import numpy as np
import pytest


from mcf import MCNF

# data for toy example
partitions = [[0, 1, 2], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]]
filtration_indices = [1, 2, 3, 4, 5]


def test_compute_partition_size():
    """Test for computing partition sizes."""

    # initialise MCNF object
    mcnf = MCNF()
    mcnf.load_data(partitions, filtration_indices)

    # compute size of partitions
    s_partitions = mcnf.compute_partition_size()

    # check if sizes match
    assert np.array_equal(s_partitions, np.array([3, 2, 2, 2, 1]))


def test_compute_persistence():
    """Test for computing MCNF persistent homology."""

    # initialise MCNF object
    mcnf = MCNF()
    mcnf.load_data(partitions, filtration_indices)

    # build filtration
    mcnf.build_filtration()

    # compute persistent homology
    mcnf.compute_persistence()

    # check if persistence pairs match for 0-dim
    assert len(mcnf.persistence[0]) == 3
    assert np.array_equal(mcnf.persistence[0][0], np.array([1.0, 2.0]))
    assert np.array_equal(mcnf.persistence[0][1], np.array([1.0, 3.0]))
    assert np.array_equal(mcnf.persistence[0][2], np.array([1.0, np.inf]))

    # check if persistence pairs match for 1-dim
    assert len(mcnf.persistence[1]) == 1
    assert np.array_equal(mcnf.persistence[1][0], np.array([4.0, 5.0]))

    # check if persistence pairs match for 2-dim
    assert len(mcnf.persistence[2]) == 0


def test_compute_bettis():
    """Test for computing MCNF Betti curves."""

    # initialise MCNF object
    mcnf = MCNF()
    mcnf.load_data(partitions, filtration_indices)

    # build filtration
    mcnf.build_filtration()

    # compute persistent homology
    mcnf.compute_persistence()

    # compute Betti curves
    betti_0, betti_1, betti_2 = mcnf.compute_bettis()

    # check if Bett curves match
    assert np.array_equal(betti_0, np.array([3, 2, 1, 1, 1]))
    assert np.array_equal(betti_1, np.array([0, 0, 0, 1, 0]))
    assert np.array_equal(betti_2, np.array([0, 0, 0, 0, 0]))


def test_compute_persistent_hierarchy():
    """Test for computing MCNF persistent hierarchy."""

    # initialise MCNF object
    mcnf = MCNF()
    mcnf.load_data(partitions, filtration_indices)

    # build filtration
    mcnf.build_filtration()

    # compute persistent homology
    mcnf.compute_persistence()

    # compute persistent hierarchy
    h, h_bar = mcnf.compute_persistent_hierarchy()

    # check if persistent hierachy matches
    assert h_bar == 0.75
    assert np.array_equal(h, np.array([1.0, 1.0, 0.5, 0.5, 1.0]))


def test_compute_persistent_conflict():
    """Test for computing MCNF persistent conflict."""

    # initialise MCNF object
    mcnf = MCNF()
    mcnf.load_data(partitions, filtration_indices)

    # build filtration
    mcnf.build_filtration()

    # compute persistent homology
    mcnf.compute_persistence()

    # compute persistent conflict
    c_1, c_2, c = mcnf.compute_persistent_conflict()

    # check if persistent conflict matches
    assert np.array_equal(c_1, np.array([0.0, 0.0, 0.0, 1.0, -1.0]))
    assert np.array_equal(c_2, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(c, np.array([0.0, 0.0, 0.0, 1.0, -1.0]))


def test_compute_all_measures():
    """Test for computing all MCF measures."""

    # initialise MCNF object
    mcnf = MCNF()
    mcnf.load_data(partitions, filtration_indices)

    # compute all MCNF measures
    mcnf_results = mcnf.compute_all_measures()

    # check if results match
    assert np.array_equal(mcnf_results["filtration_indices"], filtration_indices)
    assert mcnf_results["max_dim"] == 3
    assert np.array_equal(
        mcnf_results["persistence"][0],
        np.array([[1.0, 2.0], [1.0, 3.0], [1.0, np.inf]]),
    )
    assert np.array_equal(mcnf_results["persistence"][1], np.array([[4.0, 5.0]]))
    assert len(mcnf_results["persistence"][2]) == 0
    assert np.array_equal(mcnf_results["betti_0"], np.array([3, 2, 1, 1, 1]))
    assert np.array_equal(mcnf_results["betti_1"], np.array([0, 0, 0, 1, 0]))
    assert np.array_equal(mcnf_results["betti_2"], np.array([0, 0, 0, 0, 0]))
    assert np.array_equal(mcnf_results["s_partitions"], np.array([3, 2, 2, 2, 1]))
    assert np.array_equal(mcnf_results["h"], np.array([1.0, 1.0, 0.5, 0.5, 1.0]))
    assert mcnf_results["h_bar"] == 0.75
    assert np.array_equal(mcnf_results["c_1"], np.array([0.0, 0.0, 0.0, 1.0, -1.0]))
    assert np.array_equal(mcnf_results["c_2"], np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(mcnf_results["c"], np.array([0.0, 0.0, 0.0, 1.0, -1.0]))