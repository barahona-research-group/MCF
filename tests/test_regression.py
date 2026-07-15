"""Regression tests against frozen MCF outputs.

The expected values in tests/data/regression_data.json were computed with a
verified implementation (standard and nerve-based construction agree, toy
example matches the paper). If one of these tests fails after a code change,
either the change introduced a bug or the outputs changed intentionally; in
the latter case regenerate the data with
`python tests/generate_regression_data.py` and review the diff.
"""

import json
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mcf import MultiscaleClusteringFiltration as MCF

DATA_PATH = Path(__file__).parent / "data" / "regression_data.json"

with open(DATA_PATH) as data_file:
    REGRESSION_DATA = json.load(data_file)

RESULT_KEYS = [
    "betti_0",
    "betti_1",
    "betti_2",
    "s_partitions",
    "conflict_0",
    "conflict_0_avg",
    "conflict_1_avg",
    "conflict_2_avg",
    "conflict_1_diff",
    "conflict_2_diff",
    "conflict_total_diff",
]


@pytest.mark.parametrize("method", ["standard", "nerve"])
@pytest.mark.parametrize("name", REGRESSION_DATA)
def test_regression(name, method):
    """MCF outputs match the frozen expected values."""
    case = REGRESSION_DATA[name]

    mcf = MCF(method=method, max_dim=case["max_dim"])
    mcf.load_data(case["partitions"], case["filtration_indices"])
    results = mcf.compute_all_measures(file_path=None, tqdm_disable=True)

    assert mcf.n_simplices == case["n_simplices"][method]

    for dim in range(case["max_dim"]):
        computed = np.array(sorted(map(tuple, results["persistence"][dim])))
        expected = np.array(case["persistence"][dim])
        npt.assert_allclose(
            computed.reshape(-1, 2),
            expected.reshape(-1, 2),
            err_msg=f"persistence differs in dim {dim}",
        )

    for key in RESULT_KEYS:
        npt.assert_allclose(results[key], case[key], err_msg=key)
