"""Regenerate the frozen regression data used by tests/test_regression.py.

Run `python tests/generate_regression_data.py` and review the diff of
tests/data/regression_data.json before committing. Only regenerate when an
intentional change to the MCF outputs has been made and verified.

The file stores the input partitions together with the expected outputs, so
the tests do not depend on the stability of numpy's random number stream.
Expected values are computed with the standard method after asserting that
the nerve-based method gives identical results.
"""

import json
from pathlib import Path

import numpy as np

from conftest import (
    TOY_FILTRATION_INDICES,
    TOY_PARTITIONS,
    build_mcf,
    hierarchical_sequence,
    quasi_hierarchical_sequence,
    random_sequence,
    sorted_diagrams,
)

DATA_PATH = Path(__file__).parent / "data" / "regression_data.json"

# selected cases: name -> (partitions, filtration_indices, max_dim)
CASES = {
    "toy_example": (TOY_PARTITIONS, TOY_FILTRATION_INDICES, 3),
    "random": (*random_sequence(60, 8, 20, 4, seed=0), 3),
    "quasi_hierarchical": (*quasi_hierarchical_sequence(60, 8, 5, seed=6), 3),
    "hierarchical": (*hierarchical_sequence(50, 8, seed=5), 3),
    "non_equidistant_indices": (
        random_sequence(50, 7, 18, 4, seed=4)[0],
        np.cumsum(np.random.default_rng(4).uniform(0.1, 2.0, 7)),
        3,
    ),
    "cycle_of_pairs": ([[0, 0, 1, 1], [0, 1, 1, 0]], [1, 2], 3),
}

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


def main():
    data = {}
    for name, (partitions, filtration_indices, max_dim) in CASES.items():
        standard = build_mcf("standard", partitions, filtration_indices, max_dim)
        nerve = build_mcf("nerve", partitions, filtration_indices, max_dim)

        # freeze only outputs on which both construction methods agree
        diagrams = sorted_diagrams(standard)
        diagrams_nerve = sorted_diagrams(nerve)
        for dim in range(max_dim):
            np.testing.assert_allclose(diagrams[dim], diagrams_nerve[dim])

        results = standard.compute_all_measures(file_path=None, tqdm_disable=True)

        data[name] = {
            "partitions": [np.asarray(p).tolist() for p in standard.partitions],
            "filtration_indices": np.asarray(standard.filtration_indices).tolist(),
            "max_dim": max_dim,
            "n_simplices": {
                "standard": standard.n_simplices,
                "nerve": nerve.n_simplices,
            },
            "persistence": [dgm.tolist() for dgm in diagrams],
            **{key: np.asarray(results[key]).tolist() for key in RESULT_KEYS},
        }
        print(f"stored {name}")

    DATA_PATH.parent.mkdir(exist_ok=True)
    with open(DATA_PATH, "w") as data_file:
        json.dump(data, data_file, indent=1)
    print(f"written to {DATA_PATH}")


if __name__ == "__main__":
    main()
