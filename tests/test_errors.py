"""False-negative analysis helpers (D-033, item 7)."""

import numpy as np
import pandas as pd
import pytest

from fraudgraph.analysis.errors import missed_illicit, seed_agreement, standardised_difference


def test_missed_illicit_uses_the_given_threshold_only():
    y = np.array([1, 1, 1, 0, 0])
    score = np.array([0.9, 0.5, 0.1, 0.95, 0.0])
    # Flagged means score >= threshold, as in evaluation.metrics.
    np.testing.assert_array_equal(missed_illicit(y, score, 0.5), [False, False, True, False, False])


def test_standardised_difference_known_values():
    a = pd.DataFrame({"x": [2.0, 4.0], "y": [1.0, 1.0]})
    b = pd.DataFrame({"x": [0.0, 2.0], "y": [1.0, 1.0]})
    d = standardised_difference(a, b, ["x", "y"])
    # means 3 vs 1; both sample sds sqrt(2); pooled sd sqrt(2) -> 2 / sqrt(2)
    assert d["x"] == pytest.approx(2 / np.sqrt(2))
    assert np.isnan(d["y"])  # no spread: undefined, not infinite
    assert d.index[0] == "x"  # sorted by size of the difference


def test_seed_agreement_counts_illicit_by_number_of_seeds_missing_them():
    masks = [
        np.array([True, True, False, False]),
        np.array([True, False, False, False]),
        np.array([True, True, True, False]),
    ]
    out = seed_agreement(masks)
    assert out.to_dict() == {0: 1, 1: 1, 2: 1, 3: 1}
