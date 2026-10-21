"""Distribution-shift tools for Phase 5 (D-033): PSI, adversarial validation, period metrics."""

import numpy as np
import pandas as pd
import pytest

from fraudgraph.analysis.drift import (
    adversarial_validation,
    feature_psi,
    grouped_folds,
    period_metrics,
    psi,
    time_trend,
)


def test_psi_is_zero_for_the_same_distribution():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) == pytest.approx(0, abs=1e-9)


def test_psi_is_small_for_a_fresh_sample_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20000)
    assert psi(ref, rng.normal(size=20000)) < 0.01
    assert psi(ref, rng.normal(loc=1.0, size=20000)) > 0.25  # the usual "large shift" line


def test_psi_bins_come_from_the_reference_only():
    # Values far outside the reference range land in the outer bins; they still count.
    ref = np.linspace(0, 1, 1000)
    cur = np.full(1000, 100.0)
    assert psi(ref, cur) > 1


def test_psi_handles_a_reference_with_few_distinct_values():
    # Elliptic has many near-constant columns: quantile edges collapse, but PSI stays finite.
    ref = np.r_[np.zeros(990), np.ones(10)]
    cur = np.r_[np.zeros(500), np.ones(500)]
    value = psi(ref, cur)
    assert np.isfinite(value) and value > 0.25


def test_feature_psi_sorted_descending():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000)})
    cur = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(loc=2, size=2000)})
    out = feature_psi(ref, cur, ["a", "b"])
    assert list(out.index) == ["b", "a"]


def test_grouped_folds_never_split_a_step_and_have_both_classes():
    steps = np.repeat(np.arange(1, 37), 20)
    target = (steps > 29).astype(int)
    folds = grouped_folds(target, steps, n_folds=5, seed=0)
    assert len(folds) == 5
    seen = np.zeros(len(steps), dtype=int)
    for train_idx, test_idx in folds:
        assert not set(steps[train_idx]) & set(steps[test_idx])
        assert set(target[test_idx]) == {0, 1}
        seen[test_idx] += 1
    assert (seen == 1).all()  # every row scored out-of-fold exactly once


PARAMS = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}


def _two_periods(shift: float):
    rng = np.random.default_rng(0)
    steps = np.repeat(np.arange(1, 37), 60)
    target = (steps > 29).astype(int)
    X = rng.normal(size=(len(steps), 4)).astype(np.float32)
    X[:, 2] += shift * target  # only feature 2 differs between the periods
    return X, target, steps


def test_adversarial_auc_near_half_without_shift():
    X, target, steps = _two_periods(shift=0.0)
    r = adversarial_validation(X, target, steps, PARAMS, n_folds=5, seed=0)
    assert abs(r["auc"] - 0.5) < 0.1


def test_adversarial_auc_high_and_points_at_the_shifted_feature():
    X, target, steps = _two_periods(shift=3.0)
    r = adversarial_validation(X, target, steps, PARAMS, n_folds=5, seed=0)
    assert r["auc"] > 0.9
    assert int(np.argmax(r["importance"])) == 2


def test_period_metrics_rows_and_oracle():
    steps = np.repeat([35, 36, 43, 44], 10)
    y = np.tile([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 4)
    # Before: illicit score high. After: illicit score low but still ranked first.
    score = np.where(y == 1, np.where(steps < 43, 0.9, 0.2), 0.1)
    periods = {"before": (35, 42), "after": (43, 49)}
    out = period_metrics(y, score, steps, threshold=0.5, k=2, periods=periods)
    assert list(out.index) == ["before", "after"]
    assert out.loc["before", "recall"] == 1.0
    assert out.loc["after", "recall"] == 0.0
    # Ranking is still perfect after, so recall@2 and the per-step oracle F1 are 1:
    # the threshold is the problem, not the ranking.
    assert out.loc["after", "recall_at_k"] == 1.0
    assert out.loc["after", "oracle_f1"] == 1.0
    assert out.loc["after", "n_illicit"] == 4
    assert out.loc["after", "median_illicit_score"] == pytest.approx(0.2)


def test_time_trend_finds_a_clock_feature():
    rng = np.random.default_rng(0)
    steps = np.repeat(np.arange(1, 30), 50)
    nodes = pd.DataFrame(
        {
            "clock": 0.06 * steps + rng.normal(scale=0.01, size=len(steps)),
            "noise": rng.normal(size=len(steps)),
        }
    )
    out = time_trend(nodes, ["clock", "noise"], steps)
    assert out.index[0] == "clock"
    assert out["clock"] == pytest.approx(1.0)
    assert abs(out["noise"]) < 0.5
