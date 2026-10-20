"""Metric tests on tiny hand-worked examples. Every expected number is derived in the
comments, so a wrong label direction or an off-by-one in the ranking fails loudly."""

import numpy as np
import pytest

from fraudgraph.evaluation.metrics import (
    bootstrap_threshold,
    cost_threshold,
    f1_threshold,
    illicit_prf,
    per_step_metrics,
    pr_auc,
    precision_recall_at_k,
)

# Five transactions ranked by score: illicit, licit, illicit, licit, licit.
Y = np.array([1, 0, 1, 0, 0])
S = np.array([0.9, 0.8, 0.7, 0.2, 0.1])


def test_pr_auc_is_average_precision():
    # Hits at ranks 1 and 3: precision there is 1/1 and 2/3; AP is their mean.
    assert pr_auc(Y, S) == pytest.approx((1 + 2 / 3) / 2)


def test_pr_auc_rewards_illicit_not_licit():
    # Ranking the licit ones on top must score badly: guards against inverted labels.
    assert pr_auc(Y, -S) < 0.5


def test_prf_at_threshold():
    # score >= 0.75 flags the first two: 1 true positive, 1 false positive, 1 missed.
    m = illicit_prf(Y, S, 0.75)
    assert m == pytest.approx({"precision": 0.5, "recall": 0.5, "f1": 0.5})


def test_prf_with_nothing_flagged_is_zero_not_nan():
    assert illicit_prf(Y, S, 2.0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_f1_threshold():
    # Thresholds 0.9 / 0.8 / 0.7 / 0.2 / 0.1 give F1 0.667 / 0.5 / 0.8 / 0.667 / 0.571.
    t, f1 = f1_threshold(Y, S)
    assert t == 0.7
    assert f1 == pytest.approx(0.8)


def test_cost_threshold_misses_expensive():
    # FN costs 10, FP costs 1. Flag nothing: 2 misses = 20. Threshold 0.9: 10.
    # 0.8: 11. 0.7: 1 (one false alarm, no misses). 0.2: 2. 0.1: 3.
    assert cost_threshold(Y, S, fn_cost=10, fp_cost=1) == 0.7


def test_cost_threshold_false_alarms_expensive():
    # Flip the costs (FN 1, FP 10): nothing = 2, 0.9 = 1, 0.8 = 11, 0.7 = 10.
    # The cost weights must change the answer, in this direction.
    assert cost_threshold(Y, S, fn_cost=1, fp_cost=10) == 0.9


def test_cost_threshold_can_flag_nothing():
    # One illicit buried under licit ones and false alarms costing a lot: flag nothing.
    y, s = np.array([0, 0, 1]), np.array([0.9, 0.8, 0.1])
    assert cost_threshold(y, s, fn_cost=1, fp_cost=10) == np.inf


def test_precision_recall_at_k():
    # Top 2 are illicit, licit: 1 hit of 2 reviewed, 1 of the 2 illicit found.
    assert precision_recall_at_k(Y, S, 2) == pytest.approx((0.5, 0.5))


def test_k_larger_than_step_reviews_everything():
    # Only 5 transactions: the investigator reviews 5, not 10, and finds both illicit.
    assert precision_recall_at_k(Y, S, 10) == pytest.approx((0.4, 1.0))


def test_ties_at_cutoff_use_expected_value():
    # One clear hit, then three tied at 0.5 (one illicit) competing for one slot.
    # Expected hits = 1 + 1/3, so precision@2 = (4/3) / 2 and recall@2 = (4/3) / 2.
    y, s = np.array([1, 0, 0, 1]), np.array([0.9, 0.5, 0.5, 0.5])
    p, r = precision_recall_at_k(y, s, 2)
    assert p == pytest.approx(2 / 3) and r == pytest.approx(2 / 3)
    # Row order must not matter (it would with a plain sort).
    perm = np.array([3, 1, 2, 0])
    assert precision_recall_at_k(y[perm], s[perm], 2) == pytest.approx((p, r))


def test_recall_at_k_undefined_without_illicit():
    p, r = precision_recall_at_k(np.array([0, 0]), np.array([0.9, 0.1]), 1)
    assert p == 0.0 and np.isnan(r)


def test_per_step_metrics():
    steps = np.array([1, 1, 1, 2, 2])
    y = np.array([1, 0, 0, 0, 0])
    s = np.array([0.9, 0.1, 0.2, 0.8, 0.3])
    df = per_step_metrics(y, s, steps, threshold=0.5, ks=[1])
    assert list(df.index) == [1, 2]
    assert df.loc[1, "n_labelled"] == 3 and df.loc[1, "n_illicit"] == 1
    assert df.loc[1, "precision_at_1"] == 1.0 and df.loc[1, "f1"] == 1.0
    # Step 2 has no illicit: PR-AUC and recall are undefined, the false alarm still counts.
    assert df.loc[2, "n_illicit"] == 0
    assert np.isnan(df.loc[2, "pr_auc"]) and np.isnan(df.loc[2, "recall_at_1"])
    assert df.loc[2, "precision_at_1"] == 0.0


def test_bootstrap_is_reproducible_and_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.1
    s = y * 0.3 + rng.random(2000)  # informative but noisy
    a = bootstrap_threshold(y.astype(int), s, n_boot=200, seed=1)
    b = bootstrap_threshold(y.astype(int), s, n_boot=200, seed=1)
    assert a == b
    _, f1 = f1_threshold(y.astype(int), s)
    assert a["f1_lo"] <= f1 <= a["f1_hi"]
    assert a["threshold_lo"] <= a["threshold_hi"]
