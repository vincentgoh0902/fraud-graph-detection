"""Paired bootstrap over test steps for the change between two feature sets (D-024)."""

import numpy as np
import pytest

from fraudgraph.evaluation.compare import paired_step_bootstrap


@pytest.fixture
def test_set():
    rng = np.random.default_rng(0)
    steps = np.repeat(np.arange(35, 50), 40)
    y = (rng.uniform(size=len(steps)) < 0.2).astype(int)
    noise = rng.normal(size=len(steps))
    return y, steps, noise


def test_identical_scores_give_zero_change(test_set):
    y, steps, noise = test_set
    scores = [y + noise, y + 0.5 * noise]
    r = paired_step_bootstrap(y, steps, scores, scores, k=5, n_boot=200, seed=0)
    for metric in ("pr_auc", "precision_at_k"):
        assert r[f"{metric}_change"] == pytest.approx(0)
        assert r[f"{metric}_lo"] == pytest.approx(0) and r[f"{metric}_hi"] == pytest.approx(0)


def test_clearly_better_scores_give_an_interval_above_zero(test_set):
    y, steps, noise = test_set
    worse, better = [y + 2 * noise], [y + 0.3 * noise]
    r = paired_step_bootstrap(y, steps, worse, better, k=5, n_boot=200, seed=0)
    assert r["pr_auc_change"] > 0
    assert r["pr_auc_lo"] > 0
    assert r["share_positive"] == pytest.approx(1.0)


def test_change_is_difference_of_seed_means(test_set):
    from fraudgraph.evaluation.metrics import pr_auc

    y, steps, noise = test_set
    a = [y + noise, y + 1.5 * noise]
    b = [y + 0.5 * noise, y + 0.7 * noise]
    r = paired_step_bootstrap(y, steps, a, b, k=5, n_boot=50, seed=0)
    expected = np.mean([pr_auc(y, s) for s in b]) - np.mean([pr_auc(y, s) for s in a])
    assert r["pr_auc_change"] == pytest.approx(expected)


def test_resamples_whole_steps():
    # One step only: every resample is that same step, so the interval has zero width.
    # Resampling individual transactions would vary.
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=200) < 0.3).astype(int)
    steps = np.full(200, 40)
    a, b = [y + rng.normal(size=200)], [y + rng.normal(size=200)]
    r = paired_step_bootstrap(y, steps, a, b, k=5, n_boot=100, seed=0)
    assert r["pr_auc_lo"] == pytest.approx(r["pr_auc_hi"])


def test_misaligned_scores_rejected(test_set):
    y, steps, noise = test_set
    with pytest.raises(ValueError):
        paired_step_bootstrap(y, steps, [noise], [noise[:-1]], k=5, n_boot=10, seed=0)
