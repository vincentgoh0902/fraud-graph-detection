"""Expanding-window retraining simulation (D-035)."""

import numpy as np
import pytest

from fraudgraph.analysis.retrain import retrain_step, rolling_windows


def test_first_window_is_the_main_protocol_split():
    # Scoring step 35 with a 5-step validation window = train 1-29, validate 30-34.
    w = rolling_windows(test_steps=(35, 49), val_len=5)[0]
    assert w.train == (1, 29) and w.val == (30, 34) and w.step == 35


def test_windows_move_forward_and_never_touch_the_scored_step():
    windows = rolling_windows(test_steps=(35, 49), val_len=5)
    assert [w.step for w in windows] == list(range(35, 50))
    for w in windows:
        assert w.train[1] < w.val[0] and w.val[1] == w.step - 1
        assert w.val[1] - w.val[0] + 1 == 5


def test_retrain_step_trains_only_on_earlier_labelled_steps():
    rng = np.random.default_rng(0)
    steps = np.repeat(np.arange(1, 11), 50)
    y = (rng.uniform(size=len(steps)) < 0.3).astype(np.int64)
    y[rng.uniform(size=len(steps)) < 0.2] = -1  # unknown labels must be ignored
    X = (y[:, None] + rng.normal(size=(len(steps), 3))).astype(np.float32)
    # Poison the scored step: if it leaked into training, the scores would change.
    w = rolling_windows(test_steps=(10, 10), val_len=2)[0]
    params = {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1}
    s1, t1 = retrain_step(X, y, steps, w, params, seed=0)
    y_poisoned = y.copy()
    flip = (steps == 10) & (y != -1)
    y_poisoned[flip] = 1 - y[flip]
    s2, t2 = retrain_step(X, y_poisoned, steps, w, params, seed=0)
    np.testing.assert_array_equal(s1, s2)
    assert t1 == pytest.approx(t2)
    assert len(s1) == ((steps == 10) & (y != -1)).sum()  # labelled rows of step t only
