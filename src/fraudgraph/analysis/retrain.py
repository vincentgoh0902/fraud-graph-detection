"""Expanding-window retraining simulation (D-035): a "what if", not a result.

For each test step t the model is retrained as if labels up to t-1 had arrived: train on
steps 1..t-val_len-1, tune the threshold on the val_len steps before t, score t. With
t = 35 and val_len = 5 this is exactly the main protocol's split, which the analysis
script uses as a check.
"""

from dataclasses import dataclass

import numpy as np

from fraudgraph.data.load import UNKNOWN
from fraudgraph.evaluation.metrics import f1_threshold
from fraudgraph.models.baselines import fit_model

Steps = tuple[int, int]


@dataclass(frozen=True)
class Window:
    train: Steps
    val: Steps
    step: int


def rolling_windows(test_steps: Steps, val_len: int, first_step: int = 1) -> list[Window]:
    first, last = test_steps
    return [
        Window(train=(first_step, t - val_len - 1), val=(t - val_len, t - 1), step=t)
        for t in range(first, last + 1)
    ]


def _mask(steps: np.ndarray, labelled: np.ndarray, rng: Steps) -> np.ndarray:
    return labelled & (steps >= rng[0]) & (steps <= rng[1])


def retrain_step(
    X: np.ndarray, y: np.ndarray, steps: np.ndarray, w: Window, params: dict, seed: int
) -> tuple[np.ndarray, float]:
    """Scores for the labelled rows of step w.step, and the max-F1 threshold from w.val."""
    labelled = y != UNKNOWN
    model = fit_model("xgboost", params, seed, X, y, _mask(steps, labelled, w.train))
    val = _mask(steps, labelled, w.val)
    threshold, _ = f1_threshold(y[val], model.predict_proba(X[val])[:, 1])
    test = _mask(steps, labelled, (w.step, w.step))
    return model.predict_proba(X[test])[:, 1], threshold
