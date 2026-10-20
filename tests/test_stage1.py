"""Stage-1 out-of-fold risk (D-001): no node may be scored by a model that trained on
its own time step, and validation/test nodes are scored by the model trained on the
whole training period.

The fake model below records the time steps of the rows it was actually fitted on, so
these tests check the training data itself, not just the bookkeeping.
"""

import numpy as np
import pytest

from fraudgraph.data.load import UNKNOWN
from fraudgraph.features.stage1 import fit_stage1, oof_risk

TRAIN_STEPS = (1, 4)
N_STEPS, PER_STEP = 7, 5


class StepRecorder:
    """Fits nothing; remembers which steps (column 0 of X) and labels it saw."""

    def __init__(self):
        self.steps, self.labels = set(), set()

    def fit(self, X, y):
        self.steps = set(X[:, 0].astype(int))
        self.labels = set(y.tolist())
        return self

    def predict_proba(self, X):
        p = np.full(len(X), 0.5)
        return np.c_[1 - p, p]


@pytest.fixture
def toy():
    steps = np.repeat(np.arange(1, N_STEPS + 1), PER_STEP)
    # Per step: one illicit, two licit, two unknown.
    y = np.tile([1, 0, 0, UNKNOWN, UNKNOWN], N_STEPS)
    X = np.c_[steps, np.random.default_rng(0).normal(size=(len(steps), 3))]
    models = []

    def make_model(y_train):
        models.append(StepRecorder())
        return models[-1]

    res = oof_risk(X, y, steps, TRAIN_STEPS, n_folds=2, make_model=make_model)
    return steps, y, res, models


def test_every_node_is_scored(toy):
    _, _, res, _ = toy
    assert not np.isnan(res.risk).any()


def test_no_node_scored_by_a_model_trained_on_its_step(toy):
    steps, _, res, models = toy
    for i, step in enumerate(steps):
        assert step not in models[res.node_model[i]].steps
        assert step not in res.model_steps[res.node_model[i]]


def test_recorded_steps_match_what_the_model_saw(toy):
    _, _, res, models = toy
    for m, recorded in zip(models, res.model_steps, strict=True):
        assert m.steps == set(recorded)


def test_val_and_test_nodes_use_the_full_training_model(toy):
    steps, _, res, models = toy
    outside = (steps < TRAIN_STEPS[0]) | (steps > TRAIN_STEPS[1])
    full = set(range(TRAIN_STEPS[0], TRAIN_STEPS[1] + 1))
    assert outside.any()
    for i in np.flatnonzero(outside):
        assert models[res.node_model[i]].steps == full


def test_train_nodes_use_fold_models_that_saw_only_training_steps(toy):
    steps, _, res, models = toy
    inside = (steps >= TRAIN_STEPS[0]) & (steps <= TRAIN_STEPS[1])
    full = set(range(TRAIN_STEPS[0], TRAIN_STEPS[1] + 1))
    for i in np.flatnonzero(inside):
        seen = models[res.node_model[i]].steps
        assert seen < full  # a strict subset: some training steps were held out


def test_models_never_train_on_unknown_labels(toy):
    _, _, _, models = toy
    for m in models:
        assert m.labels <= {0, 1}


def test_unknown_nodes_are_scored_too(toy):
    _, y, res, _ = toy
    assert not np.isnan(res.risk[y == UNKNOWN]).any()


def test_more_folds_than_steps_rejected():
    steps = np.repeat([1, 2], 3)
    y = np.tile([1, 0, 0], 2)
    with pytest.raises(ValueError):
        oof_risk(np.c_[steps], y, steps, (1, 2), n_folds=3, make_model=lambda y: StepRecorder())


def test_fit_stage1_with_xgboost(data, split_cfg):
    params = {"n_estimators": 5, "max_depth": 2}
    res = fit_stage1(data, split_cfg, "local", params, seed=0, n_folds=3)
    assert res.risk.shape == (len(data.nodes),)
    assert ((res.risk >= 0) & (res.risk <= 1)).all()
    steps = data.nodes["time_step"].to_numpy()
    for i, step in enumerate(steps):
        assert step not in res.model_steps[res.node_model[i]]
