"""Stage-1 predicted risk for every node, computed out-of-fold (D-001, D-018).

Neighbour risk can't use labels: a test node's neighbours are all test nodes, whose
labels are unknown at prediction time. So stage 1, a tabular model, predicts risk for
every node, and the graph features average those predictions over neighbours.

For that to be leak-free, no node's risk may come from a model that trained on its time
step. Edges never cross time steps, so a node and its whole neighbourhood share a step;
holding out whole steps keeps every neighbour out of training too.
- Training-period nodes: scored by a model trained on the other folds of steps.
- Every other node (validation, test): scored by one model trained on all training steps,
  the same model the protocol keeps (D-009).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold

from fraudgraph.data.load import UNKNOWN, EllipticData
from fraudgraph.data.split import SplitConfig
from fraudgraph.features.tabular import feature_matrix
from fraudgraph.models.baselines import make_model


@dataclass(frozen=True)
class Stage1Result:
    risk: np.ndarray  # predicted illicit probability, one per node, aligned with nodes
    node_model: np.ndarray  # index into model_steps of the model that scored each node
    model_steps: list[frozenset[int]]  # time steps each model was trained on


def oof_risk(
    X: np.ndarray,
    y: np.ndarray,
    steps: np.ndarray,
    train_steps: tuple[int, int],
    n_folds: int,
    make_model: Callable[[np.ndarray], object],
) -> Stage1Result:
    """Score every node with a model that never trained on that node's time step.

    make_model(y_train) returns an unfitted classifier with fit / predict_proba; it gets
    the training labels because XGBoost's scale_pos_weight depends on them.
    """
    in_train = (steps >= train_steps[0]) & (steps <= train_steps[1])
    labelled = y != UNKNOWN
    train_step_ids = np.unique(steps[in_train])
    if n_folds > len(train_step_ids):
        raise ValueError(f"{n_folds} folds but only {len(train_step_ids)} training steps")

    risk = np.full(len(y), np.nan)
    node_model = np.full(len(y), -1)
    model_steps: list[frozenset[int]] = []

    def fit_and_score(fit_steps: np.ndarray, score_mask: np.ndarray) -> None:
        fit_mask = labelled & np.isin(steps, fit_steps)  # unknowns are never trained on
        model = make_model(y[fit_mask])
        model.fit(X[fit_mask], y[fit_mask])
        risk[score_mask] = model.predict_proba(X[score_mask])[:, 1]
        node_model[score_mask] = len(model_steps)
        model_steps.append(frozenset(int(s) for s in np.unique(steps[fit_mask])))

    # Folds are formed over the training steps themselves, one group per step, so a
    # step is never split between training and held-out.
    folds = GroupKFold(n_splits=n_folds).split(train_step_ids, groups=train_step_ids)
    for fit_idx, held_idx in folds:
        held_out = train_step_ids[held_idx]
        fit_and_score(train_step_ids[fit_idx], np.isin(steps, held_out))

    # Validation and test: the model trained on the whole training period.
    fit_and_score(train_step_ids, ~in_train)
    return Stage1Result(risk=risk, node_model=node_model, model_steps=model_steps)


def fit_stage1(
    data: EllipticData,
    split_cfg: SplitConfig,
    base: str,
    params: dict,
    seed: int,
    n_folds: int,
) -> Stage1Result:
    """Stage 1 with XGBoost on the given base feature set ('local' or 'all', D-018)."""
    X = feature_matrix(data, base)
    y = data.nodes["label"].to_numpy().astype(np.int64)
    steps = data.nodes["time_step"].to_numpy()
    return oof_risk(
        X,
        y,
        steps,
        split_cfg.train_steps,
        n_folds,
        lambda y_train: make_model("xgboost", params, seed, y_train),
    )
