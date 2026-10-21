"""Tabular baselines: logistic regression, random forest, XGBoost.

Each handles the ~1:9 illicit:licit imbalance among labelled training nodes by
reweighting, not resampling, so no training rows are thrown away or duplicated:
- logistic regression and random forest: class_weight="balanced", which weights each
  class by n_samples / (2 * n_in_class)
- XGBoost: scale_pos_weight = n_licit / n_illicit, its equivalent for the positive class
"""

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraudgraph.data.load import EllipticData
from fraudgraph.data.split import (
    SplitConfig,
    TemporalSplit,
    assert_no_cross_split_edges,
    temporal_split,
)
from fraudgraph.evaluation.protocol import EvalConfig, FitFn, ProtocolResult, run_protocol
from fraudgraph.features.tabular import feature_matrix

MODELS = ("logreg", "random_forest", "xgboost")


def make_model(name: str, params: dict[str, Any], seed: int, y_train: np.ndarray):
    if name == "logreg":
        # Scaling sits inside the pipeline, so its means and variances are fitted on the
        # training rows only, never on validation or test.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(**params, class_weight="balanced", max_iter=5000),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            **params, class_weight="balanced", random_state=seed, n_jobs=-1
        )
    if name == "xgboost":
        n_pos = int(y_train.sum())
        return XGBClassifier(
            **params,
            scale_pos_weight=(len(y_train) - n_pos) / n_pos,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model {name!r}; expected one of {MODELS}")


def fit_model(
    name: str, params: dict[str, Any], seed: int, X: np.ndarray, y: np.ndarray, mask: np.ndarray
):
    """Fit one model on the rows in mask. Phase 5 uses the fitted model itself (SHAP)."""
    model = make_model(name, params, seed, y[mask])
    return model.fit(X[mask], y[mask])


def tabular_fit_fn(name: str, X: np.ndarray, y: np.ndarray, split: TemporalSplit) -> FitFn:
    """Adapt a model to the protocol: train on the train mask, score any period's mask."""

    def fit(params, seed):
        model = fit_model(name, params, seed, X, y, split.train)

        def score(period: str) -> np.ndarray:
            return model.predict_proba(X[getattr(split, period)])[:, 1]

        return score

    return fit


def run_baselines(
    data: EllipticData,
    split_cfg: SplitConfig,
    cfg: dict[str, Any],
    run_group: str,
    models: list[str] | None = None,
    feature_sets: list[str] | None = None,
) -> dict[tuple[str, str], ProtocolResult]:
    """Run every (feature set, model) pair through the protocol. MLflow must be set up."""
    assert_no_cross_split_edges(data.nodes, data.edges, split_cfg)
    split = temporal_split(data.nodes, split_cfg)
    y = data.nodes["label"].to_numpy().astype(np.int64)  # -1 for unknown; masks exclude them
    steps = data.nodes["time_step"].to_numpy()
    eval_cfg = EvalConfig(**{**cfg["evaluation"], "extra_ks": tuple(cfg["evaluation"]["extra_ks"])})

    results = {}
    for fs in feature_sets or cfg["feature_sets"]:
        X = feature_matrix(data, fs)
        for name in models or list(cfg["models"]):
            mcfg = cfg["models"][name]
            seeds = cfg["seeds"][:1] if mcfg.get("deterministic") else cfg["seeds"]
            tags = {
                "phase": str(cfg.get("phase", "2")),
                "model": name,
                "feature_set": fs,
                "run_group": run_group,
            }
            results[(name, fs)] = run_protocol(
                tabular_fit_fn(name, X, y, split),
                mcfg["grid"],
                seeds,
                y,
                steps,
                split,
                eval_cfg,
                tags,
                ids=data.nodes["tx_id"].to_numpy(),
            )
    return results
