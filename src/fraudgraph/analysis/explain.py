"""SHAP values for XGBoost, from XGBoost's own TreeSHAP (D-034).

TreeSHAP is exact for tree models: it gives each feature its average contribution to the
prediction over every order in which features could be revealed, computed from the trees
directly. Values are on the log-odds (margin) scale, where contributions add up; on the
probability scale they would not.
"""

import numpy as np
import pandas as pd
from xgboost import DMatrix, XGBClassifier

from fraudgraph.features.graph import RISK_FEATURES, STRUCTURE_FEATURES

GROUPS = ("local", "aggregated", "structure", "risk")


def tree_shap(model: XGBClassifier, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(contributions per row and feature, bias per row), checked for additivity.

    pred_contribs returns one extra last column: the bias (the model's average margin).
    Each row's contributions plus its bias equal the model's raw output; checked here, so a
    silent mismatch (wrong feature order, wrong model) cannot reach a figure.
    """
    booster = model.get_booster()
    out = booster.predict(DMatrix(X), pred_contribs=True)
    contribs, bias = out[:, :-1], out[:, -1]
    margin = model.predict(X, output_margin=True)
    # float32 sums over hundreds of features: allow rounding, nothing more.
    if not np.allclose(contribs.sum(axis=1) + bias, margin, atol=1e-3):
        raise ValueError("SHAP values do not add up to the model output")
    return contribs, bias


def feature_group(col: str) -> str:
    if col.startswith("local_"):
        return "local"
    if col.startswith("agg_"):
        return "aggregated"
    if col in STRUCTURE_FEATURES:
        return "structure"
    # Risk columns carry their stage-1 base as a suffix: nbr_risk_max_all.
    if any(col.startswith(f"{r}_") for r in RISK_FEATURES):
        return "risk"
    raise ValueError(f"No feature group for column {col!r}")


def mean_abs_shap(contribs: np.ndarray, cols: list[str]) -> pd.Series:
    """Global importance: the mean size of each feature's contribution, largest first."""
    return pd.Series(np.abs(contribs).mean(axis=0), index=cols).sort_values(ascending=False)


def group_shares(contribs: np.ndarray, cols: list[str]) -> pd.Series:
    """Each feature group's share of the total mean |SHAP|, in GROUPS order."""
    imp = mean_abs_shap(contribs, cols)
    by_group = imp.groupby([feature_group(c) for c in imp.index]).sum()
    return (by_group / by_group.sum()).reindex(GROUPS, fill_value=0.0)
