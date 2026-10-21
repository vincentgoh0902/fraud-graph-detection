"""Distribution shift between the training period and the test periods (D-033).

Three views of "what changed after step 42":
- period_metrics: is the model still *ranking* illicit transactions on top, with only the
  threshold out of date, or has the ranking itself broken?
- psi / feature_psi: which input features moved, one feature at a time.
- adversarial_validation: can a classifier tell the periods apart at all, using every
  feature together, and which features does it use?
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from fraudgraph.analysis.explain import tree_shap
from fraudgraph.evaluation.metrics import f1_threshold, illicit_prf, pr_auc, precision_recall_at_k

Steps = tuple[int, int]

# Floor for an empty bin's share, so log(0) never happens. A bin empty in one period and
# not the other then contributes a large but finite amount, which is the right signal.
PSI_EPS = 1e-4


def psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    """Population stability index of cur against ref.

    PSI = sum over bins of (cur% - ref%) * ln(cur% / ref%). Bin edges are ref's quantiles,
    so each bin holds about 1/n_bins of the reference. The outer bins are open-ended: values
    beyond ref's range still count. Rule of thumb: < 0.1 stable, > 0.25 a large shift.
    """
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1))[1:-1])

    # Many Elliptic columns are mostly one value, so several quantiles land on it and
    # collapse into one edge. Each edge therefore gets its own "exactly equal" bin:
    # left + right search gives 2*i for values between edges i-1 and i, and 2*i + 1 for
    # values equal to edge i. A point mass is then never merged with its neighbours.
    def bins(x):
        return np.searchsorted(edges, x, side="left") + np.searchsorted(edges, x, side="right")

    ref_bin, cur_bin = bins(ref), bins(cur)
    n = 2 * len(edges) + 1
    p = np.maximum(np.bincount(ref_bin, minlength=n) / len(ref), PSI_EPS)
    q = np.maximum(np.bincount(cur_bin, minlength=n) / len(cur), PSI_EPS)
    return float(np.sum((q - p) * np.log(q / p)))


def feature_psi(
    ref: pd.DataFrame, cur: pd.DataFrame, cols: list[str], n_bins: int = 10
) -> pd.Series:
    """PSI per feature, largest first. NaNs (graph features of isolated nodes) are dropped."""
    out = {c: psi(ref[c].dropna().to_numpy(), cur[c].dropna().to_numpy(), n_bins) for c in cols}
    return pd.Series(out, name="psi").sort_values(ascending=False)


def grouped_folds(target: np.ndarray, groups: np.ndarray, n_folds: int, seed: int):
    """Folds that keep each time step whole and put both classes in every test fold.

    A step split across folds would let the classifier recognise the step (its market
    conditions, its graph) rather than the period, which overstates the AUC.
    """
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(target)), target, groups))


def adversarial_validation(
    X: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    params: dict,
    n_folds: int,
    seed: int,
) -> dict:
    """Out-of-fold AUC of a classifier telling target=1 rows from target=0 rows.

    AUC near 0.5: the periods look alike. Near 1: they are easy to tell apart.
    importance: mean |SHAP| per feature over the out-of-fold rows, i.e. where they differ.
    Class weights balance the two periods, which differ in size.
    """
    oof = np.zeros(len(target))
    importance = np.zeros(X.shape[1])
    for train_idx, test_idx in grouped_folds(target, groups, n_folds, seed):
        yt = target[train_idx]
        model = XGBClassifier(
            **params,
            scale_pos_weight=(len(yt) - yt.sum()) / yt.sum(),
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        ).fit(X[train_idx], yt)
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        contribs, _ = tree_shap(model, X[test_idx])
        importance += np.abs(contribs).sum(axis=0)
    return {"auc": float(roc_auc_score(target, oof)), "importance": importance / len(target)}


def period_metrics(
    y: np.ndarray,
    score: np.ndarray,
    steps: np.ndarray,
    threshold: float,
    k: int,
    periods: dict[str, Steps],
) -> pd.DataFrame:
    """Ranking vs threshold, one row per period (labelled test rows only).

    precision/recall/f1: at the validation-tuned threshold (the deployed decision).
    recall_at_k: per-step mean, like the protocol; it needs no threshold.
    oracle_f1: per-step mean of the best F1 any threshold could reach on that step, chosen
    with the step's own labels. A diagnostic ceiling, never a result (D-033): if it is high
    while f1 is low, the threshold is the problem; if it is low too, the ranking is.
    """
    rows = {}
    for name, (first, last) in periods.items():
        m = (steps >= first) & (steps <= last)
        ys, ss, st = y[m], score[m], steps[m]
        per_step = [s for s in np.unique(st) if ys[st == s].sum() > 0]  # oracle needs illicit
        row = {"n_labelled": int(m.sum()), "n_illicit": int(ys.sum()), "pr_auc": pr_auc(ys, ss)}
        row |= illicit_prf(ys, ss, threshold)
        row["recall_at_k"] = float(
            np.mean([precision_recall_at_k(ys[st == s], ss[st == s], k)[1] for s in per_step])
        )
        row["oracle_f1"] = float(np.mean([f1_threshold(ys[st == s], ss[st == s])[1]
                                          for s in per_step]))  # fmt: skip
        illicit_scores = ss[ys == 1]
        row["median_illicit_score"] = float(np.median(illicit_scores))
        row["median_licit_score"] = float(np.median(ss[ys == 0]))
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def time_trend(nodes: pd.DataFrame, cols: list[str], steps: np.ndarray) -> pd.Series:
    """Spearman correlation between time step and each feature's per-step median.

    Near +-1: the feature's typical value moves steadily with time, so it works as a hidden
    clock even though time_step itself is not a feature (D-013, D-036). Pass training-period
    rows only, so the check uses no test data. Largest |correlation| first.
    """
    medians = nodes[cols].groupby(steps).median()
    rho = medians.corrwith(pd.Series(medians.index, index=medians.index), method="spearman")
    return rho.reindex(rho.abs().sort_values(ascending=False).index)
