"""Evaluation metrics for the illicit class.

Conventions used everywhere:
- y is 1 for illicit, 0 for licit; unknown-label nodes are filtered out before this.
- A transaction is flagged when score >= threshold.
- Illicit is the positive class, so precision/recall/F1 are about catching illicit
  transactions. Accuracy is never reported: predicting "licit" for everything is ~90%
  accurate on labelled Elliptic nodes and catches nothing.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve, computed as average precision.

    Average precision is a step-wise sum, not a trapezoid; trapezoidal PR-AUC
    interpolates linearly between points and overstates the area (D-016).
    Undefined (NaN) without any illicit example.
    """
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, score))


def illicit_prf(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    flagged = score >= threshold
    tp = int((flagged & (y == 1)).sum())
    n_flagged, n_pos = int(flagged.sum()), int(y.sum())
    # Nothing flagged or nothing to find gives 0, not NaN, so averages stay defined.
    precision = tp / n_flagged if n_flagged else 0.0
    recall = tp / n_pos if n_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _counts_by_threshold(y: np.ndarray, score: np.ndarray):
    """For each distinct score t (descending): true and false positives when flagging >= t."""
    order = np.argsort(-score, kind="stable")
    s, yy = score[order], y[order]
    tp, fp = np.cumsum(yy), np.cumsum(1 - yy)
    # Keep the last position of each run of equal scores: tied scores are flagged together.
    last = np.r_[s[1:] != s[:-1], True]
    return s[last], tp[last], fp[last]


def f1_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """The threshold maximising illicit F1, and that F1. Used on validation only."""
    thresholds, tp, fp = _counts_by_threshold(y, score)
    n_pos = y.sum()
    # F1 = 2TP / (2TP + FP + FN), and FN = n_pos - TP. Same as 2PR/(P+R), without
    # dividing by zero when TP = 0.
    f1 = 2 * tp / (tp + fp + n_pos)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def cost_threshold(y: np.ndarray, score: np.ndarray, fn_cost: float, fp_cost: float) -> float:
    """The threshold minimising fn_cost * missed + fp_cost * false alarms.

    Chosen empirically rather than as fp_cost / (fn_cost + fp_cost): that formula only
    holds for calibrated probabilities, and class-weighted models are not calibrated.
    Returns inf if flagging nothing is cheapest.
    """
    thresholds, tp, fp = _counts_by_threshold(y, score)
    n_pos = y.sum()
    cost = fn_cost * (n_pos - tp) + fp_cost * fp
    # On ties argmin takes the first, i.e. the highest threshold: fewest alerts.
    best = int(np.argmin(cost))
    if fn_cost * n_pos <= cost[best]:
        return float("inf")
    return float(thresholds[best])


def precision_recall_at_k(y: np.ndarray, score: np.ndarray, k: int) -> tuple[float, float]:
    """Precision and recall among the k highest-scoring transactions.

    Models the investigator who reviews the top k alerts. If fewer than k exist, all are
    reviewed. Ties at the cut-off are scored by their expected value under random
    tie-breaking: a random forest gives many transactions the same score, and a plain
    sort would let row order decide who gets reviewed (D-015).
    """
    n = len(y)
    k = min(k, n)
    n_pos = y.sum()
    if k == 0:
        return 0.0, float("nan")
    cutoff = np.sort(score)[::-1][k - 1]
    above = score > cutoff
    at = score == cutoff
    slots = k - above.sum()  # places left for the tied group
    hits = y[above].sum() + slots * y[at].sum() / at.sum()
    recall = hits / n_pos if n_pos else float("nan")
    return float(hits / k), float(recall)


def per_step_metrics(
    y: np.ndarray, score: np.ndarray, steps: np.ndarray, threshold: float, ks: list[int]
) -> pd.DataFrame:
    """One row per time step. n_illicit is reported so the cap on precision@k is visible."""
    rows = {}
    for step in np.unique(steps):
        m = steps == step
        ys, ss = y[m], score[m]
        row = {"n_labelled": int(m.sum()), "n_illicit": int(ys.sum()), "pr_auc": pr_auc(ys, ss)}
        row |= illicit_prf(ys, ss, threshold)
        if row["n_illicit"] == 0:
            row["recall"] = row["f1"] = float("nan")  # nothing to find: undefined, not 0
        for k in ks:
            row[f"precision_at_{k}"], row[f"recall_at_{k}"] = precision_recall_at_k(ys, ss, k)
        rows[int(step)] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("time_step")


def bootstrap_threshold(
    y: np.ndarray, score: np.ndarray, n_boot: int, seed: int
) -> dict[str, float]:
    """95% percentile intervals for the F1-tuned threshold and its F1 (D-005).

    Resamples validation nodes with replacement and re-tunes each time. A wide interval
    means the threshold is fitted to noise in a handful of illicit examples.
    Limitation: nodes in one step are correlated, so node-level resampling somewhat
    understates the uncertainty.
    """
    rng = np.random.default_rng(seed)
    thresholds, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue  # no illicit in this resample: no threshold to tune
        t, f1 = f1_threshold(y[idx], score[idx])
        thresholds.append(t)
        f1s.append(f1)
    if not thresholds:
        nan = float("nan")
        return {"threshold_lo": nan, "threshold_hi": nan, "f1_lo": nan, "f1_hi": nan}
    t_lo, t_hi = np.percentile(thresholds, [2.5, 97.5])
    f_lo, f_hi = np.percentile(f1s, [2.5, 97.5])
    return {
        "threshold_lo": float(t_lo),
        "threshold_hi": float(t_hi),
        "f1_lo": float(f_lo),
        "f1_hi": float(f_hi),
    }
