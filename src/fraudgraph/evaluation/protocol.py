"""The evaluation protocol every model goes through (D-009, D-014).

1. Selection: fit every grid configuration with every seed; score validation only.
   The configuration with the best mean validation PR-AUC wins.
2. Final: refit the winner per seed, check it reproduces its validation score, tune
   thresholds on validation, and only then score test.

Models plug in as a `fit(params, seed) -> score` function, where `score(period)`
returns scores for the nodes in that split mask. The same protocol then serves
logistic regression, XGBoost and (in Phase 4) GNNs. Test scores are requested in
exactly one place, after selection is finished.
"""

import itertools
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from fraudgraph.data.split import TemporalSplit
from fraudgraph.evaluation.metrics import (
    bootstrap_threshold,
    cost_threshold,
    f1_threshold,
    illicit_prf,
    per_step_metrics,
    pr_auc,
)

Scorer = Callable[[str], np.ndarray]
FitFn = Callable[[dict[str, Any], int], Scorer]


class NonDeterministicError(RuntimeError):
    """A refit with the same config and seed gave a different validation score."""


@dataclass(frozen=True)
class EvalConfig:
    k: int  # primary precision@K (D-004)
    extra_ks: tuple[int, ...]  # robustness check
    fn_cost: float  # cost of a missed illicit transaction...
    fp_cost: float  # ...relative to a false alarm
    n_bootstrap: int


@dataclass(frozen=True)
class ProtocolResult:
    params: dict[str, Any]
    final_metrics: list[dict[str, float]]  # one per seed


def expand_grid(grid: dict[str, list]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def evaluate_final(
    y_val: np.ndarray,
    val_score: np.ndarray,
    y_test: np.ndarray,
    test_score: np.ndarray,
    test_steps: np.ndarray,
    cfg: EvalConfig,
    seed: int,
):
    """Thresholds come from validation; test is only measured at them."""
    t_f1, val_f1 = f1_threshold(y_val, val_score)
    t_cost = cost_threshold(y_val, val_score, cfg.fn_cost, cfg.fp_cost)
    boot = bootstrap_threshold(y_val, val_score, cfg.n_bootstrap, seed)

    m = {
        "val_pr_auc": pr_auc(y_val, val_score),
        "val_f1": val_f1,
        "val_f1_lo": boot["f1_lo"],
        "val_f1_hi": boot["f1_hi"],
        "threshold_f1": t_f1,
        "threshold_f1_lo": boot["threshold_lo"],
        "threshold_f1_hi": boot["threshold_hi"],
        "threshold_cost": t_cost,
        "test_pr_auc": pr_auc(y_test, test_score),
    }
    m |= {f"test_{k}": v for k, v in illicit_prf(y_test, test_score, t_f1).items()}
    m |= {f"test_cost_{k}": v for k, v in illicit_prf(y_test, test_score, t_cost).items()}

    ks = [cfg.k, *cfg.extra_ks]
    steps = per_step_metrics(y_test, test_score, test_steps, t_f1, ks)
    # Per-step metrics are averaged over steps (macro): each two-week step counts
    # equally, as each is a separate batch of investigator work. nanmean skips steps
    # where a metric is undefined (no illicit).
    for k in ks:
        m[f"test_precision_at_{k}"] = float(np.nanmean(steps[f"precision_at_{k}"]))
        m[f"test_recall_at_{k}"] = float(np.nanmean(steps[f"recall_at_{k}"]))
    m["test_step_f1_mean"] = float(np.nanmean(steps["f1"]))
    return m, steps


def run_protocol(
    fit: FitFn,
    grid: dict[str, list],
    seeds: list[int],
    y: np.ndarray,
    steps: np.ndarray,
    split: TemporalSplit,
    cfg: EvalConfig,
    tags: dict[str, str],
    ids: np.ndarray | None = None,
    on_final: Callable[[Scorer, int], None] | None = None,
) -> ProtocolResult:
    """Select on validation, evaluate the selection on test, log everything to MLflow.

    y and steps cover all nodes; the split masks pick out each period's labelled ones.
    ids (transaction ids, optional) are saved with the test scores.
    on_final(scorer, seed), optional, runs inside each final run after it is scored, so it
    can save extra outputs of the kept model (GNN embeddings for the hybrid, Phase 4).
    """
    y_val = y[split.val]
    prefix = "/".join(tags[k] for k in ("model", "feature_set") if k in tags)

    # 1. Selection: validation only.
    best = None  # (mean val PR-AUC, config id, params, per-seed val PR-AUC)
    for i, params in enumerate(expand_grid(grid)):
        val_scores = []
        for seed in seeds:
            run_tags = tags | {"role": "candidate", "config_id": str(i), "seed": str(seed)}
            with mlflow.start_run(run_name=f"{prefix}/c{i}/s{seed}", tags=run_tags):
                mlflow.log_params(params | {"seed": seed})
                score = pr_auc(y_val, fit(params, seed)("val"))
                mlflow.log_metric("val_pr_auc", score)
            val_scores.append(score)
        mean = float(np.mean(val_scores))
        # Strict ">" keeps the earlier config on an exact tie; grids list simpler first.
        if best is None or mean > best[0]:
            best = (mean, i, params, val_scores)

    _, best_id, best_params, best_val = best

    # 2. Final: refit the winner and measure it on test.
    finals = []
    for seed, candidate_val in zip(seeds, best_val, strict=True):
        run_tags = tags | {"role": "final", "config_id": str(best_id), "seed": str(seed)}
        with mlflow.start_run(run_name=f"{prefix}/final/s{seed}", tags=run_tags):
            mlflow.log_params(best_params | {"seed": seed})
            scorer = fit(best_params, seed)
            val_score = scorer("val")
            # Refitting rather than keeping every candidate in memory is only valid if
            # the refit is the same model. Check it rather than assume it.
            refit_val = pr_auc(y_val, val_score)
            if not np.isclose(refit_val, candidate_val, rtol=0, atol=1e-9):
                raise NonDeterministicError(
                    f"{prefix} seed {seed}: val PR-AUC {candidate_val} at selection, "
                    f"{refit_val} on refit"
                )
            test_score = scorer("test")
            metrics, step_table = evaluate_final(
                y_val, val_score, y[split.test], test_score, steps[split.test], cfg, seed
            )
            mlflow.log_metrics(metrics)
            mlflow.log_text(step_table.to_csv(), "per_step_test.csv")
            # Per-transaction test scores, for the paired bootstrap between feature sets
            # (D-024) and per-step analysis later. Rows are in test-mask order.
            scores = pd.DataFrame(
                {"time_step": steps[split.test], "label": y[split.test], "score": test_score}
            )
            if ids is not None:
                scores.insert(0, "tx_id", ids[split.test])
            with tempfile.TemporaryDirectory() as tmp:
                path = f"{tmp}/test_scores.parquet"
                scores.to_parquet(path, index=False)
                mlflow.log_artifact(path)
            if on_final is not None:
                on_final(scorer, seed)
        finals.append(metrics)

    return ProtocolResult(params=best_params, final_metrics=finals)
