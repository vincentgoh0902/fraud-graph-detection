"""Write the results tables into README.md from the logged MLflow runs.

Usage:
    uv run python scripts/make_results_table.py

Replaces the text between <!-- results:start --> and <!-- results:end --> (Phase 2) and
between <!-- results-phase3:start --> and <!-- results-phase3:end --> (Phase 3). Every
number comes from MLflow; nothing in those blocks is edited by hand. The Phase 3 block
includes a paired bootstrap over test steps, computed from each final run's logged
test scores (D-024).
"""

import sys

import mlflow
import numpy as np
import pandas as pd

from fraudgraph.config import REPO_ROOT, load_experiment_config
from fraudgraph.evaluation.compare import paired_step_bootstrap
from fraudgraph.evaluation.report import (
    COMPARISONS,
    comparison_table,
    interval_table,
    latest_finals,
    replace_between_markers,
    results_tables,
)
from fraudgraph.tracking import setup_tracking


def runs_of(experiment: str) -> pd.DataFrame:
    setup_tracking(experiment)
    runs = mlflow.search_runs(experiment_names=[experiment])
    if runs.empty or "tags.role" not in runs:
        return pd.DataFrame()
    return runs[runs["tags.role"].isin(["candidate", "final"])]


def load_test_scores(finals: pd.DataFrame, model: str, feature_set: str) -> list[pd.DataFrame]:
    """The logged test scores of each seed's final run, sorted by seed."""
    runs = finals[(finals["tags.model"] == model) & (finals["tags.feature_set"] == feature_set)]
    out = []
    for _, run in runs.sort_values("tags.seed").iterrows():
        path = mlflow.artifacts.download_artifacts(
            run_id=run["run_id"], artifact_path="test_scores.parquet"
        )
        out.append(pd.read_parquet(path))
    return out


def bootstrap_intervals(finals: pd.DataFrame, k: int, cfg: dict) -> dict:
    """Paired step bootstrap for each XGBoost comparison (finals: XGBoost runs only)."""
    intervals = {}
    for src, dst in COMPARISONS:
        a, b = load_test_scores(finals, "xgboost", src), load_test_scores(finals, "xgboost", dst)
        ref = a[0]
        for s in a + b:  # same transactions in the same order, so the pairing is valid
            if not s["tx_id"].equals(ref["tx_id"]):
                raise ValueError(f"Test rows differ between {src} and {dst}")
        r = paired_step_bootstrap(
            ref["label"].to_numpy(),
            ref["time_step"].to_numpy(),
            [s["score"].to_numpy() for s in a],
            [s["score"].to_numpy() for s in b],
            k=k,
            n_boot=cfg["n_bootstrap"],
            seed=cfg["seed"],
        )
        # The bootstrap recomputes the observed change from the scores; it must match
        # the change computed from the logged metrics, or the scores are not those runs'.
        by_set = finals.groupby("tags.feature_set")["metrics.test_pr_auc"].mean()
        if not np.isclose(r["pr_auc_change"], by_set[dst] - by_set[src], atol=1e-9):
            raise ValueError(f"Scores and logged metrics disagree for {src} -> {dst}")
        intervals[(src, dst)] = r
    return intervals


def main() -> int:
    base_cfg = load_experiment_config("baselines")
    graph_cfg = load_experiment_config("graph_features")
    k = base_cfg["evaluation"]["k"]
    base_runs = runs_of(base_cfg["experiment"])
    if base_runs.empty:
        print("No runs logged yet; run scripts/run_experiment.py first.", file=sys.stderr)
        return 1

    readme = REPO_ROOT / "README.md"
    text = replace_between_markers(readme.read_text(), results_tables(base_runs, k), "results")

    graph_runs = runs_of(graph_cfg["experiment"])
    if not graph_runs.empty:
        both = pd.concat([latest_finals(base_runs), latest_finals(graph_runs)])
        intervals = bootstrap_intervals(
            both[both["tags.model"] == "xgboost"], k, graph_cfg["comparison"]
        )
        block = (
            results_tables(graph_runs, k)
            + "\n\n**XGBoost: what each feature set adds** (test steps 35–49; change = "
            "difference of seed means)\n\n"
            + comparison_table(both, "xgboost", k)
            + "\n\n**How sure is each change?** Paired bootstrap over test steps: "
            f"{graph_cfg['comparison']['n_bootstrap']} resamples of the 15 test steps, both "
            "feature sets scored on the same resample ([D-024](docs/decisions.md)).\n\n"
            + interval_table(intervals, k)
        )
        text = replace_between_markers(text, block, "results-phase3")
        print(block)
    readme.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
