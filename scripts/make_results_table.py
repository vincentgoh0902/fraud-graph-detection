"""Write the results tables into README.md from the logged MLflow runs.

Usage:
    uv run python scripts/make_results_table.py

Replaces the text between <!-- results:start --> and <!-- results:end --> (Phase 2) and
between <!-- results-phase3:start --> and <!-- results-phase3:end --> (Phase 3). Every
number comes from MLflow; nothing in those blocks is edited by hand.
"""

import sys

import mlflow
import pandas as pd

from fraudgraph.config import REPO_ROOT, load_experiment_config
from fraudgraph.evaluation.report import (
    comparison_table,
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
        block = (
            results_tables(graph_runs, k)
            + "\n\n**XGBoost: what each feature set adds** (test steps 35–49; change = "
            "difference of seed means)\n\n" + comparison_table(both, "xgboost", k)
        )
        text = replace_between_markers(text, block, "results-phase3")
        print(block)
    readme.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
