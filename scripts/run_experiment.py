"""Run an experiment config through the evaluation protocol, logging to MLflow.

Usage:
    uv run python scripts/run_experiment.py                         # Phase 2 baselines
    uv run python scripts/run_experiment.py --models xgboost --feature-sets all
    uv run python scripts/run_experiment.py --config graph_features # Phase 3

Each invocation is one run_group (a timestamp tag); the results table uses the
latest run_group per model and feature set.
"""

import argparse
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from fraudgraph.config import load_config, load_experiment_config, resolve
from fraudgraph.data.load import load_elliptic
from fraudgraph.data.split import load_split_config
from fraudgraph.features.graph import attach_graph_features
from fraudgraph.models.baselines import run_baselines
from fraudgraph.tracking import setup_tracking


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="baselines")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--feature-sets", nargs="+")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    setup_tracking(cfg["experiment"])
    run_group = datetime.now().strftime("%Y%m%d-%H%M%S")

    data = load_elliptic()
    feature_sets = args.feature_sets or cfg["feature_sets"]
    if any("+" in fs for fs in feature_sets):  # a graph feature set
        graph_table = pd.read_parquet(resolve(load_config("graph")["output"]))
        data = attach_graph_features(data, graph_table)

    start = time.perf_counter()
    results = run_baselines(
        data,
        load_split_config(),
        cfg,
        run_group,
        models=args.models,
        feature_sets=feature_sets,
    )

    k = cfg["evaluation"]["k"]
    print(f"\nrun_group {run_group}, {time.perf_counter() - start:.0f} s")
    for (model, fs), r in results.items():
        pr = np.mean([m["test_pr_auc"] for m in r.final_metrics])
        pk = np.mean([m[f"test_precision_at_{k}"] for m in r.final_metrics])
        print(f"  {model:14s} {fs:15s} test PR-AUC {pr:.3f}  P@{k} {pk:.3f}  params {r.params}")
    print("Write the README table with: uv run python scripts/make_results_table.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
