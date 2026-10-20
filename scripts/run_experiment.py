"""Run an experiment config through the evaluation protocol, logging to MLflow.

Usage:
    uv run python scripts/run_experiment.py                         # Phase 2 baselines
    uv run python scripts/run_experiment.py --models xgboost --feature-sets all
    uv run python scripts/run_experiment.py --config graph_features # Phase 3
    uv run python scripts/run_experiment.py --config gnn            # Phase 4 GNNs
    uv run python scripts/run_experiment.py --config hybrid         # Phase 4 hybrid

Each invocation is one run_group (a timestamp tag); the results table uses the
latest run_group per model and feature set.

The GNN and hybrid runners are imported only when used: torch and XGBoost must not run
in the same process (D-030). The GNN runner never imports XGBoost, and the hybrid only
reads the embeddings the GNN runs saved.
"""

import argparse
import sys
import time
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd

from fraudgraph.config import load_config, load_experiment_config, resolve
from fraudgraph.data.load import load_elliptic
from fraudgraph.data.split import load_split_config
from fraudgraph.features.graph import attach_graph_features
from fraudgraph.tracking import setup_tracking


def run(cfg, data, run_group, models, feature_sets):
    runner = cfg.get("runner", "tabular")
    split_cfg = load_split_config()
    if runner == "gnn":
        from fraudgraph.models.gnn import run_gnns

        emb_dir = resolve(cfg["embeddings_dir"])
        return run_gnns(data, split_cfg, cfg, run_group, emb_dir, models, feature_sets)
    if runner == "hybrid":
        from fraudgraph.evaluation.report import latest_finals
        from fraudgraph.models.hybrid import EMB_SUFFIX, pick_embedding_source, run_hybrid

        gnn_cfg = load_experiment_config(cfg["embeddings_from"])
        gnn_runs = mlflow.search_runs(experiment_names=[gnn_cfg["experiment"]])
        finals = latest_finals(gnn_runs[gnn_runs["tags.role"].isin(["final"])])
        emb_dir = resolve(gnn_cfg["embeddings_dir"])
        sources = {}
        for fs in feature_sets:
            base = fs.removesuffix(EMB_SUFFIX)
            sources[base] = pick_embedding_source(finals, base, emb_dir)
            print(f"{fs}: embeddings from {sources[base][0]}")
        return run_hybrid(data, split_cfg, cfg, run_group, sources, feature_sets)
    from fraudgraph.models.baselines import run_baselines

    return run_baselines(data, split_cfg, cfg, run_group, models, feature_sets)


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
    if any(fs.endswith(("+graph", "+structure")) for fs in feature_sets):
        graph_table = pd.read_parquet(resolve(load_config("graph")["output"]))
        data = attach_graph_features(data, graph_table)

    start = time.perf_counter()
    results = run(cfg, data, run_group, args.models, feature_sets)

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
