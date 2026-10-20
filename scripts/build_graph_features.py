"""Build the Phase 3 graph feature table and save it to data/processed/graph_features.parquet.

Usage:
    uv run python scripts/load_neo4j.py              # once, with Neo4j running
    uv run python scripts/build_graph_features.py    # structural features from GDS
    uv run python scripts/build_graph_features.py --backend networkx   # no Neo4j (D-019)

Steps:
1. Structural features (degree, PageRank, clustering, Louvain) from GDS or NetworkX.
2. Stage-1 risk for each base feature set (local, all), out-of-fold on the training steps,
   with the Phase 2 XGBoost winner's settings read from MLflow (D-001, D-018).
3. Community and neighbour-risk features; write the table.
Checks are logged to MLflow (experiment graph_features, role "features").
"""

import argparse
import logging
import sys
import time

import mlflow
import numpy as np
from scipy.stats import ks_2samp

from fraudgraph.config import load_config, resolve
from fraudgraph.data.load import load_elliptic
from fraudgraph.data.split import assign_period, load_split_config, temporal_split
from fraudgraph.evaluation.metrics import pr_auc
from fraudgraph.features.graph import build_graph_features
from fraudgraph.features.stage1 import fit_stage1
from fraudgraph.features.tabular import feature_matrix
from fraudgraph.graph.structural import structural_features
from fraudgraph.tracking import setup_tracking, winning_params


def structural(backend: str, data, cfg):
    if backend == "networkx":
        return structural_features(data.nodes, data.edges, cfg["pagerank"], cfg["louvain"])
    from fraudgraph.graph.neo4j_client import connect, structural_gds

    with connect() as driver:
        return structural_gds(driver, cfg["neo4j"]["database"], cfg["pagerank"], cfg["louvain"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["gds", "networkx"], default="gds")
    args = parser.parse_args()
    logging.getLogger("neo4j").setLevel(logging.ERROR)  # GDS deprecation notices

    cfg = load_config("graph")
    exp_cfg = load_config("graph_features")
    baselines = load_config("baselines")
    data = load_elliptic()
    split_cfg = load_split_config()
    split = temporal_split(data.nodes, split_cfg)
    y = data.nodes["label"].to_numpy()
    in_val = (assign_period(data.nodes, split_cfg) == "val").to_numpy()

    start = time.perf_counter()
    struct = structural(args.backend, data, cfg)
    print(f"Structural features ({args.backend}): {time.perf_counter() - start:.0f} s")

    setup_tracking(baselines["experiment"])  # read the Phase 2 winners
    s1 = cfg["stage1"]
    risks, logged_params, metrics = {}, {"backend": args.backend}, {}
    for base in s1["bases"]:
        params = winning_params(baselines["experiment"], "xgboost", base)
        res = fit_stage1(data, split_cfg, base, params, s1["seed"], s1["n_folds"])
        risks[base] = res.risk
        logged_params |= {f"stage1_{base}_{k}": v for k, v in params.items()}

        # Same seed, settings and training rows as Phase 2's final seed-0 model, so its
        # validation PR-AUC should equal the one logged there.
        metrics[f"stage1_{base}_val_pr_auc"] = pr_auc(y[split.val], res.risk[split.val])

        # D-001's known limitation: fold models see less data than the full model. On the
        # same validation nodes, compare the fold models' average score with the full
        # model's score, so time drift doesn't mix into the comparison.
        # All validation-period nodes, labelled or not: all of them are neighbours.
        X_val = feature_matrix(data, base)[in_val]
        fold_avg = np.mean([m.predict_proba(X_val)[:, 1] for m in res.models[:-1]], axis=0)
        full = res.models[-1].predict_proba(X_val)[:, 1]
        metrics[f"stage1_{base}_fold_vs_full_ks"] = float(ks_2samp(fold_avg, full).statistic)
        metrics[f"stage1_{base}_fold_mean_risk"] = float(fold_avg.mean())
        metrics[f"stage1_{base}_full_mean_risk"] = float(full.mean())

    table = build_graph_features(data.nodes, data.edges, struct, risks)
    metrics["n_communities"] = int(table["community_id"].nunique())

    out = resolve(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)

    setup_tracking(exp_cfg["experiment"])
    with mlflow.start_run(run_name=f"graph_features/{args.backend}", tags={"role": "features"}):
        mlflow.log_params(logged_params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(cfg, "graph.yaml")

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"Wrote {len(table)} rows to {out} ({time.perf_counter() - start:.0f} s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
