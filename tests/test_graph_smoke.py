"""Smoke run of the Phase 3 pipeline on the synthetic dataset: stage 1, NetworkX structural
features, the feature table, then XGBoost through the protocol on every graph feature set.
Checks that it runs and logs; says nothing about model quality."""

import mlflow
import pytest

from fraudgraph.features.graph import attach_graph_features, build_graph_features
from fraudgraph.features.stage1 import fit_stage1
from fraudgraph.graph.structural import structural_features
from fraudgraph.models.baselines import run_baselines

GRAPH_SETS = ["local+graph", "all+graph", "local+structure"]
TINY_CFG = {
    "phase": "3",
    "feature_sets": GRAPH_SETS,
    "seeds": [0, 1],
    "evaluation": {"k": 2, "extra_ks": [5], "fn_cost": 10, "fp_cost": 1, "n_bootstrap": 10},
    "models": {"xgboost": {"grid": {"n_estimators": [10], "max_depth": [2]}}},
}


@pytest.fixture
def tracking(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("smoke")
    yield
    mlflow.end_run()


def test_graph_pipeline_runs_and_logs(data, split_cfg, tracking):
    struct = structural_features(
        data.nodes,
        data.edges,
        {"damping": 0.85, "max_iterations": 100, "tolerance": 1e-9},
        {"seed": 0},
    )
    params = {"n_estimators": 5, "max_depth": 2}
    risks = {
        base: fit_stage1(data, split_cfg, base, params, seed=0, n_folds=3).risk
        for base in ("local", "all")
    }
    attached = attach_graph_features(
        data, build_graph_features(data.nodes, data.edges, struct, risks)
    )

    results = run_baselines(attached, split_cfg, TINY_CFG, run_group="smoke")
    assert set(results) == {("xgboost", fs) for fs in GRAPH_SETS}
    final = mlflow.search_runs(filter_string="tags.role = 'final'")
    assert len(final) == 2 * len(GRAPH_SETS)
    assert (final["tags.phase"] == "3").all()
    assert final["metrics.test_pr_auc"].between(0, 1).all()
