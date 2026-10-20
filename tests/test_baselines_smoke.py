"""Smoke-training run (the brief's CI requirement): every baseline, end to end through
the protocol and MLflow, on the synthetic dataset from conftest. Checks the pipeline
runs and logs; says nothing about model quality."""

import mlflow
import pandas as pd
import pytest

from fraudgraph.models.baselines import MODELS, run_baselines

TINY_CFG = {
    "feature_sets": ["local", "all"],
    "seeds": [0, 1],
    "evaluation": {"k": 2, "extra_ks": [5], "fn_cost": 10, "fp_cost": 1, "n_bootstrap": 10},
    "models": {
        "logreg": {"deterministic": True, "grid": {"C": [0.1, 1.0]}},
        "random_forest": {"grid": {"n_estimators": [10], "max_depth": [None, 3]}},
        "xgboost": {"grid": {"n_estimators": [10], "max_depth": [2]}},
    },
}


@pytest.fixture
def tracking(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("smoke")
    yield
    mlflow.end_run()


def test_all_baselines_run_and_log(data, split_cfg, tracking):
    results = run_baselines(data, split_cfg, TINY_CFG, run_group="smoke")
    assert set(results) == {(m, fs) for m in MODELS for fs in ("local", "all")}

    final = mlflow.search_runs(filter_string="tags.role = 'final'")
    # logreg: 1 seed; the others: 2 seeds; times 2 feature sets.
    assert len(final) == (1 + 2 + 2) * 2
    assert final["metrics.test_pr_auc"].between(0, 1).all()
    assert final["metrics.test_precision_at_2"].notna().all()


def test_final_runs_log_test_scores(data, split_cfg, tracking):
    run_baselines(data, split_cfg, TINY_CFG, run_group="smoke", models=["xgboost"])
    final = mlflow.search_runs(filter_string="tags.role = 'final'").iloc[0]
    path = mlflow.artifacts.download_artifacts(
        run_id=final["run_id"], artifact_path="test_scores.parquet"
    )
    scores = pd.read_parquet(path)
    assert list(scores.columns) == ["tx_id", "time_step", "label", "score"]
    assert set(scores["label"]) <= {0, 1}  # labelled test nodes only
    assert scores["time_step"].min() >= split_cfg.test_steps[0]
