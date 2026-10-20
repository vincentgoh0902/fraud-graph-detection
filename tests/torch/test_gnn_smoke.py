"""Smoke run of the Phase 4 GNNs on the synthetic dataset: GCN and GraphSAGE through the
protocol and MLflow, with embeddings saved from each final run. Checks that it runs and
logs; says nothing about model quality. (The hybrid half is in tests/test_hybrid.py,
which must not import torch; D-030.)"""

import mlflow
import pandas as pd
import pytest

from fraudgraph.models.gnn import run_gnns

GNN_CFG = {
    "phase": "4",
    "feature_sets": ["local", "all"],
    "seeds": [0, 1],
    "evaluation": {"k": 2, "extra_ks": [5], "fn_cost": 10, "fp_cost": 1, "n_bootstrap": 10},
    "training": {
        "max_epochs": 5,
        "patience": 3,
        "dropout": 0.5,
        "weight_decay": 5e-4,
        "focal_gamma": 2.0,
    },
    "models": {
        "gcn": {"grid": {"hidden": [4], "lr": [0.01], "loss": ["weighted_ce", "focal"]}},
        "sage": {"grid": {"hidden": [4], "lr": [0.01], "loss": ["weighted_ce"]}},
    },
}


@pytest.fixture
def tracking(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("smoke")
    yield
    mlflow.end_run()


def test_gnns_run_log_and_save_embeddings(data, split_cfg, tracking, tmp_path):
    emb_dir = tmp_path / "emb"
    results = run_gnns(data, split_cfg, GNN_CFG, run_group="g1", embeddings_dir=emb_dir)
    assert set(results) == {(m, fs) for m in ("gcn", "sage") for fs in ("local", "all")}

    runs = mlflow.search_runs(experiment_names=["smoke"])
    final = runs[runs["tags.role"] == "final"]
    assert len(final) == 2 * 2 * 2  # models x feature sets x seeds
    assert (final["tags.phase"] == "4").all()
    assert final["metrics.test_pr_auc"].between(0, 1).all()
    # Every run logs where its checkpoint came from (D-027) and the training-node PR-AUC
    # for the embedding skew check (D-028).
    for col in ("best_epoch", "train_pr_auc"):
        assert runs[f"metrics.{col}"].notna().all(), col

    # Each final run saved its embeddings under its own run id, one row per transaction.
    for _, run in final.iterrows():
        table = pd.read_parquet(emb_dir / run["tags.embeddings_file"])
        assert run["tags.embeddings_file"] == f"{run['run_id']}.parquet"
        assert table["tx_id"].tolist() == data.nodes["tx_id"].tolist()
        assert list(table.columns[1:]) == ["emb_1", "emb_2", "emb_3", "emb_4"]
