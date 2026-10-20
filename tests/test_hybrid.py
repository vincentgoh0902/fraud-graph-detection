"""Hybrid (XGBoost on features + GNN embeddings), without torch: the embeddings are
synthetic files standing in for a GNN's output, so this runs in the XGBoost process
(D-030). Checks the GNN choice is made on validation, the join is by tx_id, each seed gets
its own GNN's embeddings, and the pipeline runs through the protocol."""

import mlflow
import numpy as np
import pandas as pd
import pytest

from fraudgraph.models.hybrid import load_embeddings, pick_embedding_source, run_hybrid

HYBRID_CFG = {
    "phase": "4",
    "feature_sets": ["local+emb", "all+emb"],
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


def fake_gnn_finals(emb_dir, tx_ids, val):
    """Final-run rows like MLflow returns, with an embeddings file for each run.

    val maps (model, feature set) to per-seed validation PR-AUC.
    """
    rng = np.random.default_rng(0)
    emb_dir.mkdir(exist_ok=True)
    rows = []
    for (model, fs), scores in val.items():
        for seed, v in enumerate(scores):
            run_id = f"{model}-{fs}-{seed}"
            table = pd.DataFrame(rng.normal(size=(len(tx_ids), 3)), columns=["emb_1", "emb_2",
                                 "emb_3"])  # fmt: skip
            table.insert(0, "tx_id", tx_ids)
            # Shuffled rows: the hybrid must join by tx_id, not by position.
            table.sample(frac=1, random_state=seed).to_parquet(emb_dir / f"{run_id}.parquet")
            rows.append({"run_id": run_id, "tags.model": model, "tags.feature_set": fs,
                         "tags.seed": str(seed), "tags.embeddings_file": f"{run_id}.parquet",
                         "metrics.val_pr_auc": v})  # fmt: skip
    return pd.DataFrame(rows)


def test_embedding_source_is_the_better_gnn_on_validation(tmp_path):
    finals = fake_gnn_finals(
        tmp_path, np.arange(5),
        {("gcn", "local"): [0.6, 0.7], ("sage", "local"): [0.8, 0.8],
         ("gcn", "all"): [0.9, 0.9], ("sage", "all"): [0.5, 0.95]},
    )  # fmt: skip
    model, paths = pick_embedding_source(finals, "local", tmp_path)
    assert model == "sage"
    assert paths == {0: tmp_path / "sage-local-0.parquet", 1: tmp_path / "sage-local-1.parquet"}
    assert pick_embedding_source(finals, "all", tmp_path)[0] == "gcn"  # 0.90 vs mean 0.725


def test_missing_embedding_file_is_an_error(tmp_path):
    finals = fake_gnn_finals(tmp_path, np.arange(5), {("gcn", "local"): [0.5]})
    (tmp_path / "gcn-local-0.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        pick_embedding_source(finals, "local", tmp_path)


def test_load_embeddings_aligns_by_tx_id(tmp_path):
    table = pd.DataFrame({"tx_id": [30, 10, 20], "emb_1": [3.0, 1.0, 2.0]})
    path = tmp_path / "e.parquet"
    table.to_parquet(path, index=False)
    emb = load_embeddings(path, np.array([10, 20, 30]))
    assert emb[:, 0].tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        load_embeddings(path, np.array([10, 20, 30, 40]))  # a transaction with no row


def test_hybrid_runs_through_the_protocol(data, split_cfg, tracking, tmp_path):
    tx_ids = data.nodes["tx_id"].to_numpy()
    finals = fake_gnn_finals(
        tmp_path, tx_ids,
        {("gcn", "local"): [0.5, 0.5], ("sage", "all"): [0.5, 0.5]},
    )  # fmt: skip
    sources = {fs: pick_embedding_source(finals, fs, tmp_path) for fs in ("local", "all")}
    results = run_hybrid(data, split_cfg, HYBRID_CFG, run_group="h1", sources=sources)
    assert set(results) == {("xgboost", "local+emb"), ("xgboost", "all+emb")}
    final = mlflow.search_runs(filter_string="tags.role = 'final'")
    assert len(final) == 2 * 2
    by_fs = final.groupby("tags.feature_set")["tags.embedding_model"].first().to_dict()
    assert by_fs == {"local+emb": "gcn", "all+emb": "sage"}
    assert (final["tags.model"] == "xgboost").all()
    assert final["metrics.test_pr_auc"].between(0, 1).all()


def test_hybrid_needs_embeddings_for_every_seed(data, split_cfg, tracking, tmp_path):
    finals = fake_gnn_finals(tmp_path, data.nodes["tx_id"].to_numpy(), {("gcn", "local"): [0.5]})
    sources = {"local": pick_embedding_source(finals, "local", tmp_path)}
    with pytest.raises(ValueError, match="seeds"):
        run_hybrid(data, split_cfg, HYBRID_CFG, "h1", sources, feature_sets=["local+emb"])
