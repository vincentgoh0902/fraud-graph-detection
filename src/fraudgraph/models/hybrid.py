"""Hybrid: XGBoost on tabular features plus GNN node embeddings (Phase 4).

"local+emb" = (a) local features + embeddings of a GNN trained on local features;
"all+emb"   = (b) all features + embeddings of a GNN trained on all features.
Which GNN (GCN or GraphSAGE) supplies them is decided on validation: the one whose final
runs have the higher mean validation PR-AUC for that feature set.

Seed s of XGBoost uses the embeddings of seed s's GNN, so the seed spread reported for the
hybrid includes the GNN's variation. Embeddings come from the GNN's kept checkpoint, which
was trained on the training nodes' labels, so training-node embeddings are in-sample
while validation/test ones are not. That can only lower validation and test scores, not
inflate them; why we accept it instead of out-of-fold embeddings is in D-028.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudgraph.data.load import EllipticData
from fraudgraph.data.split import (
    SplitConfig,
    TemporalSplit,
    assert_no_cross_split_edges,
    temporal_split,
)
from fraudgraph.evaluation.protocol import EvalConfig, FitFn, ProtocolResult, run_protocol
from fraudgraph.features.tabular import feature_matrix
from fraudgraph.models.baselines import make_model

EMB_SUFFIX = "+emb"

# (GNN model, {seed: embeddings file}) per base feature set
EmbeddingSource = tuple[str, dict[int, Path]]


def pick_embedding_source(
    gnn_finals: pd.DataFrame, base_fs: str, embeddings_dir: Path
) -> EmbeddingSource:
    """The GNN with the best mean validation PR-AUC on base_fs, and its embedding files.

    gnn_finals: the final runs to choose from (the latest run group of each GNN).
    """
    runs = gnn_finals[gnn_finals["tags.feature_set"] == base_fs]
    if runs.empty:
        raise LookupError(f"No final GNN runs for feature set {base_fs!r}")
    val = runs.groupby("tags.model")["metrics.val_pr_auc"].mean()
    model = str(val.idxmax())
    chosen = runs[runs["tags.model"] == model]
    paths = {}
    for _, run in chosen.iterrows():
        path = embeddings_dir / run["tags.embeddings_file"]
        if not path.exists():
            raise FileNotFoundError(f"{path} (run {run['run_id']}) is missing; re-run the GNNs")
        paths[int(run["tags.seed"])] = path
    return model, paths


def load_embeddings(path: Path, tx_ids: np.ndarray) -> np.ndarray:
    """Embedding matrix with rows in the order of tx_ids, joined by tx_id, not position."""
    table = pd.read_parquet(path).set_index("tx_id").reindex(tx_ids)
    if table.isna().any().any():
        raise ValueError(f"{path} has no embedding for some transactions")
    return table.to_numpy(dtype=np.float32)


def hybrid_fit_fn(X_by_seed: dict[int, np.ndarray], y: np.ndarray, split: TemporalSplit) -> FitFn:
    """Like baselines.tabular_fit_fn, but each seed gets its own GNN's embeddings."""

    def fit(params, seed):
        X = X_by_seed[seed]
        y_train = y[split.train]
        model = make_model("xgboost", params, seed, y_train)
        model.fit(X[split.train], y_train)
        return lambda period: model.predict_proba(X[getattr(split, period)])[:, 1]

    return fit


def run_hybrid(
    data: EllipticData,
    split_cfg: SplitConfig,
    cfg: dict[str, Any],
    run_group: str,
    sources: dict[str, EmbeddingSource],
    feature_sets: list[str] | None = None,
) -> dict[tuple[str, str], ProtocolResult]:
    """XGBoost through the protocol on each "<base>+emb" feature set.

    sources maps a base feature set ('local', 'all') to its embedding source.
    """
    assert_no_cross_split_edges(data.nodes, data.edges, split_cfg)
    split = temporal_split(data.nodes, split_cfg)
    y = data.nodes["label"].to_numpy().astype(np.int64)
    steps = data.nodes["time_step"].to_numpy()
    tx_ids = data.nodes["tx_id"].to_numpy()
    eval_cfg = EvalConfig(**{**cfg["evaluation"], "extra_ks": tuple(cfg["evaluation"]["extra_ks"])})

    results = {}
    for fs in feature_sets or cfg["feature_sets"]:
        if not fs.endswith(EMB_SUFFIX):
            raise ValueError(f"Hybrid feature sets end in {EMB_SUFFIX!r}; got {fs!r}")
        base = fs.removesuffix(EMB_SUFFIX)
        gnn, paths = sources[base]
        if set(paths) != set(cfg["seeds"]):
            raise ValueError(f"Embeddings for seeds {sorted(paths)}, need {cfg['seeds']}")
        X_base = feature_matrix(data, base)
        X_by_seed = {s: np.hstack([X_base, load_embeddings(p, tx_ids)]) for s, p in paths.items()}
        tags = {"phase": str(cfg["phase"]), "model": "xgboost", "feature_set": fs,
                "run_group": run_group, "embedding_model": gnn}  # fmt: skip
        results[("xgboost", fs)] = run_protocol(
            hybrid_fit_fn(X_by_seed, y, split),
            cfg["models"]["xgboost"]["grid"],
            cfg["seeds"],
            y,
            steps,
            split,
            eval_cfg,
            tags,
            ids=tx_ids,
        )
    return results
