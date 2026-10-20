"""Write the results tables into README.md from the logged MLflow runs.

Usage:
    uv run python scripts/make_results_table.py

Replaces the text between <!-- results:start --> and <!-- results:end --> (Phase 2) and
between <!-- results-phase3:start --> and <!-- results-phase3:end --> (Phase 3), and
between <!-- results-phase4:start --> and <!-- results-phase4:end --> (Phase 4, plus the
figure docs/figures/per_step_f1.png). Every number comes from MLflow; nothing in those
blocks is edited by hand. The Phase 3 and 4 blocks include a paired bootstrap over test
steps, computed from each final run's logged test scores (D-024).
"""

import sys

import mlflow
import numpy as np
import pandas as pd

from fraudgraph.config import REPO_ROOT, load_experiment_config
from fraudgraph.evaluation.compare import paired_step_bootstrap
from fraudgraph.evaluation.plots import per_step_f1_figure
from fraudgraph.evaluation.report import (
    COMPARISONS,
    FEATURE_NAMES,
    MODEL_NAMES,
    all_models_table,
    checkpoint_table,
    comparison_table,
    interval_table,
    latest_finals,
    loss_table,
    model_label,
    pair_interval_table,
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


Pair = tuple[tuple[str, str], tuple[str, str]]  # ((model, feature set), (model, feature set))


def runs_for(finals: pd.DataFrame, model: str, feature_set: str) -> pd.DataFrame:
    return finals[(finals["tags.model"] == model) & (finals["tags.feature_set"] == feature_set)]


def bootstrap_pairs(finals: pd.DataFrame, pairs: list[Pair], k: int, cfg: dict) -> dict:
    """Paired step bootstrap (D-024) for each (from, to) pair of model and feature set."""
    intervals = {}
    for (ma, fa), (mb, fb) in pairs:
        a, b = load_test_scores(finals, ma, fa), load_test_scores(finals, mb, fb)
        ref = a[0]
        for s in a + b:  # same transactions in the same order, so the pairing is valid
            if not s["tx_id"].equals(ref["tx_id"]):
                raise ValueError(f"Test rows differ between {ma}/{fa} and {mb}/{fb}")
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
        logged = (
            runs_for(finals, mb, fb)["metrics.test_pr_auc"].mean()
            - runs_for(finals, ma, fa)["metrics.test_pr_auc"].mean()
        )
        if not np.isclose(r["pr_auc_change"], logged, atol=1e-9):
            raise ValueError(f"Scores and logged metrics disagree for {ma}/{fa} -> {mb}/{fb}")
        intervals[((ma, fa), (mb, fb))] = r
    return intervals


def bootstrap_intervals(finals: pd.DataFrame, k: int, cfg: dict) -> dict:
    """Paired step bootstrap for each Phase 3 XGBoost feature-set comparison."""
    pairs = [(("xgboost", src), ("xgboost", dst)) for src, dst in COMPARISONS]
    r = bootstrap_pairs(finals, pairs, k, cfg)
    return {(src, dst): r[(("xgboost", src), ("xgboost", dst))] for src, dst in COMPARISONS}


def best_gnn(finals: pd.DataFrame, feature_set: str) -> str:
    """The GNN with the higher mean validation PR-AUC on this feature set (never test)."""
    g = finals[
        finals["tags.model"].isin(["gcn", "sage"]) & (finals["tags.feature_set"] == feature_set)
    ]
    return str(g.groupby("tags.model")["metrics.val_pr_auc"].mean().idxmax())


def phase4_pairs(finals: pd.DataFrame) -> list[Pair]:
    """The Phase 4 comparisons, fixed before any Phase 4 result was seen (D-031)."""
    pairs = []
    for fs in ("local", "all"):
        pairs.append((("xgboost", fs), (best_gnn(finals, fs), fs)))
    for fs in ("local", "all"):
        pairs.append((("xgboost", fs), ("xgboost", f"{fs}+emb")))
    for fs in ("local", "all"):
        pairs.append((("gcn", fs), ("sage", fs)))
    present = {
        (m, f) for m, f in zip(finals["tags.model"], finals["tags.feature_set"], strict=True)
    }
    return [p for p in pairs if p[0] in present and p[1] in present]


# The figure's series, fixed in advance: everything built on (b), the strongest base.
FIGURE_SERIES = [
    ("xgboost", "all"),
    ("xgboost", "all+graph"),
    ("gcn", "all"),
    ("sage", "all"),
    ("xgboost", "all+emb"),
]
FIGURE_LABELS = {
    ("xgboost", "all"): "XGBoost (b)",
    ("xgboost", "all+graph"): "XGBoost (b) + graph features",
    ("gcn", "all"): "GCN (b)",
    ("sage", "all"): "GraphSAGE (b)",
    ("xgboost", "all+emb"): "Hybrid: XGBoost (b) + embeddings",
}


def load_step_tables(finals: pd.DataFrame, model: str, feature_set: str) -> list[pd.DataFrame]:
    out = []
    for _, run in runs_for(finals, model, feature_set).sort_values("tags.seed").iterrows():
        path = mlflow.artifacts.download_artifacts(
            run_id=run["run_id"], artifact_path="per_step_test.csv"
        )
        out.append(pd.read_csv(path, index_col="time_step"))
    return out


def phase4_block(finals: pd.DataFrame, gnn_runs: pd.DataFrame, k: int, cfg: dict) -> str:
    phase4 = finals[finals["tags.phase"] == "4"]
    pairs = phase4_pairs(finals)
    intervals = bootstrap_pairs(finals, pairs, k, cfg)
    rows = [(model_label(*a), model_label(*b), intervals[(a, b)]) for a, b in pairs]
    chosen = ", ".join(
        f"{FEATURE_NAMES[fs]}: {MODEL_NAMES[best_gnn(finals, fs)]}" for fs in ("local", "all")
    )

    series = {
        FIGURE_LABELS[key]: load_step_tables(finals, *key)
        for key in FIGURE_SERIES
        if not runs_for(finals, *key).empty
    }
    per_step_f1_figure(series, REPO_ROOT / "docs" / "figures" / "per_step_f1.png")

    return (
        results_tables(phase4, k)
        + "\n\n**All models** (test steps 35–49; F1 at the max-F1 and the cost threshold, both "
        "tuned on validation)\n\n"
        + all_models_table(finals, k)
        + "\n\n**Imbalance handling: weighted cross-entropy vs focal loss** (validation only; "
        "best grid setting per loss, mean over seeds; [D-029](docs/decisions.md))\n\n"
        + loss_table(gnn_runs)
        + "\n\n**Kept GNN checkpoints** (epoch per seed, and PR-AUC on training vs validation "
        "nodes; [D-027](docs/decisions.md), [D-028](docs/decisions.md))\n\n"
        + checkpoint_table(finals)
        + f"\n\n**How sure is each change?** Paired bootstrap over test steps: "
        f"{cfg['n_bootstrap']} resamples of the 15 test steps ([D-024](docs/decisions.md)). "
        f"Best GNN by validation PR-AUC: {chosen}.\n\n"
        + pair_interval_table(rows, k)
        + "\n\n![Illicit F1 per test step](docs/figures/per_step_f1.png)"
    )


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

    gnn_runs = runs_of(load_experiment_config("gnn")["experiment"])
    hybrid_runs = runs_of(load_experiment_config("hybrid")["experiment"])
    if not gnn_runs.empty and (gnn_runs["tags.role"] == "final").any():
        everything = pd.concat(
            [latest_finals(r) for r in (base_runs, graph_runs, gnn_runs, hybrid_runs)
             if not r.empty]
        )  # fmt: skip
        block = phase4_block(everything, gnn_runs, k, graph_cfg["comparison"])
        text = replace_between_markers(text, block, "results-phase4")
        print(block)
    readme.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
