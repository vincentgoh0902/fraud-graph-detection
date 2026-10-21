"""Reading logged runs and their artefacts back from MLflow.

Shared by the README results script and the Phase 5 analysis, so both read the same runs
and the same planned comparisons.
"""

import mlflow
import pandas as pd

from fraudgraph.evaluation.report import COMPARISONS
from fraudgraph.tracking import setup_tracking

Pair = tuple[tuple[str, str], tuple[str, str]]  # ((model, feature set), (model, feature set))


def runs_of(experiment: str) -> pd.DataFrame:
    setup_tracking(experiment)
    runs = mlflow.search_runs(experiment_names=[experiment])
    if runs.empty or "tags.role" not in runs:
        return pd.DataFrame()
    return runs[runs["tags.role"].isin(["candidate", "final"])]


def runs_for(finals: pd.DataFrame, model: str, feature_set: str) -> pd.DataFrame:
    return finals[(finals["tags.model"] == model) & (finals["tags.feature_set"] == feature_set)]


def load_test_scores(finals: pd.DataFrame, model: str, feature_set: str) -> list[pd.DataFrame]:
    """The logged test scores of each seed's final run, sorted by seed."""
    out = []
    for _, run in runs_for(finals, model, feature_set).sort_values("tags.seed").iterrows():
        path = mlflow.artifacts.download_artifacts(
            run_id=run["run_id"], artifact_path="test_scores.parquet"
        )
        out.append(pd.read_parquet(path))
    return out


def load_step_tables(finals: pd.DataFrame, model: str, feature_set: str) -> list[pd.DataFrame]:
    out = []
    for _, run in runs_for(finals, model, feature_set).sort_values("tags.seed").iterrows():
        path = mlflow.artifacts.download_artifacts(
            run_id=run["run_id"], artifact_path="per_step_test.csv"
        )
        out.append(pd.read_csv(path, index_col="time_step"))
    return out


def best_gnn(finals: pd.DataFrame, feature_set: str) -> str:
    """The GNN with the higher mean validation PR-AUC on this feature set (never test)."""
    g = finals[
        finals["tags.model"].isin(["gcn", "sage"]) & (finals["tags.feature_set"] == feature_set)
    ]
    return str(g.groupby("tags.model")["metrics.val_pr_auc"].mean().idxmax())


def phase3_pairs() -> list[Pair]:
    """The Phase 3 XGBoost feature-set comparisons (D-003, D-021)."""
    return [(("xgboost", src), ("xgboost", dst)) for src, dst in COMPARISONS]


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
