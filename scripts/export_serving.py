"""Export what the app serves: scores, SHAP and edges for steps 35-49 (D-038 to D-040).

Usage:
    uv run python scripts/export_serving.py

Reads the served model (seed 0 of the validation-selected model, D-039) from the latest
Phase 5 analysis run, scores every transaction in the served steps, and checks the
labelled ones against the test scores logged in MLflow: the app must show exactly the
model the README reports. Writes artifacts/serving/ (full stack) and artifacts/space/
(the public demo subset). Then, for the full stack: scripts/write_predictions.py.
"""

import json
import sys
from datetime import UTC, datetime

import mlflow
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fraudgraph.analysis.explain import feature_group, tree_shap
from fraudgraph.config import load_config, load_experiment_config, resolve
from fraudgraph.data.load import UNKNOWN, load_elliptic
from fraudgraph.evaluation.report import latest_finals
from fraudgraph.evaluation.runs import runs_for, runs_of
from fraudgraph.features.graph import attach_graph_features
from fraudgraph.features.tabular import feature_columns, feature_matrix
from fraudgraph.serving.artifacts import write_artifacts
from fraudgraph.serving.export import demo_subset, top_contributions, transactions_table
from fraudgraph.tracking import setup_tracking


def latest_analysis(experiment: str) -> dict:
    """meta.json of the most recent Phase 5 analysis run: the selected model and its refits."""
    setup_tracking(experiment)
    runs = mlflow.search_runs(experiment_names=[experiment])
    runs = runs[runs["tags.role"] == "analysis"].sort_values("tags.run_group")
    if runs.empty:
        raise RuntimeError("No analysis run: run scripts/run_analysis.py first")
    run = runs.iloc[-1]
    path = mlflow.artifacts.download_artifacts(run_id=run["run_id"], artifact_path="meta.json")
    with open(path) as f:
        return json.load(f) | {"analysis_run_group": run["tags.run_group"]}


def served_run(selected: dict, seed: int) -> pd.Series:
    """The final protocol run of the selected model for this seed."""
    finals = latest_finals(runs_of(load_experiment_config(selected["experiment"])["experiment"]))
    runs = runs_for(finals, selected["model"], selected["feature_set"])
    run = runs[runs["tags.seed"].astype(int) == seed]
    if len(run) != 1:
        raise RuntimeError(f"Expected one final run for seed {seed}, found {len(run)}")
    return run.iloc[0]


def logged_table(run_id: str, name: str) -> pd.DataFrame:
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name)
    return pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)


def main() -> int:
    cfg = load_config("serving")
    first, last = cfg["steps"]
    seed = cfg["seed"]
    analysis = latest_analysis(cfg["analysis_experiment"])
    sel = analysis["selected"]
    fitted = analysis["models"][str(seed)]
    fs = sel["feature_set"]
    run = served_run(sel, seed)
    if not np.isclose(fitted["threshold"], run["metrics.threshold_f1"], rtol=0, atol=1e-12):
        raise RuntimeError("Analysis threshold differs from the logged protocol run")
    print(f"Serving {sel['model']} / {fs}, seed {seed} (run {run['run_id'][:8]})")

    data = load_elliptic()
    data = attach_graph_features(data, pd.read_parquet(resolve(load_config("graph")["output"])))
    nodes = data.nodes
    cols = feature_columns(data, fs)
    X = feature_matrix(data, fs)
    m = nodes["time_step"].between(first, last).to_numpy()

    model = XGBClassifier()
    model.load_model(fitted["path"])
    scores = model.predict_proba(X[m])[:, 1]

    # The app must serve the reported model: labelled scores equal the logged test scores.
    ids, labels = nodes["tx_id"].to_numpy()[m], nodes["label"].to_numpy()[m]
    logged = logged_table(run["run_id"], "test_scores.parquet")
    lab = labels != UNKNOWN
    if not (
        np.array_equal(ids[lab], logged["tx_id"].to_numpy())
        and np.array_equal(scores[lab], logged["score"].to_numpy())
    ):
        raise RuntimeError("Exported scores differ from the logged test scores (D-038)")
    print(f"  {lab.sum()} labelled scores match MLflow exactly; {len(ids)} transactions served")

    contribs, bias = tree_shap(model, X[m])
    if not np.allclose(bias, bias[0]):
        raise RuntimeError("SHAP base value should be the same for every row")
    groups = [feature_group(c) for c in cols]
    shap = top_contributions(ids, contribs, cols, groups, cfg["top_shap"])
    tx = transactions_table(
        ids,
        nodes["time_step"].to_numpy()[m],
        labels,
        scores,
        fitted["threshold"],
        nodes["community_id"].to_numpy()[m],
    )
    served = set(ids)
    edges = data.edges[data.edges["src"].isin(served)].reset_index(drop=True)
    per_step = logged_table(run["run_id"], "per_step_test.csv")

    meta = {
        "model_version": f"{sel['model']}-{fs}-seed{seed}-{run['tags.run_group']}",
        "experiment": sel["experiment"],
        "feature_set": fs,
        "seed": seed,
        "run_id": run["run_id"],
        "analysis_run_group": analysis["analysis_run_group"],
        "threshold": float(fitted["threshold"]),
        "base_value": float(bias[0]),
        "steps": [first, last],
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "demo": False,
        "has_full_graph": True,
        "labels_included": True,
        "max_hops": cfg["limits"]["max_hops"],
        "neighbourhood_roots": None,
    }
    out = resolve(cfg["output_dir"])
    write_artifacts(out, tx, shap, edges, meta, per_step)
    print(f"  wrote {out} ({len(edges)} edges)")

    d = cfg["demo"]
    sub_tx, sub_shap, sub_edges, roots = demo_subset(
        tx, shap, edges, d["alerts_per_step"], d["hops"], d["include_labels"]
    )
    demo_meta = meta | {
        "demo": True,
        "has_full_graph": False,
        "labels_included": bool(d["include_labels"]),
        "max_hops": d["hops"],
        "neighbourhood_roots": roots,
    }
    demo_out = resolve(d["output_dir"])
    write_artifacts(demo_out, sub_tx, sub_shap, sub_edges, demo_meta, per_step)
    print(f"  wrote {demo_out} ({len(sub_edges)} of {len(edges)} edges, {len(roots)} alerts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
