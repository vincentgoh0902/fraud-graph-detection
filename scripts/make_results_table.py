"""Write the results tables into README.md from the logged MLflow runs.

Usage:
    uv run python scripts/make_results_table.py

Replaces the text between <!-- results:start --> and <!-- results:end --> (Phase 2) and
between <!-- results-phase3:start --> and <!-- results-phase3:end --> (Phase 3), and
between <!-- results-phase4:start --> and <!-- results-phase4:end --> (Phase 4, plus the
figure docs/figures/per_step_f1.png). Every number comes from MLflow; nothing in those
blocks is edited by hand. The Phase 3 and 4 blocks include a paired bootstrap over test
steps, computed from each final run's logged test scores (D-024).

Phase 5: between <!-- findings:start --> and <!-- findings:end -->, the tables and figures of
the latest analysis run (scripts/run_analysis.py, D-033).

Phase 6: between <!-- latency:start --> and <!-- latency:end -->, the latest latency run per
graph backend (scripts/measure_latency.py).
"""

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from fraudgraph.analysis import report as findings
from fraudgraph.analysis.explain import feature_group
from fraudgraph.config import REPO_ROOT, load_config, load_experiment_config
from fraudgraph.evaluation.compare import paired_step_bootstrap
from fraudgraph.evaluation.plots import (
    group_shares_figure,
    illicit_scores_figure,
    per_step_f1_figure,
    retrain_figure,
    top_features_figure,
)
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
    per_step_table,
    replace_between_markers,
    results_tables,
)
from fraudgraph.evaluation.runs import (
    Pair,
    best_gnn,
    load_step_tables,
    load_test_scores,
    phase3_pairs,
    phase4_pairs,
    runs_for,
    runs_of,
)
from fraudgraph.serving.latency import latency_table
from fraudgraph.tracking import setup_tracking


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
    r = bootstrap_pairs(finals, phase3_pairs(), k, cfg)
    return {(src, dst): r[(("xgboost", src), ("xgboost", dst))] for src, dst in COMPARISONS}


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
        + "\n\n<details><summary>The figure's numbers (illicit F1 per test step, mean over "
        "seeds)</summary>\n\n" + per_step_table(series) + "\n\n</details>"
    )


def findings_block(k: int) -> str | None:
    """The Phase 5 Findings tables and figures, from the latest analysis run."""
    cfg = load_config("analysis")
    setup_tracking(cfg["experiment"])
    runs = mlflow.search_runs(experiment_names=[cfg["experiment"]],
                              filter_string="tags.role = 'analysis'")  # fmt: skip
    if runs.empty:
        return None
    run = runs.sort_values("tags.run_group").iloc[-1]
    root = Path(mlflow.artifacts.download_artifacts(run_id=run["run_id"]))

    def t(name: str) -> pd.DataFrame:
        return pd.read_csv(root / name, index_col=0)

    psi, imp = t("shift/psi.csv"), t("shap/importance.csv")
    train_imp = (imp[imp["period"] == "reference"].groupby("feature")["mean_abs_shap"].mean()
                 .sort_values(ascending=False))  # fmt: skip
    shap_rank = pd.Series(range(1, len(train_imp) + 1), index=train_imp.index)
    figures = REPO_ROOT / "docs" / "figures"
    seed = 0
    illicit_scores_figure(t("shift/scores.csv"), seed, figures / "illicit_scores.png")
    group_shares_figure(t("shap/group_shares.csv"), figures / "shap_groups.png")
    top = train_imp.head(cfg["shap"]["top_n"])
    top_features_figure(top, [feature_group(f) for f in top.index],
                        figures / "shap_top_features.png")  # fmt: skip
    retrain_figure(t("retrain/per_step.csv"), figures / "retrain.png")

    per_step = t("retrain/per_step.csv")
    smd = t("errors/feature_contrast.csv").head(cfg["errors"]["top_n"])
    contrast = ", ".join(f"`{f}` ({r['mean']:+.2f})" for f, r in smd.iterrows())
    agree = t("errors/seed_agreement.csv")["n_illicit"]
    return (
        f"Analysis run `{run['tags.run_group']}`: XGBoost on "
        f"{FEATURE_NAMES[run['tags.feature_set']]}, the model with the best validation PR-AUC "
        "([D-033](docs/decisions.md)), refitted and "
        "checked against its logged scores ([D-034](docs/decisions.md)). Mean ± std over 3 "
        "seeds.\n\n"
        "**1. Ranking or threshold?** (labelled transactions; precision, recall and F1 at the "
        "validation-tuned threshold; recall@K and oracle F1 are per-step means; the oracle "
        "picks each step's best threshold from that step's own labels, a ceiling, not a result)"
        "\n\n"
        + findings.period_table(t("shift/period_metrics.csv"), k)
        + "\n\n![Scores of illicit transactions](docs/figures/illicit_scores.png)"
        "\n\n**2. Feature drift against the training period** (PSI per feature, all "
        "transactions)\n\n"
        + findings.drift_table(psi, cfg["drift"]["large_psi"])
        + "\n\n**Hidden clocks**: features whose per-step median moves with the time step in the "
        f"training period (|Spearman| ≥ {findings.CLOCK_RHO}; [D-036](docs/decisions.md))\n\n"
        + findings.clock_table(psi, shap_rank)
        + "\n\n**3–4. Adversarial validation**: can a classifier tell two periods apart? (folds "
        "grouped by time step; the training early vs late row is a control added after the "
        "first results, [D-036](docs/decisions.md))\n\n"
        + findings.adversarial_table(t("shift/adversarial.csv"))
        + "\n\n**5. Do the planned comparisons hold in both periods?** (paired bootstrap over "
        "the steps of each period, [D-024](docs/decisions.md))\n\n"
        + findings.period_bootstrap_table(t("shift/period_bootstrap.csv"), k)
        + "\n\n**6. SHAP** (TreeSHAP, log-odds scale)\n\n"
        "![Top features by mean |SHAP|](docs/figures/shap_top_features.png)\n\n"
        "![SHAP share by feature group](docs/figures/shap_groups.png)\n\n"
        + findings.group_share_table(t("shap/group_shares.csv"), t("shap/rank_correlation.csv"))
        + "\n\n**7. False negatives** (illicit test transactions below the validation-tuned "
        "threshold)\n\n"
        + findings.missed_table(t("errors/summary.csv"))
        + "\n\nMissed vs caught: graph context (medians) and mean signed SHAP per feature group "
        "(negative pushes towards licit)\n\n"
        + findings.context_table(t("errors/context.csv"), t("errors/group_shap.csv"))
        + f"\n\nLargest feature differences, missed vs caught (standardised mean difference): "
        f"{contrast}.\n\nOf the {int(agree.sum())} illicit test transactions, "
        f"{int(agree.iloc[-1])} are missed by all three seeds and {int(agree.iloc[0])} by none; "
        f"{int(agree.iloc[1:-1].sum())} depend on the seed.\n\n"
        "Would another model have caught XGBoost's misses? (each at its own validation threshold, "
        "seed matched to seed)\n\n"
        + findings.other_models_table(t("errors/other_models.csv"))
        + "\n\n**8. Retraining simulation** ([D-035](docs/decisions.md); uses test-period labels, "
        "so not comparable with the results above)\n\n"
        "![Frozen vs retrained F1 per step](docs/figures/retrain.png)\n\n"
        + findings.retrain_table(t("retrain/pooled.csv"))
        + "\n\n<details><summary>Per step</summary>\n\n"
        + findings.retrain_step_table(per_step)
        + "\n\n</details>"
    )


def latency_block() -> str | None:
    """The latest latency run of each backend, as one table."""
    lat = load_config("serving")["latency"]
    setup_tracking(lat["experiment"])
    runs = mlflow.search_runs(experiment_names=[lat["experiment"]])
    if runs.empty:
        return None
    runs = runs[runs["tags.role"] == "latency"].sort_values("tags.run_group")
    by_backend, notes = {}, []
    for backend, g in runs.groupby("tags.backend"):
        run = g.iloc[-1]
        path = mlflow.artifacts.download_artifacts(
            run_id=run["run_id"], artifact_path="latency.csv"
        )
        by_backend[backend] = pd.read_csv(path)
        notes.append(run)
    run = notes[0]
    return (
        latency_table(by_backend)
        + f"\n\n{int(run['params.n_requests'])} requests per endpoint after "
        f"{int(run['params.warmup'])} warm-up requests; {run['params.client']}; "
        f"`{run['params.machine']}`. Model `{run['tags.model_version']}`."
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
    block = findings_block(k)
    if block is not None:
        text = replace_between_markers(text, block, "findings")
        print(block)
    block = latency_block()
    if block is not None:
        text = replace_between_markers(text, block, "latency")
        print(block)
    readme.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
