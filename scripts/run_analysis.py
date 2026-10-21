"""Phase 5 analysis (D-033): distribution shift, SHAP, false negatives, retraining.

Usage:
    uv run python scripts/run_analysis.py

Reads the Phase 2-4 runs from MLflow, refits the model selected on validation, and logs
one MLflow run (experiment "analysis") whose artefacts are the tables behind the README
Findings section. Then: uv run python scripts/make_results_table.py

Nothing here selects or tunes a model on test data: the model is chosen by validation
PR-AUC, thresholds are the ones logged by the protocol, and the retraining simulation
(D-035) is a separate what-if.
"""

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fraudgraph.analysis.drift import (
    adversarial_validation,
    feature_psi,
    period_metrics,
    time_trend,
)
from fraudgraph.analysis.errors import missed_illicit, seed_agreement, standardised_difference
from fraudgraph.analysis.explain import GROUPS, feature_group, group_shares, tree_shap
from fraudgraph.analysis.retrain import retrain_step, rolling_windows
from fraudgraph.config import load_config, load_experiment_config, resolve
from fraudgraph.data.load import UNKNOWN, load_elliptic
from fraudgraph.data.split import load_split_config, temporal_split
from fraudgraph.evaluation.compare import paired_step_bootstrap
from fraudgraph.evaluation.metrics import illicit_prf, pr_auc, precision_recall_at_k
from fraudgraph.evaluation.report import latest_finals
from fraudgraph.evaluation.runs import (
    load_step_tables,
    load_test_scores,
    phase3_pairs,
    phase4_pairs,
    runs_for,
    runs_of,
)
from fraudgraph.features.graph import attach_graph_features
from fraudgraph.features.tabular import feature_columns, feature_matrix
from fraudgraph.models.baselines import fit_model
from fraudgraph.tracking import setup_tracking, winning_params


def in_period(steps: np.ndarray, period) -> np.ndarray:
    return (steps >= period[0]) & (steps <= period[1])


def all_finals(experiments: list[str]) -> pd.DataFrame:
    frames = []
    for exp in experiments:
        runs = runs_of(load_experiment_config(exp)["experiment"])
        if not runs.empty:
            frames.append(latest_finals(runs).assign(experiment=exp))
    return pd.concat(frames, ignore_index=True)


def select_model(finals: pd.DataFrame) -> tuple[str, str, str]:
    """(experiment, model, feature set) with the highest mean validation PR-AUC (D-033)."""
    means = finals.groupby(["experiment", "tags.model", "tags.feature_set"])[
        "metrics.val_pr_auc"
    ].mean()
    exp, model, fs = means.idxmax()
    print(f"Selected on validation: {model} / {fs} (val PR-AUC {means.max():.4f})")
    if model != "xgboost":
        # SHAP via pred_contribs and the refit below assume XGBoost (D-034).
        raise RuntimeError(f"Analysis expects an XGBoost winner, got {model}")
    return exp, model, fs


def refit_and_check(exp, fs, finals, X, y, split, models_dir: Path) -> dict[int, dict]:
    """Refit each seed's final model; it must reproduce the logged val PR-AUC and test scores."""
    params = winning_params(load_experiment_config(exp)["experiment"], "xgboost", fs)
    runs = runs_for(finals, "xgboost", fs).sort_values("tags.seed")
    logged_scores = load_test_scores(finals, "xgboost", fs)
    out = {}
    models_dir.mkdir(parents=True, exist_ok=True)
    for (_, run), logged in zip(runs.iterrows(), logged_scores, strict=True):
        seed = int(run["tags.seed"])
        model = fit_model("xgboost", params, seed, X, y, split.train)
        val_pr = pr_auc(y[split.val], model.predict_proba(X[split.val])[:, 1])
        test_score = model.predict_proba(X[split.test])[:, 1]
        if not np.isclose(
            val_pr, run["metrics.val_pr_auc"], rtol=0, atol=1e-9
        ) or not np.array_equal(test_score, logged["score"].to_numpy()):
            raise RuntimeError(f"Refit of seed {seed} does not reproduce the logged run (D-034)")
        name = f"xgboost_{fs.replace('+', '_')}_s{seed}"
        model.save_model(models_dir / f"{name}.ubj")
        out[seed] = {
            "model": model,
            "params": params,
            "threshold": float(run["metrics.threshold_f1"]),
            "path": str(models_dir / f"{name}.ubj"),
        }
        print(f"  seed {seed}: refit reproduces val PR-AUC {val_pr:.4f} and test scores")
    return out


def shift_tables(fits, X, y, steps, labelled, periods, k):
    """Ranking vs threshold per period (val, before, after), per seed."""
    rows = []
    eval_mask = labelled & in_period(steps, (periods["val"][0], periods["after"][1]))
    use = {p: periods[p] for p in ("val", "before", "after")}
    for seed, f in fits.items():
        score = f["model"].predict_proba(X[eval_mask])[:, 1]
        t = period_metrics(y[eval_mask], score, steps[eval_mask], f["threshold"], k, use)
        rows.append(t.assign(seed=seed))
    return pd.concat(rows).rename_axis("period").reset_index()


def illicit_scores(fits, X, y, steps, labelled, periods):
    """Every labelled val/test row's score per seed, for the score-distribution figure."""
    m = labelled & in_period(steps, (periods["val"][0], periods["after"][1]))
    out = pd.DataFrame({"time_step": steps[m], "label": y[m]})
    for seed, f in fits.items():
        out[f"score_s{seed}"] = f["model"].predict_proba(X[m])[:, 1]
        out[f"threshold_s{seed}"] = f["threshold"]
    return out


def drift_tables(nodes, cols, periods, cfg, X, y, steps):
    ref = nodes[in_period(steps, periods["reference"])]
    psi = pd.DataFrame(
        {
            p: feature_psi(ref, nodes[in_period(steps, periods[p])], cols, cfg["drift"]["psi_bins"])
            for p in ("val", "before", "after")
        }
    )
    psi["group"] = [feature_group(c) for c in psi.index]
    # Hidden clocks (D-036): training-period rows only, so no test data is involved.
    ref_m = in_period(steps, periods["reference"])
    psi["time_trend"] = time_trend(nodes[ref_m], cols, steps[ref_m]).reindex(psi.index)

    adv_cfg = cfg["adversarial"]
    adv_rows, adv_imp = [], {}
    illicit = y == 1
    for label, rows_mask in (
        ("all transactions", np.ones(len(y), bool)),
        ("illicit only", illicit),
    ):
        # The control pair splits the training period itself (D-036): it shows how
        # separable two stretches of time are with no known market event between them.
        for ref_p, p in (("reference", "before"), ("reference", "after"),
                         ("control_early", "control_late")):  # fmt: skip
            ref_m = rows_mask & in_period(steps, periods[ref_p])
            cur_m = rows_mask & in_period(steps, periods[p])
            m = ref_m | cur_m
            r = adversarial_validation(
                X[m], cur_m[m].astype(int), steps[m], adv_cfg["params"],
                adv_cfg["n_folds"], adv_cfg["seed"],
            )  # fmt: skip
            adv_rows.append({"rows": label, "reference": ref_p, "period": p,
                             "n_reference": int(ref_m.sum()), "n_period": int(cur_m.sum()),
                             "auc": r["auc"]})  # fmt: skip
            adv_imp[f"{label} / {ref_p} vs {p}"] = pd.Series(r["importance"], index=cols)
            print(f"  adversarial {label}, {ref_p} vs {p}: AUC {r['auc']:.3f}")
    return psi, pd.DataFrame(adv_rows), pd.DataFrame(adv_imp)


def period_bootstraps(finals, periods, k, cfg):
    """The planned comparisons (D-024, D-031), rerun on each test period separately."""
    rows = []
    for (ma, fa), (mb, fb) in phase3_pairs() + phase4_pairs(finals):
        a, b = load_test_scores(finals, ma, fa), load_test_scores(finals, mb, fb)
        ref = a[0]
        for s in a + b:
            if not s["tx_id"].equals(ref["tx_id"]):
                raise ValueError(f"Test rows differ between {ma}/{fa} and {mb}/{fb}")
        steps = ref["time_step"].to_numpy()
        for p in ("before", "after"):
            m = in_period(steps, periods[p])
            r = paired_step_bootstrap(
                ref["label"].to_numpy()[m], steps[m],
                [s["score"].to_numpy()[m] for s in a], [s["score"].to_numpy()[m] for s in b],
                k=k, n_boot=cfg["n_bootstrap"], seed=cfg["seed"],
            )  # fmt: skip
            rows.append({"from_model": ma, "from_fs": fa, "to_model": mb, "to_fs": fb,
                         "period": p, **r})  # fmt: skip
    return pd.DataFrame(rows)


def shap_tables(fits, X, y, steps, labelled, periods, cols):
    """Mean |SHAP| per feature and group shares, per period and seed."""
    imp_rows, share_rows, contribs_test = [], [], {}
    for seed, f in fits.items():
        for p in ("reference", "val", "before", "after"):
            m = labelled & in_period(steps, periods[p])
            contribs, _ = tree_shap(f["model"], X[m])
            imp_rows.append(
                pd.DataFrame(
                    {
                        "feature": cols,
                        "mean_abs_shap": np.abs(contribs).mean(axis=0),
                        "period": p,
                        "seed": seed,
                    }
                )  # fmt: skip
            )
            share_rows.append(group_shares(contribs, cols).rename(f"{p}/{seed}"))
        test = labelled & in_period(steps, (periods["before"][0], periods["after"][1]))
        contribs_test[seed] = tree_shap(f["model"], X[test])[0]
    importance = pd.concat(imp_rows, ignore_index=True)
    shares = pd.DataFrame(share_rows)
    shares["period"] = [i.split("/")[0] for i in shares.index]
    shares["seed"] = [int(i.split("/")[1]) for i in shares.index]
    shares = shares.reset_index(drop=True)

    # Do the same features matter after the shutdown as in training? Rank correlation of
    # per-feature importance between the reference period and each later period.
    rank_rows = []
    for seed in fits:
        imp = importance[importance["seed"] == seed].pivot(
            index="feature", columns="period", values="mean_abs_shap"
        )
        for p in ("val", "before", "after"):
            rho = spearmanr(imp["reference"], imp[p]).statistic
            rank_rows.append({"seed": seed, "period": p, "spearman_vs_reference": rho})
    return importance, shares, pd.DataFrame(rank_rows), contribs_test


def error_tables(fits, finals, nodes, X, y, steps, labelled, periods, cols, cfg, contribs_test):
    """False negatives vs true positives on test illicit transactions (D-033, item 7)."""
    test = labelled & in_period(steps, (periods["before"][0], periods["after"][1]))
    yt, st = y[test], steps[test]
    tn = nodes.loc[test].reset_index(drop=True)
    tn["degree"] = tn["in_degree"] + tn["out_degree"]
    tn["period"] = np.where(st >= periods["after"][0], "after", "before")
    illicit = yt == 1
    context_cols = ["degree", "pagerank", "clustering", "nbr_risk_mean_all", "nbr_risk_max_all",
                    "community_risk_mean_all"]  # fmt: skip

    summary, context, smd_list, shap_rows, missed_masks, scores = [], [], [], [], {}, {}
    for seed, f in fits.items():
        score = f["model"].predict_proba(X[test])[:, 1]
        scores[seed] = score
        missed = missed_illicit(yt, score, f["threshold"])
        caught = illicit & ~missed
        missed_masks[seed] = missed
        for p in ("before", "after", "all"):
            pm = np.ones(len(yt), bool) if p == "all" else (tn["period"] == p).to_numpy()
            ms, cs = score[missed & pm], score[caught & pm]
            summary.append({
                "seed": seed, "period": p, "n_illicit": int((illicit & pm).sum()),
                "n_missed": int((missed & pm).sum()),
                "missed_share": float((missed & pm).sum() / max((illicit & pm).sum(), 1)),
                "threshold": f["threshold"],
                "missed_score_q25": float(np.quantile(ms, 0.25)) if len(ms) else np.nan,
                "missed_score_median": float(np.median(ms)) if len(ms) else np.nan,
                "missed_score_q75": float(np.quantile(ms, 0.75)) if len(ms) else np.nan,
                "caught_score_median": float(np.median(cs)) if len(cs) else np.nan,
            })  # fmt: skip
        for outcome, mask in (("missed", missed), ("caught", caught)):
            row = tn.loc[mask, context_cols].median().to_dict()
            row |= {
                "seed": seed,
                "outcome": outcome,
                "n": int(mask.sum()),
                "no_neighbour_share": float(tn.loc[mask, "nbr_risk_max_all"].isna().mean()),
            }
            context.append(row)  # fmt: skip
        smd_list.append(standardised_difference(tn.loc[missed, cols], tn.loc[caught, cols], cols))
        # Signed SHAP per group: which kind of feature pushes the misses towards licit?
        c = contribs_test[seed]
        groups = np.array([feature_group(col) for col in cols])
        for outcome, mask in (("missed", missed), ("caught", caught)):
            for g in GROUPS:
                shap_rows.append(
                    {
                        "seed": seed,
                        "outcome": outcome,
                        "group": g,
                        "mean_shap": float(c[mask][:, groups == g].sum(axis=1).mean()),
                    }
                )

    smd = pd.concat(smd_list, axis=1)
    smd.columns = [f"seed_{s}" for s in fits]
    smd["mean"] = smd.mean(axis=1)
    smd = smd.reindex(smd["mean"].abs().sort_values(ascending=False).index)
    smd["group"] = [feature_group(c) for c in smd.index]

    agreement = seed_agreement([missed_masks[s][illicit] for s in fits]).rename("n_illicit")
    agreement.index.name = "seeds_missing"

    # Would another model have flagged XGBoost's misses, at its own validation threshold?
    others = []
    ids = tn["tx_id"].to_numpy()
    for model, fs in cfg["errors"]["compare_with"]:
        runs = runs_for(finals, model, fs).sort_values("tags.seed")
        for (_, run), other in zip(runs.iterrows(), load_test_scores(finals, model, fs),
                                   strict=True):  # fmt: skip
            seed = int(run["tags.seed"])
            if not np.array_equal(other["tx_id"].to_numpy(), ids):
                raise ValueError(f"Test rows of {model}/{fs} differ from the analysed model's")
            flagged = other["score"].to_numpy() >= run["metrics.threshold_f1"]
            m = missed_masks[seed]
            for p in ("before", "after", "all"):
                pm = np.ones(len(yt), bool) if p == "all" else (tn["period"] == p).to_numpy()
                others.append({
                    "model": model, "feature_set": fs, "seed": seed, "period": p,
                    "xgb_missed": int((m & pm).sum()),
                    "caught_of_xgb_missed": int((m & pm & flagged).sum()),
                    "other_recall": float((flagged & illicit & pm).sum()
                                          / max((illicit & pm).sum(), 1)),
                })  # fmt: skip
    per_tx = pd.DataFrame({"tx_id": ids, "time_step": st, "label": yt})
    for seed in fits:
        per_tx[f"score_s{seed}"] = scores[seed]
        per_tx[f"missed_s{seed}"] = missed_masks[seed]
    return (pd.DataFrame(summary), pd.DataFrame(context), smd, pd.DataFrame(shap_rows),
            agreement, pd.DataFrame(others), per_tx)  # fmt: skip


def retrain_tables(data, finals, cfg, eval_cfg, split_cfg, seeds):
    """Expanding-window retraining (D-035) next to the frozen model, step by step."""
    rcfg = cfg["retrain"]
    fs = rcfg["feature_set"]
    X = feature_matrix(data, fs)
    y = data.nodes["label"].to_numpy().astype(np.int64)
    steps = data.nodes["time_step"].to_numpy()
    params = winning_params(load_experiment_config("baselines")["experiment"], "xgboost", fs)
    frozen_scores = load_test_scores(finals, "xgboost", fs)
    frozen_runs = runs_for(finals, "xgboost", fs).sort_values("tags.seed")
    k = eval_cfg["k"]

    rows, pooled = [], []
    windows = rolling_windows(split_cfg.test_steps, rcfg["val_len"])
    for seed, frozen, (_, run) in zip(seeds, frozen_scores, frozen_runs.iterrows(), strict=True):
        assert int(run["tags.seed"]) == seed
        f_thr = run["metrics.threshold_f1"]
        decisions = {"retrained": [], "frozen": []}
        for w in windows:
            score, thr = retrain_step(X, y, steps, w, params, seed)
            fr = frozen[frozen["time_step"] == w.step]
            ys = fr["label"].to_numpy()
            if w.step == split_cfg.test_steps[0] and not np.array_equal(
                score, fr["score"].to_numpy()
            ):
                # The first window is the main protocol's split, so it must give the frozen
                # model's scores exactly; otherwise the simulation is not the same model.
                raise RuntimeError("Retraining window for the first test step != frozen model")
            for kind, s, t in (
                ("retrained", score, thr),
                ("frozen", fr["score"].to_numpy(), f_thr),
            ):
                prf = illicit_prf(ys, s, t)
                rows.append({
                    "seed": seed, "time_step": w.step, "kind": kind, "n_illicit": int(ys.sum()),
                    "pr_auc": pr_auc(ys, s), "threshold": t,
                    "f1": prf["f1"] if ys.sum() else np.nan, "recall": prf["recall"],
                    "precision": prf["precision"],
                    "recall_at_k": precision_recall_at_k(ys, s, k)[1],
                })  # fmt: skip
                decisions[kind].append(pd.DataFrame({"y": ys, "flag": s >= t, "step": w.step}))
        for kind, parts in decisions.items():
            d = pd.concat(parts)
            for p, (lo, hi) in (("before", cfg["periods"]["before"]),
                                ("after", cfg["periods"]["after"])):  # fmt: skip
                dp = d[(d["step"] >= lo) & (d["step"] <= hi)]
                tp = int((dp["flag"] & (dp["y"] == 1)).sum())
                fl, pos = int(dp["flag"].sum()), int(dp["y"].sum())
                prec, rec = (tp / fl if fl else 0.0), tp / pos
                f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
                pooled.append({"seed": seed, "kind": kind, "period": p, "n_illicit": pos,
                               "precision": prec, "recall": rec, "f1": f1})  # fmt: skip
        print(f"  retraining seed {seed} done")
    # Cross-check: the frozen rows were recomputed from scores; they must match the logged
    # per-step table of the same runs.
    per_step = pd.DataFrame(rows)
    for seed, table in zip(seeds, load_step_tables(finals, "xgboost", fs), strict=True):
        mine = per_step[(per_step["seed"] == seed) & (per_step["kind"] == "frozen")]
        if not np.allclose(mine["f1"].fillna(-1), table["f1"].fillna(-1).to_numpy()):
            raise RuntimeError("Frozen per-step F1 disagrees with the logged per-step table")
    return per_step, pd.DataFrame(pooled)


def main() -> int:
    cfg = load_config("analysis")
    eval_cfg = load_experiment_config("baselines")["evaluation"]
    k = eval_cfg["k"]
    periods = {p: tuple(v) for p, v in cfg["periods"].items()}
    split_cfg = load_split_config()
    start = time.perf_counter()

    finals = all_finals(cfg["candidates_from"])
    exp, _, fs = select_model(finals)

    data = load_elliptic()
    graph_table = pd.read_parquet(resolve(load_config("graph")["output"]))
    data = attach_graph_features(data, graph_table)
    split = temporal_split(data.nodes, split_cfg)
    y = data.nodes["label"].to_numpy().astype(np.int64)
    steps = data.nodes["time_step"].to_numpy()
    labelled = y != UNKNOWN
    cols = feature_columns(data, fs)
    X = feature_matrix(data, fs)

    print("Refitting the selected model")
    fits = refit_and_check(exp, fs, finals, X, y, split, resolve(cfg["models_dir"]))

    tables = {}
    print("Ranking vs threshold")
    tables["shift/period_metrics.csv"] = shift_tables(fits, X, y, steps, labelled, periods, k)
    tables["shift/scores.csv"] = illicit_scores(fits, X, y, steps, labelled, periods)
    print("Feature drift and adversarial validation")
    psi, adv, adv_imp = drift_tables(data.nodes, cols, periods, cfg, X, y, steps)
    tables["shift/psi.csv"], tables["shift/adversarial.csv"] = psi, adv
    tables["shift/adversarial_importance.csv"] = adv_imp
    print("Planned comparisons per period")
    tables["shift/period_bootstrap.csv"] = period_bootstraps(finals, periods, k, cfg["comparison"])
    print("SHAP")
    importance, shares, ranks, contribs_test = shap_tables(
        fits, X, y, steps, labelled, periods, cols
    )
    tables["shap/importance.csv"], tables["shap/group_shares.csv"] = importance, shares
    tables["shap/rank_correlation.csv"] = ranks
    print("False negatives")
    summary, context, smd, fn_shap, agreement, others, per_tx = error_tables(
        fits, finals, data.nodes, X, y, steps, labelled, periods, cols, cfg, contribs_test
    )
    tables["errors/summary.csv"], tables["errors/context.csv"] = summary, context
    tables["errors/feature_contrast.csv"], tables["errors/group_shap.csv"] = smd, fn_shap
    tables["errors/seed_agreement.csv"], tables["errors/other_models.csv"] = agreement, others
    tables["errors/test_outcomes.csv"] = per_tx
    print("Retraining simulation")
    per_step, pooled = retrain_tables(data, finals, cfg, eval_cfg, split_cfg, list(fits))
    tables["retrain/per_step.csv"], tables["retrain/pooled.csv"] = per_step, pooled

    setup_tracking(cfg["experiment"])
    run_group = datetime.now().strftime("%Y%m%d-%H%M%S")
    tags = {"phase": "5", "role": "analysis", "model": "xgboost", "feature_set": fs,
            "run_group": run_group}  # fmt: skip
    with mlflow.start_run(run_name=f"analysis/{fs}", tags=tags):
        mlflow.log_params(fits[next(iter(fits))]["params"] | {"selected_from": exp})
        for _, r in adv.iterrows():
            name = f"adv_auc_{r['rows']}_{r['reference']}_vs_{r['period']}".replace(" ", "_")
            mlflow.log_metric(name, r["auc"])
        with tempfile.TemporaryDirectory() as tmp:
            for name, table in tables.items():
                path = Path(tmp) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                table.to_csv(path)
            meta = {
                "selected": {"experiment": exp, "model": "xgboost", "feature_set": fs},
                "models": {s: {"path": f["path"], "threshold": f["threshold"]}
                           for s, f in fits.items()},
                "periods": periods,
            }  # fmt: skip
            (Path(tmp) / "meta.json").write_text(json.dumps(meta, indent=2))
            mlflow.log_artifacts(tmp)
    print(f"\nrun_group {run_group}, {time.perf_counter() - start:.0f} s")
    print("Write the README Findings with: uv run python scripts/make_results_table.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
