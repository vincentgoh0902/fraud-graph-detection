"""README Findings tables (Phase 5), built from the analysis run's artefacts.

Every function takes a table exactly as scripts/run_analysis.py logged it, so no number in
the Findings block is typed by hand.
"""

import pandas as pd

from fraudgraph.analysis.explain import GROUPS
from fraudgraph.evaluation.plots import GROUP_LABELS
from fraudgraph.evaluation.report import FEATURE_NAMES, MODEL_NAMES, mean_std, signed

PERIOD_NAMES = {
    "reference": "Training (1–29)",
    "val": "Validation (30–34)",
    "before": "Test before (35–42)",
    "after": "Test after (43–49)",
    "control_early": "Training early (1–24)",
    "control_late": "Training late (25–29)",
}
# Features whose per-step median tracks the step this closely are treated as clocks (D-036).
CLOCK_RHO = 0.95


def per_seed(values: pd.Series) -> str:
    """Counts differ by seed only slightly; list them rather than average them."""
    v = [str(int(x)) for x in values]
    return v[0] if len(set(v)) == 1 else " / ".join(v)


def _lines(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def period_table(pm: pd.DataFrame, k: int) -> str:
    cols = {
        "pr_auc": "PR-AUC",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "recall_at_k": f"Recall@{k}",
        "oracle_f1": "Oracle F1 (diagnostic)",
        "median_illicit_score": "Median illicit score",
    }
    rows = []
    for p in ("val", "before", "after"):
        g = pm[pm["period"] == p]
        rows.append([PERIOD_NAMES[p], str(int(g["n_illicit"].iloc[0]))]
                    + [mean_std(g[c]) for c in cols])  # fmt: skip
    return _lines(["Period", "Illicit", *cols.values()], rows)


def drift_table(psi: pd.DataFrame, large: float) -> str:
    rows = []
    for p in ("val", "before", "after"):
        row = [PERIOD_NAMES[p], f"{int((psi[p] > large).sum())} of {len(psi)}"]
        row += [f"{psi.loc[psi['group'] == g, p].median():.2f}" for g in GROUPS]
        rows.append(row)
    header = ["Period vs training", f"Features with PSI > {large}",
              *[f"Median PSI: {GROUP_LABELS[g].lower()}" for g in GROUPS]]  # fmt: skip
    return _lines(header, rows)


def clock_table(psi: pd.DataFrame, shap_rank: pd.Series) -> str:
    """Features that move steadily with time in the training period, and their SHAP rank."""
    clocks = psi[psi["time_trend"].abs() >= CLOCK_RHO].sort_values("time_trend", key=abs,
                                                                     ascending=False)  # fmt: skip
    rows = [
        [f"`{f}`", f"{r['time_trend']:+.3f}", f"{r['after']:.2f}",
         f"{int(shap_rank[f])} of {len(shap_rank)}"]
        for f, r in clocks.iterrows()
    ]  # fmt: skip
    header = ["Feature", "Correlation of per-step median with step (training)",
              "PSI, test after", "SHAP importance rank (training)"]  # fmt: skip
    return _lines(header, rows)


def adversarial_table(adv: pd.DataFrame) -> str:
    rows = [
        [r["rows"], PERIOD_NAMES[r["reference"]], PERIOD_NAMES[r["period"]],
         f"{r['n_reference']:,} / {r['n_period']:,}", f"{r['auc']:.3f}"]
        for _, r in adv.iterrows()
    ]  # fmt: skip
    return _lines(["Transactions", "Period A", "Period B", "Rows A / B", "Out-of-fold AUC"], rows)


def period_bootstrap_table(pb: pd.DataFrame, k: int) -> str:
    rows = []
    for _, r in pb.iterrows():
        rows.append([
            f"{MODEL_NAMES[r['from_model']]}, {FEATURE_NAMES[r['from_fs']]}",
            f"{MODEL_NAMES[r['to_model']]}, {FEATURE_NAMES[r['to_fs']]}",
            PERIOD_NAMES[r["period"]],
            f"{signed(r['pr_auc_change'])} [{signed(r['pr_auc_lo'])}, {signed(r['pr_auc_hi'])}]",
            f"{signed(r['precision_at_k_change'])} "
            f"[{signed(r['precision_at_k_lo'])}, {signed(r['precision_at_k_hi'])}]",
        ])  # fmt: skip
    header = ["From", "To", "Period", "test PR-AUC change [95% interval]",
              f"precision@{k} change [95% interval]"]  # fmt: skip
    return _lines(header, rows)


def group_share_table(shares: pd.DataFrame, ranks: pd.DataFrame) -> str:
    rows = []
    for p in ("reference", "val", "before", "after"):
        g = shares[shares["period"] == p]
        row = [PERIOD_NAMES[p], *[mean_std(g[c]) for c in GROUPS]]
        rho = ranks.loc[ranks["period"] == p, "spearman_vs_reference"]
        row.append("–" if rho.empty else mean_std(rho))
        rows.append(row)
    header = ["Period", *[GROUP_LABELS[c] for c in GROUPS],
              "Rank correlation of feature importance with training"]  # fmt: skip
    return _lines(header, rows)


def missed_table(summary: pd.DataFrame) -> str:
    rows = []
    for p in ("before", "after", "all"):
        g = summary[summary["period"] == p]
        name = "All test (35–49)" if p == "all" else PERIOD_NAMES[p]
        rows.append([name, str(int(g["n_illicit"].iloc[0])), per_seed(g["n_missed"]),
                     mean_std(g["missed_share"]), mean_std(g["missed_score_median"]),
                     mean_std(g["caught_score_median"]), mean_std(g["threshold"])])  # fmt: skip
    header = ["Period", "Illicit", "Missed (per seed)", "Share missed", "Median score, missed",
              "Median score, caught", "Threshold"]  # fmt: skip
    return _lines(header, rows)


def context_table(context: pd.DataFrame, group_shap: pd.DataFrame) -> str:
    cols = {"degree": "Degree", "nbr_risk_mean_all": "Neighbour risk, mean",
            "nbr_risk_max_all": "Neighbour risk, max"}  # fmt: skip
    rows = []
    for outcome in ("missed", "caught"):
        g = context[context["outcome"] == outcome]
        s = group_shap[group_shap["outcome"] == outcome]
        row = [outcome.capitalize(), per_seed(g["n"])] + [mean_std(g[c]) for c in cols]
        row += [mean_std(s.loc[s["group"] == grp, "mean_shap"]) for grp in GROUPS]
        rows.append(row)
    header = ["Illicit test transactions", "Count (per seed)",
              *[f"Median {v.lower()}" for v in cols.values()],
              *[f"Mean SHAP: {GROUP_LABELS[g].lower()}" for g in GROUPS]]  # fmt: skip
    return _lines(header, rows)


def other_models_table(others: pd.DataFrame) -> str:
    rows = []
    for (model, fs), g in others.groupby(["model", "feature_set"], sort=False):
        for p in ("before", "after"):
            gp = g[g["period"] == p]
            rows.append([f"{MODEL_NAMES[model]}, {FEATURE_NAMES[fs]}", PERIOD_NAMES[p],
                         per_seed(gp["xgb_missed"]), per_seed(gp["caught_of_xgb_missed"]),
                         mean_std(gp["other_recall"])])  # fmt: skip
    header = ["Other model", "Period", "XGBoost misses (per seed)",
              "Of those, flagged by the other model (per seed)",
              "Other model's own recall"]  # fmt: skip
    return _lines(header, rows)


def retrain_table(pooled: pd.DataFrame) -> str:
    rows = []
    for p in ("before", "after"):
        for kind in ("frozen", "retrained"):
            g = pooled[(pooled["period"] == p) & (pooled["kind"] == kind)]
            n = str(int(g["n_illicit"].iloc[0]))
            rows.append([PERIOD_NAMES[p], kind.capitalize(), n,
                         *[mean_std(g[c]) for c in ("precision", "recall", "f1")]])  # fmt: skip
    return _lines(["Period", "Model", "Illicit", "Precision", "Recall", "F1"], rows)


def retrain_step_table(per_step: pd.DataFrame) -> str:
    m = per_step.groupby(["time_step", "kind"])[["f1", "pr_auc"]].mean().unstack("kind")
    n = per_step.groupby("time_step")["n_illicit"].first()
    rows = [
        [str(t), str(int(n[t])), f"{m.loc[t, ('f1', 'frozen')]:.3f}",
         f"{m.loc[t, ('f1', 'retrained')]:.3f}", f"{m.loc[t, ('pr_auc', 'frozen')]:.3f}",
         f"{m.loc[t, ('pr_auc', 'retrained')]:.3f}"]
        for t in m.index
    ]  # fmt: skip
    header = ["Step", "Illicit", "F1 frozen", "F1 retrained", "PR-AUC frozen",
              "PR-AUC retrained"]  # fmt: skip
    return _lines(header, rows)
