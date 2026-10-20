"""Build README results tables from logged MLflow runs, so no number is typed by hand."""

import re

import pandas as pd

MODEL_NAMES = {
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
FEATURE_NAMES = {
    "local": "(a) local",
    "all": "(b) local + aggregated",
    "local+graph": "(c) local + graph",
    "all+graph": "(b) + graph",
    "local+structure": "local + structure only",
}
# The comparisons that answer the research question (D-003, D-021), as (from, to).
COMPARISONS = [
    ("local", "all"),
    ("local", "local+graph"),
    ("local", "local+structure"),
    ("all", "all+graph"),
]


def latest_finals(runs: pd.DataFrame) -> pd.DataFrame:
    """Final runs from the most recent run_group of each (model, feature set).

    Re-running an experiment adds a new run_group (a timestamp); older ones stay in
    MLflow for the record but don't enter the table.
    """
    final = runs[runs["tags.role"] == "final"]
    latest = final.groupby(["tags.model", "tags.feature_set"])["tags.run_group"].transform("max")
    return final[final["tags.run_group"] == latest]


def mean_std(values: pd.Series) -> str:
    v = values.to_numpy(dtype=float)
    if len(v) == 1:
        return f"{v[0]:.3f}"  # deterministic model: one run, no spread (D-006)
    # ddof=1: sample standard deviation across seeds.
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def _table(finals: pd.DataFrame, columns: dict[str, str]) -> str:
    header = ["Model", "Features", "Seeds", *columns.values()]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    order = [(m, f) for f in FEATURE_NAMES for m in MODEL_NAMES]
    for model, fs in order:
        g = finals[(finals["tags.model"] == model) & (finals["tags.feature_set"] == fs)]
        if g.empty:
            continue
        cells = [MODEL_NAMES[model], FEATURE_NAMES[fs], str(len(g))]
        cells += [mean_std(g[f"metrics.{c}"]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def results_tables(runs: pd.DataFrame, k: int) -> str:
    finals = latest_finals(runs)
    ranking = {
        "val_pr_auc": "Val PR-AUC",
        "test_pr_auc": "Test PR-AUC",
        f"test_precision_at_{k}": f"Precision@{k}",
        f"test_recall_at_{k}": f"Recall@{k}",
    }
    decisions = {
        "test_precision": "Precision",
        "test_recall": "Recall",
        "test_f1": "F1",
        "test_cost_precision": "Precision (cost)",
        "test_cost_recall": "Recall (cost)",
        "test_cost_f1": "F1 (cost)",
    }
    return (
        "**Ranking quality** (test steps 35–49; precision/recall@K are per-step means)\n\n"
        + _table(finals, ranking)
        + "\n\n**Decisions at the validation-tuned threshold** (left: max-F1 threshold; "
        "right: cost threshold, a miss costs 10× a false alarm)\n\n"
        + _table(finals, decisions)
        + "\n\nMean ± sample standard deviation over seeds."
    )


def replace_between_markers(text: str, block: str, name: str) -> str:
    """Replace the content between <!-- name:start --> and <!-- name:end -->."""
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), flags=re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"Markers {start} ... {end} not found")
    # A function replacement, so backslashes in the block are not read as escapes.
    return pattern.sub(lambda _: f"{start}\n{block}\n{end}", text)


def comparison_table(finals: pd.DataFrame, model: str, k: int) -> str:
    """Change in each metric along each planned comparison, for one model.

    The change is the difference of the seed means. Seeds are not paired across feature
    sets, so no spread is given for it; the per-set spreads are in the main tables.
    """
    metrics = {
        "test_pr_auc": "test PR-AUC",
        f"test_precision_at_{k}": f"precision@{k}",
        "test_f1": "F1 (max-F1 threshold)",
    }
    header = ["From", "To", *[f"{name}: from → to (change)" for name in metrics.values()]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    m = finals[finals["tags.model"] == model]
    for src, dst in COMPARISONS:
        a, b = m[m["tags.feature_set"] == src], m[m["tags.feature_set"] == dst]
        if a.empty or b.empty:
            continue
        cells = [FEATURE_NAMES[src], FEATURE_NAMES[dst]]
        for col in metrics:
            x, y = a[f"metrics.{col}"].mean(), b[f"metrics.{col}"].mean()
            change = round(y - x, 3) + 0.0  # + 0.0 turns -0.0 into 0.0, so no "-0.000"
            cells.append(f"{x:.3f} → {y:.3f} ({change:+.3f})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _signed(v: float) -> str:
    return f"{round(v, 3) + 0.0:+.3f}"  # + 0.0 turns -0.0 into 0.0


def interval_table(intervals: dict[tuple[str, str], dict[str, float]], k: int) -> str:
    """The paired step bootstrap (D-024) for each comparison, in COMPARISONS order."""
    header = [
        "From",
        "To",
        "test PR-AUC change [95% interval]",
        f"precision@{k} change [95% interval]",
        "share of resamples where To has higher PR-AUC",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for src, dst in COMPARISONS:
        r = intervals.get((src, dst))
        if r is None:
            continue
        cells = [
            FEATURE_NAMES[src],
            FEATURE_NAMES[dst],
            f"{_signed(r['pr_auc_change'])} [{_signed(r['pr_auc_lo'])}, {_signed(r['pr_auc_hi'])}]",
            f"{_signed(r['precision_at_k_change'])} "
            f"[{_signed(r['precision_at_k_lo'])}, {_signed(r['precision_at_k_hi'])}]",
            f"{r['share_positive']:.0%}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
