"""Build README results tables from logged MLflow runs, so no number is typed by hand."""

import re

import pandas as pd

MODEL_NAMES = {
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
FEATURE_NAMES = {"local": "(a) local", "all": "(b) local + aggregated"}


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
