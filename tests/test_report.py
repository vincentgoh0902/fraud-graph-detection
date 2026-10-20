import pandas as pd
import pytest

from fraudgraph.evaluation.report import (
    comparison_table,
    latest_finals,
    mean_std,
    replace_between_markers,
    results_tables,
)


def fake_runs():
    rows = []
    # An old xgboost run group that must be ignored, then the current one.
    for group, base in [("20260101-0000", 0.1), ("20260201-0000", 0.5)]:
        for seed in range(3):
            rows.append(("xgboost", "all", group, "final", base + seed * 0.1))
    rows.append(("logreg", "local", "20260201-0000", "final", 0.4))
    rows.append(("logreg", "local", "20260201-0000", "candidate", 0.99))  # never in table
    df = pd.DataFrame(
        rows, columns=["tags.model", "tags.feature_set", "tags.run_group", "tags.role", "v"]
    )
    for m in ["val_pr_auc", "test_pr_auc", "test_precision_at_20", "test_recall_at_20",
              "test_precision", "test_recall", "test_f1",
              "test_cost_precision", "test_cost_recall", "test_cost_f1"]:  # fmt: skip
        df[f"metrics.{m}"] = df["v"]
    return df


def test_latest_finals_drops_old_groups_and_candidates():
    f = latest_finals(fake_runs())
    assert len(f) == 4  # 3 xgboost seeds from the new group + 1 logreg
    assert set(f["tags.role"]) == {"final"}
    assert f.loc[f["tags.model"] == "xgboost", "v"].min() == pytest.approx(0.5)


def test_mean_std():
    # 0.5, 0.6, 0.7: mean 0.6, sample std 0.1.
    assert mean_std(pd.Series([0.5, 0.6, 0.7])) == "0.600 ± 0.100"
    assert mean_std(pd.Series([0.4])) == "0.400"


def test_tables_contain_expected_rows():
    out = results_tables(fake_runs(), k=20)
    assert "| Logistic regression | (a) local | 1 | 0.400 |" in out
    assert "| XGBoost | (b) local + aggregated | 3 | 0.600 ± 0.100 |" in out
    assert "0.99" not in out


def test_replace_between_markers():
    text = "intro\n<!-- r:start -->\nold\n<!-- r:end -->\noutro"
    assert replace_between_markers(text, "new", "r") == (
        "intro\n<!-- r:start -->\nnew\n<!-- r:end -->\noutro"
    )
    with pytest.raises(ValueError):
        replace_between_markers("no markers", "new", "r")


def test_comparison_table_uses_seed_means():
    rows = []
    for fs, vals in [("all", [0.5, 0.7]), ("all+graph", [0.8, 0.8]), ("local", [0.4, 0.4])]:
        for seed, v in enumerate(vals):
            rows.append(("xgboost", fs, "g", "final", str(seed), v))
    df = pd.DataFrame(
        rows,
        columns=["tags.model", "tags.feature_set", "tags.run_group", "tags.role", "tags.seed", "v"],
    )
    for m in ["test_pr_auc", "test_precision_at_20", "test_f1"]:
        df[f"metrics.{m}"] = df["v"]
    out = comparison_table(df, "xgboost", k=20)
    assert "| (b) local + aggregated | (b) + graph | 0.600 → 0.800 (+0.200) |" in out
    assert "| (a) local | (b) local + aggregated | 0.400 → 0.600 (+0.200) |" in out
    assert "(c) local + graph" not in out  # no runs for it: row skipped
