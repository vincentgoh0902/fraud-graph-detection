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


def test_interval_table_formats_changes_and_skips_missing():
    from fraudgraph.evaluation.report import interval_table

    r = {
        "pr_auc_change": 0.004,
        "pr_auc_lo": -0.0001,
        "pr_auc_hi": 0.01,
        "precision_at_k_change": 0.009,
        "precision_at_k_lo": -0.002,
        "precision_at_k_hi": 0.02,
        "share_positive": 0.9,
    }
    out = interval_table({("all", "all+graph"): r}, k=20)
    assert "| (b) local + aggregated | (b) + graph | +0.004 [+0.000, +0.010] |" in out
    assert "| 90% |" in out
    assert "(c) local + graph" not in out


def gnn_runs():
    """Candidates and finals for one GNN: config 0 weighted CE, config 1 focal."""
    rows = []
    for config, loss, vals in [("0", "weighted_ce", [0.70, 0.72]), ("1", "focal", [0.80, 0.78])]:
        for seed, v in enumerate(vals):
            rows.append(("gcn", "all", "g", "candidate", config, str(seed), loss, v, 0.9, 12.0))
    for seed, v in enumerate([0.80, 0.78]):
        rows.append(("gcn", "all", "g", "final", "1", str(seed), "focal", v, 0.95, 40.0 + seed))
    cols = ["tags.model", "tags.feature_set", "tags.run_group", "tags.role", "tags.config_id",
            "tags.seed", "params.loss", "metrics.val_pr_auc", "metrics.train_pr_auc",
            "metrics.best_epoch"]  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


def test_loss_table_compares_best_config_per_loss_on_validation():
    from fraudgraph.evaluation.report import loss_table

    out = loss_table(gnn_runs())
    assert "| GCN | (b) local + aggregated | 0.710 | 0.790 | focal |" in out


def test_checkpoint_table_lists_epochs_and_train_vs_val():
    from fraudgraph.evaluation.report import checkpoint_table

    out = checkpoint_table(latest_finals(gnn_runs()))
    assert "| GCN | (b) local + aggregated | 2 | 40, 41 | 0.950 ± 0.000 | 0.790 ± 0.014 |" in out


def test_pair_interval_table_uses_given_labels():
    from fraudgraph.evaluation.report import pair_interval_table

    r = {"pr_auc_change": -0.02, "pr_auc_lo": -0.05, "pr_auc_hi": 0.01,
         "precision_at_k_change": 0.0, "precision_at_k_lo": -0.01, "precision_at_k_hi": 0.01,
         "share_positive": 0.1}  # fmt: skip
    out = pair_interval_table([("XGBoost, (b)", "GCN, (b)", r)], k=20)
    assert "| XGBoost, (b) | GCN, (b) | -0.020 [-0.050, +0.010] |" in out
