"""Config inheritance, and the D-014 rule that Phase 3 gets the same tuning budget."""

from fraudgraph.config import load_experiment_config


def test_graph_experiment_reuses_the_baseline_budget():
    graph, base = load_experiment_config("graph_features"), load_experiment_config("baselines")
    assert graph["models"] == {"xgboost": base["models"]["xgboost"]}  # D-014, D-022
    assert graph["seeds"] == base["seeds"]
    assert graph["evaluation"] == base["evaluation"]


def test_graph_experiment_overrides():
    graph = load_experiment_config("graph_features")
    assert graph["experiment"] == "graph_features"
    assert graph["feature_sets"] == ["local+graph", "all+graph", "local+structure"]
    assert "base" not in graph
