"""TreeSHAP via XGBoost (D-034) and feature groups."""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from fraudgraph.analysis.explain import (
    feature_group,
    group_shares,
    mean_abs_shap,
    tree_shap,
)


@pytest.fixture(scope="module")
def toy_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3)).astype(np.float32)
    y = (X[:, 0] + 0.2 * rng.normal(size=500) > 0).astype(int)
    model = XGBClassifier(n_estimators=30, max_depth=3, random_state=0).fit(X, y)
    return model, X


def test_shap_values_add_up_to_the_raw_output(toy_model):
    model, X = toy_model
    contribs, bias = tree_shap(model, X)
    assert contribs.shape == X.shape
    margin = model.predict(X, output_margin=True)
    np.testing.assert_allclose(contribs.sum(axis=1) + bias, margin, atol=1e-4)


def test_shap_puts_the_signal_on_the_informative_feature(toy_model):
    model, X = toy_model
    contribs, _ = tree_shap(model, X)
    imp = mean_abs_shap(contribs, ["f0", "f1", "f2"])
    assert imp.index[0] == "f0"


@pytest.mark.parametrize(
    ("col", "group"),
    [
        ("local_1", "local"),
        ("agg_72", "aggregated"),
        ("pagerank", "structure"),
        ("community_size", "structure"),
        ("nbr_risk_max_all", "risk"),
        ("community_risk_mean_all", "risk"),
    ],
)
def test_feature_group(col, group):
    assert feature_group(col) == group


def test_feature_group_rejects_unknown_columns():
    with pytest.raises(ValueError):
        feature_group("time_step")


def test_group_shares_sum_to_one():
    contribs = np.array([[1.0, -1.0, 2.0], [-1.0, 1.0, 0.0]])
    shares = group_shares(contribs, ["local_1", "local_2", "pagerank"])
    assert shares.sum() == pytest.approx(1.0)
    # mean |SHAP|: local_1 1, local_2 1, pagerank 1 -> local 2/3, structure 1/3
    assert shares["local"] == pytest.approx(2 / 3)
    assert isinstance(shares, pd.Series)
