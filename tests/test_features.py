import pytest

from fraudgraph.features.tabular import FEATURE_SETS, NON_FEATURES, feature_columns, feature_matrix


@pytest.mark.parametrize("fs", FEATURE_SETS)
def test_no_identifier_label_or_time_in_features(data, fs):
    assert not NON_FEATURES & set(feature_columns(data, fs))


def test_feature_set_sizes(data):
    assert len(feature_columns(data, "local")) == 93
    assert len(feature_columns(data, "all")) == 93 + 72


def test_matrix_rows_align_with_nodes(data):
    X = feature_matrix(data, "all")
    assert X.shape == (len(data.nodes), 165)
    assert X[0, 0] == pytest.approx(data.nodes["local_1"].iloc[0])


def test_unknown_feature_set_rejected(data):
    with pytest.raises(ValueError):
        feature_columns(data, "graph")
