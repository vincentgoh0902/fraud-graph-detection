"""GNN rules that the evaluation depends on, checked on the synthetic dataset from conftest:
unknown nodes never enter the loss, scaling sees only the training period, no information
passes between periods, the kept checkpoint is the best validation epoch, and a refit with
the same seed is the same model (the protocol's refit check needs that)."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from fraudgraph.data.split import assign_period, temporal_split
from fraudgraph.evaluation.metrics import pr_auc
from fraudgraph.models.gnn import (
    GNN,
    build_graph,
    class_weights,
    node_loss,
    train_gnn,
)

TRAINING = {
    "max_epochs": 30,
    "patience": 10,
    "dropout": 0.5,
    "weight_decay": 5e-4,
    "focal_gamma": 2.0,
}
PARAMS = {"hidden": 8, "lr": 0.01, "loss": "weighted_ce"}


@pytest.fixture
def graph(data, split_cfg):
    return build_graph(data, "local", split_cfg)


@pytest.fixture
def split(data, split_cfg):
    return temporal_split(data.nodes, split_cfg)


def test_class_weights_are_balanced():
    # 1 illicit, 3 licit: "balanced" weights n / (2 * n_class) = 4/6 licit, 4/2 illicit.
    w = class_weights(torch.tensor([1, 0, 0, 0]))
    assert w.tolist() == pytest.approx([4 / 6, 2.0])


def test_focal_with_gamma_zero_is_weighted_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(20)
    y = torch.randint(0, 2, (20,))
    mask = torch.ones(20, dtype=torch.bool)
    w = torch.tensor([0.6, 3.0])
    expected = F.binary_cross_entropy_with_logits(logits, y.float(), weight=w[y])
    assert node_loss(logits, y, mask, w, gamma=0.0) == pytest.approx(expected.item())


def test_focal_downweights_easy_examples():
    # A confident correct prediction loses almost all of its loss under focal (gamma 2).
    logits, y = torch.tensor([6.0]), torch.tensor([1])
    mask, w = torch.tensor([True]), torch.tensor([1.0, 1.0])
    assert node_loss(logits, y, mask, w, gamma=2.0) < 0.01 * node_loss(
        logits, y, mask, w, gamma=0.0
    )


def test_nodes_outside_the_mask_have_no_effect_on_the_loss():
    torch.manual_seed(0)
    logits = torch.randn(10)
    y = torch.tensor([1, 0, 0, -1, -1, 1, 0, -1, 0, 0])
    mask = y >= 0
    w = torch.tensor([1.0, 2.0])
    base = node_loss(logits, y, mask, w, gamma=0.0)
    changed = logits.clone()
    changed[~mask] = 100.0  # wild predictions on unknown nodes
    assert node_loss(changed, y, mask, w, gamma=0.0) == pytest.approx(base.item())


def test_scaling_uses_training_period_only(data, split_cfg):
    period = assign_period(data.nodes, split_cfg).to_numpy()
    shifted = data.nodes.copy()
    later = period != "train"
    shifted.loc[later, data.local_cols] = shifted.loc[later, data.local_cols] * 100 + 7
    a = build_graph(data, "local", split_cfg)
    b = build_graph(type(data)(shifted, data.edges, data.local_cols, data.agg_cols), "local",
                    split_cfg)  # fmt: skip
    train = torch.from_numpy(period == "train")
    assert torch.equal(a.x[train], b.x[train])
    # Training-period features are standardised: mean ~0, std ~1 per column.
    assert a.x[train].mean(0).abs().max() < 1e-4


def test_graph_is_undirected_and_stays_within_periods(data, split_cfg, graph):
    src, dst = graph.edge_index
    pairs = set(zip(src.tolist(), dst.tolist(), strict=True))
    assert all((d, s) in pairs for s, d in pairs)
    assert len(pairs) == 2 * len(data.edges)  # the synthetic chain has no reciprocal edges
    period = assign_period(data.nodes, split_cfg).to_numpy()
    assert (period[src.numpy()] == period[dst.numpy()]).all()


@pytest.mark.parametrize("kind", ["gcn", "sage"])
def test_no_information_passes_between_periods(data, split_cfg, graph, kind):
    """Scrambling test-period features must not change any train/val node's output."""
    torch.manual_seed(0)
    model = GNN(kind, graph.x.shape[1], hidden=8, dropout=0.5).eval()
    period = assign_period(data.nodes, split_cfg).to_numpy()
    test = torch.from_numpy(period == "test")
    x2 = graph.x.clone()
    x2[test] = torch.randn(int(test.sum()), graph.x.shape[1]) * 50
    with torch.no_grad():
        out1, out2 = model(graph.x, graph.edge_index), model(x2, graph.edge_index)
    assert torch.equal(out1[~test], out2[~test])
    assert not torch.equal(out1[test], out2[test])  # the scramble did reach the test nodes


@pytest.mark.parametrize("kind", ["gcn", "sage"])
def test_kept_checkpoint_is_the_best_validation_epoch(graph, split, kind):
    r = train_gnn(kind, graph, split, PARAMS, seed=0, training=TRAINING)
    best = int(np.argmax(r.val_history))  # first epoch with the highest val PR-AUC
    assert r.best_epoch == best
    y_val = graph.y[torch.from_numpy(split.val)].numpy()
    assert pr_auc(y_val, r.scores()[split.val]) == pytest.approx(r.val_history[best])


@pytest.mark.parametrize("loss", ["weighted_ce", "focal"])
def test_refit_with_same_seed_is_identical(graph, split, loss):
    params = PARAMS | {"loss": loss}
    a = train_gnn("sage", graph, split, params, seed=3, training=TRAINING).scores()
    b = train_gnn("sage", graph, split, params, seed=3, training=TRAINING).scores()
    c = train_gnn("sage", graph, split, params, seed=4, training=TRAINING).scores()
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_embeddings_have_one_row_per_node(graph, split):
    r = train_gnn("gcn", graph, split, PARAMS, seed=0, training=TRAINING)
    emb = r.embeddings()
    assert emb.shape == (graph.x.shape[0], PARAMS["hidden"])
    assert np.isfinite(emb).all()


@pytest.mark.parametrize("kind", ["gcn", "sage"])
def test_per_period_scores_equal_whole_graph_scores(graph, split, kind):
    """Running each period's subgraph separately is only a speed-up: it must give the
    same scores as running the whole graph, because no edge crosses periods."""
    r = train_gnn(kind, graph, split, PARAMS, seed=0, training=TRAINING)
    r.model.eval()
    with torch.no_grad():
        whole = torch.sigmoid(r.model(graph.x, graph.edge_index)).numpy()
    np.testing.assert_allclose(r.scores(), whole, rtol=1e-5, atol=1e-6)
