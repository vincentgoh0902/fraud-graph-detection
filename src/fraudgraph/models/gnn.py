"""Graph neural networks: a 2-layer GCN and a 2-layer GraphSAGE (Phase 4).

Training is full-batch: every node, labelled or not, takes part in message passing; only
labelled training nodes enter the loss. Edges never cross time steps, so each period's
nodes form a graph of their own. The model is run on one period's subgraph at a time
(training on the training period, scoring validation on the validation period), which
gives the same result as the whole graph with less work, and makes "no information passes
between periods" true by construction (also checked on the whole graph in tests).

Choices, with their decision-log entries:
- CPU, with deterministic algorithms: a refit must reproduce its validation score exactly
  for the protocol's refit check (D-025).
- Edges are made undirected; features are standardised on training-period nodes (D-026).
- The kept checkpoint is the epoch with the best validation PR-AUC, never the last (D-027).
- Imbalance: class-weighted cross-entropy or focal loss, chosen by the grid (D-029).
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import subgraph, to_undirected

from fraudgraph.data.load import EllipticData
from fraudgraph.data.split import (
    SplitConfig,
    TemporalSplit,
    assert_no_cross_split_edges,
    assign_period,
    temporal_split,
)
from fraudgraph.evaluation.metrics import pr_auc
from fraudgraph.evaluation.protocol import EvalConfig, FitFn, ProtocolResult, run_protocol
from fraudgraph.features.tabular import feature_matrix

GNN_MODELS = ("gcn", "sage")
LOSSES = ("weighted_ce", "focal")
PERIODS = ("train", "val", "test")


@dataclass(frozen=True)
class GraphTensors:
    x: torch.Tensor  # float32 [n_nodes, n_features], standardised
    edge_index: torch.Tensor  # int64 [2, n_edges], both directions
    y: torch.Tensor  # int64 [n_nodes]: 1 illicit, 0 licit, -1 unknown
    period: np.ndarray  # "train" / "val" / "test" per node, by time step

    def period_graph(self, period: str) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Node indices, features and (relabelled) edges of one period's subgraph."""
        nodes = np.flatnonzero(self.period == period)
        idx = torch.from_numpy(nodes)
        edges, _ = subgraph(idx, self.edge_index, relabel_nodes=True, num_nodes=len(self.y))
        return nodes, self.x[idx], edges


def build_graph(data: EllipticData, feature_set: str, split_cfg: SplitConfig) -> GraphTensors:
    """Node features, undirected edges and labels as tensors, rows aligned with data.nodes."""
    X = feature_matrix(data, feature_set)
    # Standardise with statistics from training-period nodes only (labelled or not: no
    # label is used), so nothing about validation or test shapes the inputs (D-010).
    period = assign_period(data.nodes, split_cfg).to_numpy()
    if not np.isin(period, PERIODS).all():
        raise ValueError("Every time step must belong to a period")
    train_period = period == "train"
    mean = X[train_period].mean(axis=0)
    std = X[train_period].std(axis=0)
    std[std == 0] = 1.0  # a constant column stays constant instead of dividing by zero
    X = (X - mean) / std

    row = pd.Series(np.arange(len(data.nodes)), index=data.nodes["tx_id"])
    src = row.loc[data.edges["src"]].to_numpy()
    dst = row.loc[data.edges["dst"]].to_numpy()
    # Undirected: illicit funds are suspicious in both directions, payer and payee (D-026).
    # to_undirected also removes duplicates when a pair already appears both ways.
    edge_index = to_undirected(torch.from_numpy(np.stack([src, dst])))
    return GraphTensors(
        x=torch.from_numpy(X.astype(np.float32)),
        edge_index=edge_index,
        y=torch.from_numpy(data.nodes["label"].to_numpy().astype(np.int64)),
        period=period.astype(str),
    )


def class_weights(y_train: torch.Tensor) -> torch.Tensor:
    """[licit weight, illicit weight] = n / (2 * n_class), as class_weight="balanced"."""
    counts = torch.bincount(y_train, minlength=2).float()
    return len(y_train) / (2 * counts)


def node_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Class-weighted binary cross-entropy over masked nodes, with optional focal term.

    Focal loss multiplies each node's loss by (1 - p_t)^gamma, where p_t is the predicted
    probability of the true class, so confidently correct nodes contribute little and
    training concentrates on hard ones. gamma = 0 is plain weighted cross-entropy.
    Nodes outside the mask (unknown labels, other periods) contribute nothing.
    """
    z, t = logits[mask], y[mask]
    ce = F.binary_cross_entropy_with_logits(z, t.float(), reduction="none")
    if gamma:
        p_t = torch.exp(-ce)  # ce = -log(p_t)
        ce = (1 - p_t) ** gamma * ce
    return (weights[t] * ce).mean()


class GNN(nn.Module):
    """Two message-passing layers, then a linear head giving one illicit logit per node.

    The second layer's output (after ReLU) is the node embedding the hybrid uses: each
    node's features mixed with its neighbours' up to two hops away.
    """

    def __init__(self, kind: str, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        if kind == "gcn":
            # Not cached: the model sees a different subgraph per period, and a cached
            # normalisation would silently reuse the first one's.
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        elif kind == "sage":
            # Mean over neighbours, plus a separate weight for the node's own features
            # (root weight), so a node is not averaged away by many neighbours.
            self.conv1 = SAGEConv(in_dim, hidden, aggr="mean")
            self.conv2 = SAGEConv(hidden, hidden, aggr="mean")
        else:
            raise ValueError(f"Unknown GNN {kind!r}; expected one of {GNN_MODELS}")
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, self.dropout, self.training)
        return F.relu(self.conv2(h, edge_index))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(self.embed(x, edge_index), self.dropout, self.training)
        return self.head(h).squeeze(-1)


@dataclass
class TrainResult:
    """A trained GNN restored to its best validation epoch."""

    model: GNN
    graph: GraphTensors
    best_epoch: int
    val_history: list[float]  # validation PR-AUC after each epoch
    train_pr_auc: float  # on labelled training nodes, at the kept checkpoint

    def _per_period(self, fn) -> np.ndarray:
        """fn(x, edges) on each period's subgraph, stitched back into node order."""
        self.model.eval()
        parts = {}
        with torch.no_grad():
            for p in PERIODS:
                nodes, x, edges = self.graph.period_graph(p)
                parts[p] = (nodes, fn(x, edges).numpy())
        width = next(iter(parts.values()))[1].shape[1:]
        out = np.empty((len(self.graph.y), *width), dtype=np.float32)
        for nodes, values in parts.values():
            out[nodes] = values
        return out

    def scores(self) -> np.ndarray:
        """Predicted illicit probability for every node."""
        return self._per_period(lambda x, e: torch.sigmoid(self.model(x, e)))

    def embeddings(self) -> np.ndarray:
        return self._per_period(self.model.embed)


def train_gnn(
    kind: str,
    graph: GraphTensors,
    split: TemporalSplit,
    params: dict[str, Any],
    seed: int,
    training: dict[str, Any],
) -> TrainResult:
    """Train with early stopping on validation PR-AUC; return the best epoch's model.

    params come from the grid (hidden, lr, loss); training holds the fixed settings.
    """
    if params["loss"] not in LOSSES:
        raise ValueError(f"Unknown loss {params['loss']!r}; expected one of {LOSSES}")
    # Same seed -> same initial weights and dropout masks; deterministic kernels -> same
    # arithmetic. Together a refit reproduces the model exactly (D-025).
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    model = GNN(kind, graph.x.shape[1], params["hidden"], training["dropout"])
    opt = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=training["weight_decay"]
    )
    # Training-period subgraph; the loss mask picks its labelled nodes.
    tr_nodes, tr_x, tr_edges = graph.period_graph("train")
    tr_y = graph.y[torch.from_numpy(tr_nodes)]
    tr_mask = torch.from_numpy(split.train[tr_nodes])
    weights = class_weights(tr_y[tr_mask])
    gamma = training["focal_gamma"] if params["loss"] == "focal" else 0.0
    va_nodes, va_x, va_edges = graph.period_graph("val")
    va_mask = split.val[va_nodes]
    y_val = graph.y[torch.from_numpy(va_nodes[va_mask])].numpy()

    history: list[float] = []
    best_val, best_epoch, best_state = -np.inf, 0, None
    for epoch in range(training["max_epochs"]):
        model.train()
        opt.zero_grad()
        loss = node_loss(model(tr_x, tr_edges), tr_y, tr_mask, weights, gamma)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_score = torch.sigmoid(model(va_x, va_edges))[va_mask].numpy()
        history.append(pr_auc(y_val, val_score))
        # Strict ">" keeps the earliest epoch on a tie, as the grid keeps the earliest config.
        if history[-1] > best_val:
            best_val, best_epoch = history[-1], epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= training["patience"]:
            break  # no improvement for `patience` epochs

    model.load_state_dict(best_state)
    result = TrainResult(model, graph, best_epoch, history, train_pr_auc=float("nan"))
    result.train_pr_auc = pr_auc(tr_y[tr_mask].numpy(), result.scores()[split.train])
    return result


class GNNScorer:
    """The protocol's scorer for one trained GNN: scores of a period's labelled nodes."""

    def __init__(self, result: TrainResult, split: TemporalSplit):
        self.result = result
        self.split = split
        self._scores = result.scores()

    def __call__(self, period: str) -> np.ndarray:
        return self._scores[getattr(self.split, period)]


def gnn_fit_fn(
    kind: str, graph: GraphTensors, split: TemporalSplit, training: dict[str, Any]
) -> FitFn:
    def fit(params, seed):
        result = train_gnn(kind, graph, split, params, seed, training)
        # The protocol calls fit inside an MLflow run (candidate or final), so these land
        # on that run: where the checkpoint came from, and the training-node PR-AUC for the
        # embedding skew check (D-028).
        if mlflow.active_run() is not None:
            mlflow.log_params({f"train_{k}": v for k, v in training.items()})
            mlflow.log_metrics(
                {
                    "best_epoch": result.best_epoch,
                    "epochs_run": len(result.val_history),
                    "train_pr_auc": result.train_pr_auc,
                }
            )
        return GNNScorer(result, split)

    return fit


def save_embeddings(tx_ids: np.ndarray, embeddings_dir: Path):
    """An on_final hook: write the kept model's embeddings, named by the final run's id.

    Embeddings are ~100 MB per run, too big for the MLflow store; the run gets a tag
    pointing at the file instead, so the hybrid can find exactly this model's output.
    """

    def on_final(scorer: GNNScorer, seed: int) -> None:
        emb = scorer.result.embeddings()
        table = pd.DataFrame(emb, columns=[f"emb_{i + 1}" for i in range(emb.shape[1])])
        table.insert(0, "tx_id", tx_ids)
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        path = embeddings_dir / f"{mlflow.active_run().info.run_id}.parquet"
        table.to_parquet(path, index=False)
        mlflow.set_tag("embeddings_file", path.name)

    return on_final


def run_gnns(
    data: EllipticData,
    split_cfg: SplitConfig,
    cfg: dict[str, Any],
    run_group: str,
    embeddings_dir: Path,
    models: list[str] | None = None,
    feature_sets: list[str] | None = None,
) -> dict[tuple[str, str], ProtocolResult]:
    """Run every (feature set, GNN) pair through the protocol. MLflow must be set up."""
    assert_no_cross_split_edges(data.nodes, data.edges, split_cfg)
    split = temporal_split(data.nodes, split_cfg)
    y = data.nodes["label"].to_numpy().astype(np.int64)
    steps = data.nodes["time_step"].to_numpy()
    tx_ids = data.nodes["tx_id"].to_numpy()
    eval_cfg = EvalConfig(**{**cfg["evaluation"], "extra_ks": tuple(cfg["evaluation"]["extra_ks"])})

    results = {}
    for fs in feature_sets or cfg["feature_sets"]:
        graph = build_graph(data, fs, split_cfg)
        for name in models or list(cfg["models"]):
            tags = {"phase": str(cfg["phase"]), "model": name, "feature_set": fs,
                    "run_group": run_group}  # fmt: skip
            results[(name, fs)] = run_protocol(
                gnn_fit_fn(name, graph, split, cfg["training"]),
                cfg["models"][name]["grid"],
                cfg["seeds"],
                y,
                steps,
                split,
                eval_cfg,
                tags,
                ids=tx_ids,
                on_final=save_embeddings(tx_ids, embeddings_dir),
            )
    return results
