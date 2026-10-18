"""Temporal train / validation / test split, with a leakage check on the graph.

Split by time, not at random: the real task is "train on the past, flag the future",
and a random split would let the model learn from transactions that happen after the
ones it is tested on.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fraudgraph.config import load_config
from fraudgraph.data.load import UNKNOWN, edge_steps

Steps = tuple[int, int]  # inclusive (first, last) time step


class LeakageError(AssertionError):
    """An edge connects two different split periods."""


@dataclass(frozen=True)
class SplitConfig:
    train_steps: Steps
    val_steps: Steps
    test_steps: Steps

    def __post_init__(self) -> None:
        periods = [self.train_steps, self.val_steps, self.test_steps]
        for first, last in periods:
            if first > last:
                raise ValueError(f"Step range {first}-{last} starts after it ends")
        # Strictly chronological and non-overlapping: every train step comes before
        # every val step, which comes before every test step.
        for (_, earlier_last), (later_first, _) in zip(periods, periods[1:], strict=False):
            if later_first <= earlier_last:
                raise ValueError(f"Periods must be chronological and non-overlapping: {periods}")

    def periods(self) -> dict[str, Steps]:
        return {"train": self.train_steps, "val": self.val_steps, "test": self.test_steps}


def load_split_config(raw: dict[str, Any] | None = None) -> SplitConfig:
    raw = raw or load_config("split")
    return SplitConfig(**{k: tuple(v) for k, v in raw.items()})


@dataclass(frozen=True)
class TemporalSplit:
    """Boolean masks aligned with the rows of `nodes`.

    Each mask selects labelled transactions in that period: these are the nodes used
    for the loss and the metrics. Unknown-label nodes are in no mask, but they stay in
    `nodes` (and so in the graph), because GNNs still pass messages through them.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def assign_period(nodes: pd.DataFrame, cfg: SplitConfig) -> pd.Series:
    """Label every node, labelled or not, with 'train', 'val' or 'test' by its time step.

    Steps outside every period (none, with the project config) get NaN.
    """
    step = nodes["time_step"]
    period = pd.Series(np.nan, index=nodes.index, dtype=object)
    for name, (first, last) in cfg.periods().items():
        period[step.between(first, last)] = name
    return period


def temporal_split(nodes: pd.DataFrame, cfg: SplitConfig) -> TemporalSplit:
    lo, hi = int(nodes["time_step"].min()), int(nodes["time_step"].max())
    for name, (first, last) in cfg.periods().items():
        if first < lo or last > hi:
            raise ValueError(f"{name} steps {first}-{last} are outside the data's {lo}-{hi}")

    period = assign_period(nodes, cfg).to_numpy()
    labelled = (nodes["label"] != UNKNOWN).to_numpy()
    return TemporalSplit(
        train=labelled & (period == "train"),
        val=labelled & (period == "val"),
        test=labelled & (period == "test"),
    )


def assert_no_cross_split_edges(nodes: pd.DataFrame, edges: pd.DataFrame, cfg: SplitConfig) -> None:
    """Raise LeakageError if any edge links two different periods.

    If a train node were connected to a test node, a GNN would pass information
    across the split (and test features would leak into training). In Elliptic,
    edges never cross time steps, so this holds; it is checked rather than assumed.
    """
    period = assign_period(nodes, cfg)
    period.index = nodes["tx_id"]
    es = edge_steps(nodes, edges)  # also confirms both endpoints exist via the mapping
    src_p, dst_p = es["src"].map(period), es["dst"].map(period)
    crossing = src_p != dst_p
    if crossing.any():
        examples = es.loc[crossing, ["src", "dst", "src_step", "dst_step"]].head(5)
        raise LeakageError(
            f"{crossing.sum()} edges cross split periods, e.g.:\n{examples.to_string()}"
        )
