"""Load the raw Elliptic CSVs into two clean tables, and validate them.

nodes: one row per transaction
    tx_id, time_step, label (1 illicit, 0 licit, -1 unknown),
    local_1..local_93, agg_1..agg_72
edges: one row per flow of bitcoin, src -> dst (both tx_ids)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudgraph.config import load_config, resolve

ILLICIT, LICIT, UNKNOWN = 1, 0, -1

# The raw file encodes illicit as "1" and licit as "2". Mapping "2" -> 0 here, once,
# means illicit is the positive class everywhere downstream. Getting this backwards
# silently inverts every metric.
RAW_LABEL_MAP = {"1": ILLICIT, "2": LICIT, "unknown": UNKNOWN}


class DataValidationError(ValueError):
    """The data does not look like the Elliptic dataset we expect."""


@dataclass(frozen=True)
class EllipticData:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    local_cols: list[str]
    agg_cols: list[str]

    @property
    def feature_cols(self) -> list[str]:
        # time_step is deliberately not a feature (D-013).
        return self.local_cols + self.agg_cols


def feature_names(n_local: int, n_agg: int) -> tuple[list[str], list[str]]:
    # The CSV has no header and Elliptic does not publish feature names,
    # so columns are named by position (D-003).
    return (
        [f"local_{i}" for i in range(1, n_local + 1)],
        [f"agg_{i}" for i in range(1, n_agg + 1)],
    )


def read_raw(
    features_path: Path, classes_path: Path, edges_path: Path, n_local: int, n_agg: int
) -> EllipticData:
    local_cols, agg_cols = feature_names(n_local, n_agg)
    features = pd.read_csv(
        features_path,
        header=None,
        names=["tx_id", "time_step", *local_cols, *agg_cols],
        # float32 halves memory (~135 MB instead of ~270 MB) and is what XGBoost
        # and PyTorch use internally anyway.
        dtype={c: np.float32 for c in local_cols + agg_cols}
        | {"tx_id": np.int64, "time_step": np.int16},
        engine="pyarrow",  # multi-threaded; several times faster on the 690 MB file
    ).copy()  # read_csv stores each column separately; copy() packs them into one block

    # Read labels as strings so "1"/"2"/"unknown" are compared exactly.
    classes = pd.read_csv(classes_path, dtype={"txId": np.int64, "class": str})
    unexpected = set(classes["class"]) - set(RAW_LABEL_MAP)
    if unexpected:
        raise DataValidationError(f"Unexpected label values: {sorted(unexpected)}")
    classes = classes.rename(columns={"txId": "tx_id"})
    classes["label"] = classes["class"].map(RAW_LABEL_MAP).astype(np.int8)

    # validate="one_to_one" raises if a tx_id appears twice in either file.
    try:
        nodes = features.merge(
            classes[["tx_id", "label"]],
            on="tx_id",
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
    except pd.errors.MergeError as e:
        raise DataValidationError(f"Duplicate tx_id: {e}") from e
    if (nodes["_merge"] != "both").any():
        raise DataValidationError("Features and classes files cover different transactions")
    nodes = nodes.drop(columns="_merge")
    nodes["label"] = nodes["label"].astype(np.int8)
    nodes = nodes[["tx_id", "time_step", "label", *local_cols, *agg_cols]]

    edges = pd.read_csv(edges_path, dtype=np.int64).rename(columns={"txId1": "src", "txId2": "dst"})
    return EllipticData(nodes=nodes, edges=edges, local_cols=local_cols, agg_cols=agg_cols)


def edge_steps(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Return edges with the time step of each endpoint (src_step, dst_step)."""
    step = nodes.set_index("tx_id")["time_step"]
    return edges.assign(src_step=edges["src"].map(step), dst_step=edges["dst"].map(step))


def validate(data: EllipticData, expected: dict[str, Any] | None = None) -> None:
    """Raise DataValidationError if the data breaks any assumption the project relies on."""
    nodes, edges = data.nodes, data.edges

    if nodes["tx_id"].duplicated().any():
        raise DataValidationError("Duplicate tx_id in nodes")
    if nodes[data.feature_cols].isna().any().any():
        raise DataValidationError("Missing feature values")

    known = set(nodes["tx_id"])
    dangling = ~edges["src"].isin(known) | ~edges["dst"].isin(known)
    if dangling.any():
        raise DataValidationError(f"{dangling.sum()} edges point to unknown transactions")

    # The whole evaluation design (D-001) depends on this: no edge crosses time steps,
    # so train-period and test-period graphs are disjoint. Checked, not assumed.
    es = edge_steps(nodes, edges)
    cross = es["src_step"] != es["dst_step"]
    if cross.any():
        raise DataValidationError(f"{cross.sum()} edges connect different time steps")

    if expected:
        actual = {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_illicit": int((nodes["label"] == ILLICIT).sum()),
            "n_licit": int((nodes["label"] == LICIT).sum()),
            "n_time_steps": int(nodes["time_step"].nunique()),
            "n_local_features": len(data.local_cols),
            "n_agg_features": len(data.agg_cols),
        }
        wrong = {k: (actual[k], v) for k, v in expected.items() if actual.get(k) != v}
        if wrong:
            raise DataValidationError(
                f"Counts differ from published figures (actual, expected): {wrong}"
            )


def load_elliptic(cfg: dict[str, Any] | None = None, use_cache: bool = True) -> EllipticData:
    """Load and validate the dataset, caching it as Parquet after the first read.

    Reading the 690 MB CSV takes ~30 s; the Parquet cache loads in a few seconds.
    """
    cfg = cfg or load_config("data")
    exp = cfg["expected"]
    n_local, n_agg = exp["n_local_features"], exp["n_agg_features"]
    processed = resolve(cfg["processed_dir"])
    nodes_pq, edges_pq = processed / "nodes.parquet", processed / "edges.parquet"

    if use_cache and nodes_pq.exists() and edges_pq.exists():
        local_cols, agg_cols = feature_names(n_local, n_agg)
        data = EllipticData(
            pd.read_parquet(nodes_pq), pd.read_parquet(edges_pq), local_cols, agg_cols
        )
    else:
        raw = resolve(cfg["raw_dir"])
        files = cfg["files"]
        data = read_raw(
            raw / files["features"], raw / files["classes"], raw / files["edges"], n_local, n_agg
        )
        if use_cache:
            processed.mkdir(parents=True, exist_ok=True)
            data.nodes.to_parquet(nodes_pq, index=False)
            data.edges.to_parquet(edges_pq, index=False)

    # Validate on every load, cached or not: it takes about a second and catches a
    # stale or corrupted cache as well as bad raw files.
    validate(data, exp)
    return data
