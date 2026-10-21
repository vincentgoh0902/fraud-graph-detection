"""The serving artefacts: what scripts/export_serving.py writes and the app reads (D-038).

- transactions.parquet: one row per served transaction (score, flag, label, community)
- shap.parquet: the top SHAP contributions per transaction, long format
- edges.parquet: payment edges (all test-period edges, or the Space subset, D-040)
- per_step.csv: the served model's logged test metrics per step, for the dashboard
- meta.json: model version, threshold, SHAP base value and what this export contains
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FILES = {
    "transactions": "transactions.parquet",
    "shap": "shap.parquet",
    "edges": "edges.parquet",
    "per_step": "per_step.csv",
    "meta": "meta.json",
}


@dataclass
class ServingData:
    transactions: pd.DataFrame  # indexed by tx_id
    shap: pd.DataFrame  # indexed by tx_id, rows in rank order
    edges: pd.DataFrame  # src, dst
    meta: dict
    per_step: pd.DataFrame


def write_artifacts(
    out_dir: Path,
    transactions: pd.DataFrame,
    shap: pd.DataFrame,
    edges: pd.DataFrame,
    meta: dict,
    per_step: pd.DataFrame,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transactions.to_parquet(out_dir / FILES["transactions"], index=False)
    shap.to_parquet(out_dir / FILES["shap"], index=False)
    edges.to_parquet(out_dir / FILES["edges"], index=False)
    per_step.to_csv(out_dir / FILES["per_step"], index=False)
    (out_dir / FILES["meta"]).write_text(json.dumps(meta, indent=2))


def load_artifacts(in_dir: Path) -> ServingData:
    in_dir = Path(in_dir)
    tx = pd.read_parquet(in_dir / FILES["transactions"]).set_index("tx_id").sort_index()
    shap = pd.read_parquet(in_dir / FILES["shap"]).sort_values(["tx_id", "rank"])
    return ServingData(
        transactions=tx,
        shap=shap.set_index("tx_id"),
        edges=pd.read_parquet(in_dir / FILES["edges"]),
        meta=json.loads((in_dir / FILES["meta"]).read_text()),
        per_step=pd.read_csv(in_dir / FILES["per_step"]),
    )
