"""What the API and the dashboard serve, independent of HTTP (brief section 6).

Every method returns plain JSON-ready dicts, so the FastAPI app and the dashboard's
in-process client (the Space) show exactly the same thing.
"""

import math
from dataclasses import dataclass

import pandas as pd

from fraudgraph.serving.artifacts import ServingData
from fraudgraph.serving.graph import GraphStore, truncate

LABELS = {1: "illicit", 0: "licit", -1: "unknown"}


class NotFound(LookupError):
    """An unknown transaction, or a time step outside the served period."""


class InvalidRequest(ValueError):
    """A parameter outside the allowed range (D-040)."""


class Unavailable(RuntimeError):
    """A query this export cannot answer: the Space ships only part of the graph (D-040)."""


@dataclass(frozen=True)
class Limits:
    max_hops: int
    max_k: int
    max_nodes: int
    max_path_len: int
    max_paths: int


def _num(x) -> float | None:
    """A float for JSON; NaN (a step with no illicit transactions has no F1) becomes None."""
    x = float(x)
    return None if math.isnan(x) else x


class ServingService:
    def __init__(self, data: ServingData, store: GraphStore, limits: Limits):
        self.data, self.store, self.limits = data, store, limits

    # --- helpers ---------------------------------------------------------------------------

    @property
    def _max_hops(self) -> int:
        # The Space ships complete neighbourhoods only up to its export's hops (D-040).
        return min(self.limits.max_hops, int(self.data.meta.get("max_hops", self.limits.max_hops)))

    def _row(self, tx_id: int) -> pd.Series:
        try:
            return self.data.transactions.loc[tx_id]
        except KeyError:
            raise NotFound(f"Transaction {tx_id} is not in the served period") from None

    def _node(self, tx_id: int, row: pd.Series) -> dict:
        label = None
        if "label" in row.index:
            label = LABELS[int(row["label"])]
        return {
            "tx_id": int(tx_id),
            "time_step": int(row["time_step"]),
            "risk_score": float(row["risk_score"]),
            "flagged": bool(row["predicted_label"] == 1),
            "label": label,
        }

    def _check_step(self, time_step: int) -> None:
        first, last = self.data.meta["steps"]
        if not first <= time_step <= last:
            raise NotFound(f"Time step {time_step} is outside the served steps {first}-{last}")

    @staticmethod
    def _check_range(name: str, value: int, low: int, high: int) -> None:
        if not low <= value <= high:
            raise InvalidRequest(f"{name} must be between {low} and {high}, got {value}")

    def _need_full_graph(self, what: str) -> None:
        if not self.data.meta.get("has_full_graph", True):
            raise Unavailable(f"{what} needs the whole graph: run the full stack (Neo4j)")

    # --- queries ---------------------------------------------------------------------------

    def meta(self) -> dict:
        m = self.data.meta
        return {
            "model_version": m["model_version"],
            "threshold": float(m["threshold"]),
            "steps": list(m["steps"]),
            "backend": self.store.name,
            "demo": bool(m.get("demo", False)),
            "has_full_graph": bool(m.get("has_full_graph", True)),
            "labels_included": bool(m.get("labels_included", True)),
            "neighbourhood_roots": m.get("neighbourhood_roots"),
            "max_hops": self._max_hops,
            "max_k": self.limits.max_k,
            "max_nodes": self.limits.max_nodes,
        }

    def steps(self) -> dict:
        """Per-step counts from the served scores, plus the logged per-step test metrics."""
        tx = self.data.transactions
        counts = tx.groupby("time_step").agg(
            n_transactions=("risk_score", "size"), n_flagged=("predicted_label", "sum")
        )
        logged = self.data.per_step.set_index("time_step")
        items = []
        for step, c in counts.iterrows():
            item = {"time_step": int(step), "n_transactions": int(c["n_transactions"]),
                    "n_flagged": int(c["n_flagged"])}  # fmt: skip
            for col in ("f1", "pr_auc", "recall", "n_illicit"):
                known = col in logged.columns and step in logged.index
                item[col] = _num(logged.loc[step, col]) if known else None
            items.append(item)
        return {"steps": items}

    def transaction(self, tx_id: int) -> dict:
        row = self._row(tx_id)
        shap = self.data.shap.loc[[tx_id]] if tx_id in self.data.shap.index else self.data.shap[:0]
        return self._node(tx_id, row) | {
            "threshold": float(self.data.meta["threshold"]),
            "base_value": float(self.data.meta["base_value"]),
            "top_features": [
                {"feature": r["feature"], "group": r["group"], "shap": float(r["shap"])}
                for _, r in shap.iterrows()
            ],
        }

    def alerts(self, time_step: int, k: int) -> dict:
        self._check_range("k", k, 1, self.limits.max_k)
        self._check_step(time_step)
        tx = self.data.transactions
        t = tx[tx["time_step"] == time_step].reset_index()
        t = t.sort_values(["risk_score", "tx_id"], ascending=[False, True]).head(k)
        return {
            "time_step": time_step,
            "threshold": float(self.data.meta["threshold"]),
            "alerts": [self._node(r["tx_id"], r) for _, r in t.iterrows()],
        }

    def neighbourhood(self, tx_id: int, hops: int) -> dict:
        self._check_range("hops", hops, 1, self._max_hops)
        self._row(tx_id)
        roots = self.data.meta.get("neighbourhood_roots")
        if roots is not None and tx_id not in roots:
            raise Unavailable(
                "This demo ships neighbourhoods for the top alerts only: run the full stack"
            )
        dist = self.store.hop_distances(tx_id, hops)
        risk = self.data.transactions["risk_score"]
        kept, cut = truncate(dist, risk, self.limits.max_nodes)
        edges = self.store.edges_among(kept).sort_values(["src", "dst"])
        tx = self.data.transactions
        return {
            "tx_id": tx_id,
            "hops": hops,
            "truncated": cut,
            "nodes": [self._node(t, tx.loc[t]) | {"hop": dist[t]} for t in kept],
            "edges": [{"src": int(a), "dst": int(b)}
                      for a, b in zip(edges["src"], edges["dst"], strict=True)],
        }  # fmt: skip

    def paths(self, source: int, target: int) -> dict:
        self._need_full_graph("Paths")
        self._row(source)
        self._row(target)
        if source == target:
            found = [[source]]
        else:
            found = self.store.shortest_paths(
                source, target, self.limits.max_path_len, self.limits.max_paths
            )
        return {"source": source, "target": target,
                "paths": [[int(t) for t in p] for p in found]}  # fmt: skip

    def communities(self, time_step: int, min_size: int = 5, top: int = 10) -> dict:
        self._check_step(time_step)
        self._check_range("top", top, 1, self.limits.max_k)
        c = self.store.community_summary(time_step)
        c = c[c["size"] >= min_size].copy()
        c["share_flagged"] = c["n_flagged"] / c["size"]
        c = c.sort_values(["share_flagged", "n_flagged", "community_id"],
                          ascending=[False, False, True]).head(top)  # fmt: skip
        return {
            "time_step": time_step,
            "items": [
                {"community_id": int(r["community_id"]), "size": int(r["size"]),
                 "n_flagged": int(r["n_flagged"]), "share_flagged": float(r["share_flagged"]),
                 "mean_risk": float(r["mean_risk"])}
                for _, r in c.iterrows()
            ],
        }  # fmt: skip

    def fan_in(self, time_step: int, min_flagged: int = 2, top: int = 20) -> dict:
        self._need_full_graph("Fan-in")
        self._check_step(time_step)
        self._check_range("top", top, 1, self.limits.max_k)
        f = self.store.fan_in(time_step)
        f = f[f["n_flagged_payers"] >= min_flagged].copy()
        f["risk_score"] = f["tx_id"].map(self.data.transactions["risk_score"])
        f = f.sort_values(["n_flagged_payers", "risk_score", "tx_id"],
                          ascending=[False, False, True]).head(top)  # fmt: skip
        tx = self.data.transactions
        return {
            "time_step": time_step,
            "items": [
                self._node(r["tx_id"], tx.loc[r["tx_id"]])
                | {"n_flagged_payers": int(r["n_flagged_payers"])}
                for _, r in f.iterrows()
            ],
        }
