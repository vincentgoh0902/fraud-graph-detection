"""Build-time check of the site's JSON (D-012, D-040, D-044). The deploy fails if it fails.

An allow-list, not a search for bad names: every object must have exactly the keys its file
type allows. A label value, a feature column (`local_1` as a key), an unknown field, an
unexpected file, or an edge outside the shipped subset all fail. Feature *names* may appear
as SHAP labels; feature *values* never do.
"""

import re

import pandas as pd


class SiteCheckError(ValueError):
    pass


NODE = {"tx_id", "time_step", "risk_score", "flagged", "label"}
META = {
    "model_version", "threshold", "steps", "backend", "demo", "has_full_graph",
    "labels_included", "neighbourhood_roots", "max_hops", "max_k", "max_nodes",
    "dataset_repo", "dataset_revision", "base_value", "features", "groups", "alerts_k",
    "shards_per_step",
}  # fmt: skip
STEP_SUMMARY = {"time_step", "n_transactions", "n_flagged", "f1", "pr_auc", "recall", "n_illicit"}
COMMUNITY = {"community_id", "size", "n_flagged", "share_flagged", "mean_risk"}
NEIGHBOURHOOD = {"tx_id", "hops", "truncated", "nodes", "edges", "positions"}
SHARD_ENTRY = {"tx_id", "risk_score", "flagged", "top"}

FILES = {
    "index": re.compile(r"data/index\.json"),
    "lookup": re.compile(r"data/lookup\.json"),
    "step": re.compile(r"data/steps/\d+\.json"),
    "shard": re.compile(r"data/shap/\d+-\d+\.json"),
}


def _keys(obj: dict, allowed: set, where: str) -> None:
    if not isinstance(obj, dict) or set(obj) != allowed:
        got = set(obj) if isinstance(obj, dict) else type(obj).__name__
        extra = sorted(set(got) - allowed) if isinstance(got, set) else got
        missing = sorted(allowed - set(got)) if isinstance(got, set) else []
        raise SiteCheckError(f"{where}: unexpected keys {extra}, missing {missing}")


def _node(n: dict, where: str, extra: set = frozenset()) -> None:
    _keys(n, NODE | extra, where)
    if n["label"] is not None:
        raise SiteCheckError(f"{where}: a label value is present ({n['label']!r})")


def check_site(payloads: dict[str, dict], shipped_edges: pd.DataFrame) -> None:
    edges = set(zip(shipped_edges["src"].astype(int), shipped_edges["dst"].astype(int),
                    strict=True))  # fmt: skip
    for path, p in payloads.items():
        kind = next((k for k, rx in FILES.items() if rx.fullmatch(path)), None)
        if kind is None:
            raise SiteCheckError(f"unexpected file {path}")
        if kind == "index":
            _keys(p, {"meta", "steps", "tour"}, path)
            _keys(p["meta"], META, f"{path} meta")
            if p["meta"]["labels_included"] or not p["meta"]["demo"]:
                raise SiteCheckError(f"{path}: not a label-free demo export (label)")
            for s in p["steps"]:
                _keys(s, STEP_SUMMARY, f"{path} steps")
            _keys(p["tour"], {"before", "after", "alert"}, f"{path} tour")
            _node(p["tour"]["alert"], f"{path} tour alert")
        elif kind == "lookup":
            if not all(k.isdigit() and all(isinstance(t, int) for t in v) for k, v in p.items()):
                raise SiteCheckError(f"{path}: expected step -> list of tx_id")
        elif kind == "step":
            _keys(p, {"time_step", "alerts", "communities", "neighbourhoods"}, path)
            for a in p["alerts"]:
                _node(a, f"{path} alerts")
            _keys(p["communities"], {"time_step", "items"}, f"{path} communities")
            for c in p["communities"]["items"]:
                _keys(c, COMMUNITY, f"{path} communities")
            for root, by_hops in p["neighbourhoods"].items():
                for hops, nb in by_hops.items():
                    where = f"{path} neighbourhood {root}/{hops}"
                    _keys(nb, NEIGHBOURHOOD, where)
                    for n in nb["nodes"]:
                        _node(n, where, {"hop"})
                    for e in nb["edges"]:
                        _keys(e, {"src", "dst"}, where)
                        if (e["src"], e["dst"]) not in edges:
                            raise SiteCheckError(
                                f"{where}: edge {(e['src'], e['dst'])} is not in the shipped subset"
                            )
        elif kind == "shard":
            _keys(p, {"time_step", "transactions"}, path)
            for entry in p["transactions"]:
                _keys(entry, SHARD_ENTRY, path)
                if not all(len(pair) == 2 and isinstance(pair[0], int) for pair in entry["top"]):
                    raise SiteCheckError(f"{path}: SHAP entries must be [feature index, value]")
