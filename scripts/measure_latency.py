"""Time each API endpoint over 100 requests and log p50/p95 to MLflow (brief section 6).

Usage (with the API running, e.g. `docker compose up -d`):
    uv run python scripts/measure_latency.py --url http://localhost:8000

One sequential client on the same machine, keep-alive connection, after a warm-up. The
backend (neo4j or memory) is read from the API itself, so a run cannot be mislabelled.
The README table comes from these runs: uv run python scripts/make_results_table.py
"""

import argparse
import platform
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import requests

from fraudgraph.config import load_config, resolve
from fraudgraph.serving.artifacts import load_artifacts
from fraudgraph.serving.latency import percentiles, time_calls
from fraudgraph.tracking import setup_tracking


def request_args(data, n: int, seed: int, full_graph: bool) -> dict[str, tuple[str, list]]:
    """Endpoint name -> (path template, one argument set per request). Fixed by the seed."""
    rng = np.random.default_rng(seed)
    tx = data.transactions.reset_index()
    ids = rng.choice(tx["tx_id"].to_numpy(), size=n, replace=False).tolist()
    first, last = data.meta["steps"]
    steps = [first + i % (last - first + 1) for i in range(n)]
    out = {
        "transaction": ("/transactions/{0}", [(int(t),) for t in ids]),
        "neighbourhood (2 hops)": ("/transactions/{0}/neighbourhood?hops=2",
                                   [(int(t),) for t in ids]),
        "alerts (k = 20)": ("/alerts?time_step={0}&k=20", [(s,) for s in steps]),
        "communities": ("/communities?time_step={0}", [(s,) for s in steps]),
    }  # fmt: skip
    if full_graph:
        top = tx.sort_values("risk_score", ascending=False).groupby("time_step").head(2)
        pairs = top.groupby("time_step")["tx_id"].apply(list).tolist()
        out["fan-in"] = ("/fan-in?time_step={0}", [(s,) for s in steps])
        out["paths"] = (
            "/paths?source={0}&target={1}",
            [tuple(int(t) for t in pairs[i % len(pairs)]) for i in range(n)],
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    cfg = load_config("serving")
    lat = cfg["latency"]
    session = requests.Session()
    base = args.url.rstrip("/")

    def get(path: str) -> requests.Response:
        r = session.get(base + path, timeout=30)
        r.raise_for_status()
        return r

    meta = get("/meta").json()
    data = load_artifacts(resolve(cfg["output_dir"]))
    if meta["model_version"] != data.meta["model_version"]:
        raise RuntimeError("The API serves a different export than artifacts/serving")
    print(f"Backend {meta['backend']}, {lat['n_requests']} requests per endpoint")

    rows = []
    for name, (template, arg_sets) in request_args(
        data, lat["n_requests"], lat["seed"], meta["has_full_graph"]
    ).items():
        samples = time_calls(lambda a, t=template: get(t.format(*a)), arg_sets, lat["warmup"])
        rows.append({"endpoint": name, **percentiles(samples)})
        print(f"  {name:24s} p50 {rows[-1]['p50_ms']:7.1f} ms   p95 {rows[-1]['p95_ms']:7.1f} ms")
    table = pd.DataFrame(rows)

    setup_tracking(lat["experiment"])
    tags = {"phase": "6", "role": "latency", "backend": meta["backend"],
            "model_version": meta["model_version"],
            "run_group": datetime.now().strftime("%Y%m%d-%H%M%S")}  # fmt: skip
    with mlflow.start_run(run_name=f"latency/{meta['backend']}", tags=tags):
        mlflow.log_params({"n_requests": lat["n_requests"], "warmup": lat["warmup"],
                           "url": base, "machine": platform.platform(),
                           "client": "one sequential requests.Session, same machine"})  # fmt: skip
        for r in rows:
            key = r["endpoint"].split(" ")[0].replace("-", "_")
            mlflow.log_metrics({f"{key}_p50_ms": r["p50_ms"], f"{key}_p95_ms": r["p95_ms"]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latency.csv"
            table.to_csv(path, index=False)
            mlflow.log_artifact(str(path))
    print("README table: uv run python scripts/make_results_table.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
