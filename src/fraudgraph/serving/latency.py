"""Timing endpoints for the README latency table (brief section 6)."""

import time
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def percentiles(samples_ms: Sequence[float]) -> dict:
    """p50 and p95 with linear interpolation between ranks (NumPy's default)."""
    if len(samples_ms) == 0:
        raise ValueError("no samples")
    a = np.asarray(samples_ms, dtype=float)
    return {
        "n": len(a),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
    }


def time_calls(call: Callable, args: Sequence, warmup: int) -> list[float]:
    """Milliseconds per call, one call per element of args, after `warmup` untimed calls.

    The warm-up fills caches and connection pools, which a steady-state service has
    already done. `call` must raise on a failed request, so errors are never timed.
    """
    for i in range(warmup):
        call(args[i % len(args)])
    out = []
    for a in args:
        start = time.perf_counter()
        call(a)
        out.append((time.perf_counter() - start) * 1000)
    return out


BACKEND_NAMES = {"neo4j": "Neo4j (full stack)", "memory": "in-memory"}


def latency_table(by_backend: dict[str, pd.DataFrame]) -> str:
    """README table: one row per endpoint, p50 / p95 per backend; '–' where not measured."""
    backends = [b for b in BACKEND_NAMES if b in by_backend]
    endpoints = list(dict.fromkeys(e for b in backends for e in by_backend[b]["endpoint"]))
    header = ["Endpoint"] + [f"{BACKEND_NAMES[b]} p50 / p95 (ms)" for b in backends]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for e in endpoints:
        cells = [e]
        for b in backends:
            r = by_backend[b].set_index("endpoint")
            cells.append(f"{r.loc[e, 'p50_ms']:.1f} / {r.loc[e, 'p95_ms']:.1f}" if e in r.index
                         else "–")  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
