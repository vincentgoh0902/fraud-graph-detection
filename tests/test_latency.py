"""Latency summary for the README (brief section 6: p50 and p95 over 100 requests)."""

import pytest

from fraudgraph.serving.latency import percentiles, time_calls


def test_percentiles_linear_interpolation():
    out = percentiles([float(x) for x in range(1, 101)])
    assert out["n"] == 100
    assert out["p50_ms"] == pytest.approx(50.5)
    assert out["p95_ms"] == pytest.approx(95.05)
    assert out["mean_ms"] == pytest.approx(50.5)


def test_percentiles_refuses_empty():
    with pytest.raises(ValueError):
        percentiles([])


def test_time_calls_skips_warmup_and_records_each_call():
    seen = []
    samples = time_calls(seen.append, list(range(5)), warmup=2)
    assert len(samples) == 5  # one sample per measured call
    assert seen == [0, 1, 0, 1, 2, 3, 4]  # warm-up calls reuse the first arguments
    assert all(s >= 0 for s in samples)


def test_time_calls_fails_on_an_error_response():
    def bad(_):
        raise RuntimeError("500")

    with pytest.raises(RuntimeError):
        time_calls(bad, [1], warmup=0)


def test_latency_table_marks_missing_endpoints():
    import pandas as pd

    from fraudgraph.serving.latency import latency_table

    mem = pd.DataFrame({"endpoint": ["transaction"], "p50_ms": [1.24], "p95_ms": [2.0]})
    neo = pd.DataFrame({"endpoint": ["transaction", "paths"], "p50_ms": [3.0, 9.0],
                        "p95_ms": [4.0, 12.5]})  # fmt: skip
    out = latency_table({"memory": mem, "neo4j": neo}).splitlines()
    assert out[0] == "| Endpoint | Neo4j (full stack) p50 / p95 (ms) | in-memory p50 / p95 (ms) |"
    assert out[2] == "| transaction | 3.0 / 4.0 | 1.2 / 2.0 |"
    assert out[3] == "| paths | 9.0 / 12.5 | – |"
