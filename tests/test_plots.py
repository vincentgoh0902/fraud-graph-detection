import pandas as pd
import pytest

from fraudgraph.evaluation.plots import per_step_f1_figure


def step_table(f1s):
    steps = range(35, 35 + len(f1s))
    return pd.DataFrame({"f1": f1s, "n_illicit": [10] * len(f1s)}, index=pd.Index(steps,
                        name="time_step"))  # fmt: skip


def test_per_step_figure_is_written(tmp_path):
    series = {
        "A": [step_table([0.8, 0.7, 0.1]), step_table([0.9, 0.6, 0.2])],
        "B": [step_table([0.5, float("nan"), 0.1])],  # a step with no illicit: F1 undefined
    }
    path = tmp_path / "fig.png"
    per_step_f1_figure(series, path)
    assert path.stat().st_size > 10_000


def test_too_many_series_rejected(tmp_path):
    series = {str(i): [step_table([0.5])] for i in range(6)}
    with pytest.raises(ValueError):
        per_step_f1_figure(series, tmp_path / "fig.png")


def test_phase5_figures_are_written(tmp_path):
    import numpy as np

    from fraudgraph.evaluation.plots import (
        group_shares_figure,
        illicit_scores_figure,
        retrain_figure,
        top_features_figure,
    )

    rng = np.random.default_rng(0)
    steps = np.repeat(np.arange(30, 50), 20)
    scores = pd.DataFrame(
        {
            "time_step": steps,
            "label": rng.integers(0, 2, len(steps)),
            "score_s0": rng.uniform(size=len(steps)),
            "threshold_s0": 0.7,
        }
    )
    illicit_scores_figure(scores, 0, tmp_path / "a.png")

    shares = pd.DataFrame(
        [
            {
                "local": 0.5,
                "aggregated": 0.3,
                "structure": 0.05,
                "risk": 0.15,
                "period": p,
                "seed": s,
            }
            for p in ("reference", "val", "before", "after")
            for s in (0, 1)
        ]
    )
    group_shares_figure(shares, tmp_path / "b.png")

    imp = pd.Series([0.5, 0.3, 0.1], index=["local_1", "agg_2", "nbr_risk_max_all"])
    top_features_figure(imp, ["local", "aggregated", "risk"], tmp_path / "c.png")

    per_step = pd.DataFrame(
        [
            {"seed": s, "time_step": t, "kind": k, "f1": 0.5, "n_illicit": 10}
            for s in (0, 1)
            for t in range(35, 50)
            for k in ("frozen", "retrained")
        ]
    )
    retrain_figure(per_step, tmp_path / "d.png")
    for name in "abcd":
        assert (tmp_path / f"{name}.png").stat().st_size > 10_000
