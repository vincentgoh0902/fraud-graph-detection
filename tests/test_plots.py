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
