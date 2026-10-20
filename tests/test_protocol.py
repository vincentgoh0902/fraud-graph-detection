"""The selection protocol, tested with fake models so the rules are checked directly:
choose on validation, touch test only with the chosen configuration, and fail if a
retrained model doesn't reproduce its validation score."""

import mlflow
import numpy as np
import pytest

from fraudgraph.data.split import TemporalSplit
from fraudgraph.evaluation.protocol import (
    EvalConfig,
    NonDeterministicError,
    expand_grid,
    run_protocol,
)

N = 400


@pytest.fixture
def setup(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test")
    rng = np.random.default_rng(0)
    y = (rng.random(N) < 0.2).astype(int)
    steps = np.repeat(np.arange(1, 5), N // 4)  # steps 1-2 train, 3 val, 4 test
    split = TemporalSplit(train=steps <= 2, val=steps == 3, test=steps == 4)
    cfg = EvalConfig(k=5, extra_ks=(10,), fn_cost=10, fp_cost=1, n_bootstrap=20)
    yield y, steps, split, cfg
    mlflow.end_run()


def make_fake_fit(y, split, test_calls):
    """A fake model whose quality is its 'quality' parameter: 0 is random, 1 is perfect."""

    def fit(params, seed):
        rng = np.random.default_rng(seed)
        noise = rng.random(N)

        def score(period):
            if period == "test":
                test_calls.append(dict(params, seed=seed))
            mask = getattr(split, period)
            return (params["quality"] * y + noise)[mask]

        return score

    return fit


def test_expand_grid():
    grid = expand_grid({"a": [1, 2], "b": ["x"]})
    assert grid == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]


def test_selects_on_validation_and_touches_test_only_when_chosen(setup):
    y, steps, split, cfg = setup
    test_calls = []
    fit = make_fake_fit(y, split, test_calls)
    result = run_protocol(
        fit, {"quality": [0.0, 2.0, 0.5]}, [0, 1, 2], y, steps, split, cfg, tags={}
    )
    assert result.params == {"quality": 2.0}
    # Test predictions were made exactly once per seed, all with the chosen config.
    assert test_calls == [{"quality": 2.0, "seed": s} for s in (0, 1, 2)]
    assert len(result.final_metrics) == 3


def test_runs_are_logged_with_roles(setup):
    y, steps, split, cfg = setup
    fit = make_fake_fit(y, split, [])
    run_protocol(fit, {"quality": [0.0, 1.0]}, [0, 1], y, steps, split, cfg, tags={"model": "m"})
    runs = mlflow.search_runs()
    roles = runs["tags.role"].value_counts().to_dict()
    assert roles == {"candidate": 4, "final": 2}
    final = runs[runs["tags.role"] == "final"]
    # Candidates never have test metrics; finals have the full set.
    assert runs.loc[runs["tags.role"] == "candidate", "metrics.test_pr_auc"].isna().all()
    for col in ["test_pr_auc", "test_f1", "test_precision_at_5", "val_f1_lo", "threshold_cost"]:
        assert final[f"metrics.{col}"].notna().all(), col
    assert (final["tags.model"] == "m").all()


def test_nondeterministic_model_is_rejected(setup):
    y, steps, split, cfg = setup
    calls = {"n": 0}

    def fit(params, seed):
        calls["n"] += 1
        noise = np.random.default_rng(calls["n"]).random(N)  # differs on every fit
        # Weak signal: y + noise would separate perfectly, so every fit would score 1.0.
        return lambda period: (0.3 * y + noise)[getattr(split, period)]

    with pytest.raises(NonDeterministicError):
        run_protocol(fit, {"q": [1]}, [0], y, steps, split, cfg, tags={})


def test_on_final_runs_once_per_seed_inside_the_final_run(setup):
    y, steps, split, cfg = setup
    seen = []

    def on_final(scorer, seed):
        run = mlflow.active_run()
        seen.append((seed, run.data.tags["role"], callable(scorer)))

    fit = make_fake_fit(y, split, [])
    run_protocol(
        fit, {"quality": [0.0, 1.0]}, [0, 1], y, steps, split, cfg, tags={}, on_final=on_final
    )
    assert seen == [(0, "final", True), (1, "final", True)]
