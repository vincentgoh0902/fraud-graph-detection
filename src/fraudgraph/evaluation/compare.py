"""How sure are we that feature set B beats feature set A on test? (D-024)

Paired bootstrap over test time steps: draw 15 steps with replacement from steps 35-49,
score both feature sets on exactly those steps, and record B minus A. Repeating this
gives an interval for the change itself.
- Whole steps are resampled, not single transactions, because transactions in one step
  are correlated (same market conditions, same graph); resampling transactions would
  understate the uncertainty (the limitation noted in D-005).
- Paired: both sets are scored on the same resample, so step-to-step variation that hits
  both equally cancels out of the difference.
- Each resample compares seed means, matching the "difference of seed means" column.
This only measures; nothing here selects a model or a threshold.
"""

import numpy as np

from fraudgraph.evaluation.metrics import pr_auc, precision_recall_at_k


def _metrics(y, steps, score, k):
    """PR-AUC over all nodes, and precision@k averaged over steps (as in the protocol)."""
    p_at_k = [
        precision_recall_at_k(y[steps == s], score[steps == s], k)[0] for s in np.unique(steps)
    ]
    return pr_auc(y, score), float(np.mean(p_at_k))


def _seed_mean(y, steps, scores, k):
    return np.mean([_metrics(y, steps, s, k) for s in scores], axis=0)


def paired_step_bootstrap(
    y: np.ndarray,
    steps: np.ndarray,
    scores_a: list[np.ndarray],
    scores_b: list[np.ndarray],
    k: int,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Change B - A in PR-AUC and precision@k, with 95% percentile intervals.

    scores_a / scores_b: one score array per seed, all aligned with y and steps.
    share_positive: fraction of resamples in which B's PR-AUC is higher.
    """
    for s in [*scores_a, *scores_b]:
        if len(s) != len(y):
            raise ValueError("Every score array must align with y and steps")

    observed = _seed_mean(y, steps, scores_b, k) - _seed_mean(y, steps, scores_a, k)

    rng = np.random.default_rng(seed)
    unique_steps = np.unique(steps)
    rows_of = {s: np.flatnonzero(steps == s) for s in unique_steps}
    changes = []
    for _ in range(n_boot):
        drawn = rng.choice(unique_steps, size=len(unique_steps), replace=True)
        idx = np.concatenate([rows_of[s] for s in drawn])
        # A step drawn twice must count as two steps for precision@k, so each draw gets
        # its own label (0, 1, 2, ...) instead of keeping its real step number.
        draw_id = np.repeat(np.arange(len(drawn)), [len(rows_of[s]) for s in drawn])
        yb = y[idx]
        if yb.sum() == 0:
            continue  # no illicit in this resample: PR-AUC undefined
        a = _seed_mean(yb, draw_id, [s[idx] for s in scores_a], k)
        b = _seed_mean(yb, draw_id, [s[idx] for s in scores_b], k)
        changes.append(b - a)
    changes = np.array(changes)

    lo, hi = np.percentile(changes, [2.5, 97.5], axis=0)
    return {
        "pr_auc_change": float(observed[0]),
        "pr_auc_lo": float(lo[0]),
        "pr_auc_hi": float(hi[0]),
        "precision_at_k_change": float(observed[1]),
        "precision_at_k_lo": float(lo[1]),
        "precision_at_k_hi": float(hi[1]),
        "share_positive": float((changes[:, 0] > 0).mean()),
        "n_boot": len(changes),
    }
