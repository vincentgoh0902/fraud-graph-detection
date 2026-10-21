"""README figures built from logged MLflow artefacts (no number is typed by hand)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no display; also safe in CI
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical slots in fixed order (validated: adjacent CVD and normal-vision separation
# pass on the light surface). Three slots are below 3:1 contrast, so every line also has
# its own marker shape, and the README prints the same numbers as a table.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
SURFACE, TEXT, TEXT_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7"


def per_step_f1_figure(series: dict[str, list[pd.DataFrame]], path: Path) -> None:
    """Illicit F1 per test step (mean over seeds), with the illicit count per step below.

    series maps a label to each seed's per_step_test.csv table (indexed by time_step).
    The count is a separate panel, not a second y-axis: it shows how many illicit
    transactions each step's F1 rests on.
    """
    if len(series) > len(SERIES_COLORS):
        raise ValueError(f"At most {len(SERIES_COLORS)} series")
    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]},
        facecolor=SURFACE,
    )  # fmt: skip
    first = next(iter(series.values()))[0]
    for i, (label, tables) in enumerate(series.items()):
        f1 = pd.concat([t["f1"] for t in tables], axis=1).mean(axis=1)
        ax.plot(f1.index, f1.to_numpy(), color=SERIES_COLORS[i], lw=2, marker=MARKERS[i],
                ms=5, label=label)  # fmt: skip

    ax.set_ylabel("Illicit F1 (validation-tuned threshold)", color=TEXT_2)
    ax.set_ylim(0, 1.05)
    ax.set_title("Illicit F1 per test time step (mean over 3 seeds)", color=TEXT, loc="left")
    # Upper right is empty: every model is near zero after step 42.
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=TEXT_2)

    ax_n.bar(first.index, first["n_illicit"], color=MUTED, width=0.7)
    ax_n.set_ylabel("Illicit\ntransactions", color=TEXT_2)
    ax_n.set_xlabel("Time step", color=TEXT_2)
    ax_n.set_xticks(first.index)

    for a in (ax, ax_n):
        a.set_facecolor(SURFACE)
        a.grid(axis="y", color="#e6e5e0", lw=0.8)
        a.set_axisbelow(True)
        a.tick_params(colors=TEXT_2, labelsize=8.5)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(MUTED)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# Phase 5 figures. Feature groups keep the same colour in every figure (colour follows the
# entity), and every figure's numbers are also printed as a README table.
GROUP_COLORS = dict(zip(("local", "aggregated", "structure", "risk"), SERIES_COLORS, strict=False))
GROUP_LABELS = {
    "local": "Local",
    "aggregated": "Aggregated",
    "structure": "Graph structure",
    "risk": "Neighbour / community risk",
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_2, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def illicit_scores_figure(scores: pd.DataFrame, seed: int, path: Path) -> None:
    """Score of every illicit transaction per step (one seed), against the threshold.

    Shows whether late illicit transactions are near-misses (just under the line, a
    threshold problem) or scored like licit ones (a ranking problem). Licit transactions'
    90th percentile is drawn per step as a reference: an illicit score below it is ranked
    below at least 10% of the licit transactions of that step.
    """
    col, thr = f"score_s{seed}", scores[f"threshold_s{seed}"].iloc[0]
    rng = np.random.default_rng(0)  # horizontal jitter only, so points don't stack
    ill = scores[scores["label"] == 1]
    lic90 = scores[scores["label"] == 0].groupby("time_step")[col].quantile(0.9)
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    ax.scatter(
        ill["time_step"] + rng.uniform(-0.28, 0.28, len(ill)),
        ill[col],
        s=9,
        color=SERIES_COLORS[0],
        alpha=0.45,
        linewidths=0,
        label="Illicit transaction",
    )
    ax.plot(
        lic90.index,
        lic90.to_numpy(),
        color=TEXT_2,
        lw=1.5,
        marker="_",
        ms=10,
        label="Licit, 90th percentile",
    )
    ax.axhline(thr, color=SERIES_COLORS[1], lw=2, ls="--",
               label=f"Validation-tuned threshold ({thr:.2f})")  # fmt: skip
    ax.axvline(42.5, color=MUTED, lw=1)
    ax.text(42.6, 1.02, "after step 42", color=TEXT_2, fontsize=8.5, va="bottom")
    ax.set_ylim(-0.02, 1.08)
    ax.set_xticks(sorted(scores["time_step"].unique()))
    ax.set_xlabel("Time step (30–34 validation, 35–49 test)", color=TEXT_2)
    ax.set_ylabel("Predicted illicit probability", color=TEXT_2)
    ax.set_title(f"Scores given to illicit transactions (seed {seed})", color=TEXT, loc="left",
                 pad=28)  # fmt: skip
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    # Above the plot: every corner inside it holds illicit points.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.5,
              labelcolor=TEXT_2)  # fmt: skip
    _style(ax)
    _save(fig, path)


def group_shares_figure(shares: pd.DataFrame, path: Path) -> None:
    """Share of total mean |SHAP| per feature group, per period (mean over seeds).

    One 100% bar per period; segments are separated by a 2px surface gap and labelled
    directly when wide enough.
    """
    order = ["reference", "val", "before", "after"]
    names = {"reference": "Training (1–29)", "val": "Validation (30–34)",
             "before": "Test, before (35–42)", "after": "Test, after (43–49)"}  # fmt: skip
    mean = shares.groupby("period")[list(GROUP_COLORS)].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=SURFACE)
    left = np.zeros(len(order))
    for g, color in GROUP_COLORS.items():
        v = mean[g].to_numpy()
        ax.barh(range(len(order)), v, left=left, color=color, height=0.6,
                edgecolor=SURFACE, linewidth=2, label=GROUP_LABELS[g])  # fmt: skip
        for i, (x0, w) in enumerate(zip(left, v, strict=True)):
            if w >= 0.06:
                ax.text(x0 + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=8.5,
                        color=TEXT)  # fmt: skip
        left += v
    ax.set_yticks(range(len(order)), [names[p] for p in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Where the model's attribution goes: share of mean |SHAP| by feature group",
                 color=TEXT, loc="left")  # fmt: skip
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False,
              fontsize=8.5, labelcolor=TEXT_2)  # fmt: skip
    _style(ax)
    _save(fig, path)


def top_features_figure(importance: pd.Series, groups: list[str], path: Path) -> None:
    """Top features by mean |SHAP| on the training period, coloured by feature group."""
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(importance) + 1.2), facecolor=SURFACE)
    y = np.arange(len(importance))
    ax.barh(y, importance.to_numpy(), color=[GROUP_COLORS[g] for g in groups], height=0.7,
            edgecolor=SURFACE, linewidth=2)  # fmt: skip
    ax.set_yticks(y, importance.index)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| (log-odds), training period, mean over seeds", color=TEXT_2)
    ax.set_title("Most important features of the selected model", color=TEXT, loc="left")
    handles = [matplotlib.patches.Patch(color=c, label=GROUP_LABELS[g])
               for g, c in GROUP_COLORS.items() if g in set(groups)]  # fmt: skip
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5,
              labelcolor=TEXT_2)  # fmt: skip
    ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    _style(ax)
    _save(fig, path)


def retrain_figure(per_step: pd.DataFrame, path: Path) -> None:
    """Illicit F1 per test step: frozen model vs retrained every step (mean over seeds)."""
    mean = per_step.groupby(["kind", "time_step"])[["f1", "n_illicit"]].mean()
    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]},
        facecolor=SURFACE,
    )  # fmt: skip
    labels = {
        "frozen": "Frozen: trained once on 1–29",
        "retrained": "Retrained each step (train ≤ t−6, threshold on t−5…t−1)",
    }
    for i, kind in enumerate(("frozen", "retrained")):
        s = mean.loc[kind, "f1"]
        ax.plot(s.index, s.to_numpy(), color=SERIES_COLORS[i], lw=2, marker=MARKERS[i], ms=5,
                label=labels[kind])  # fmt: skip
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Illicit F1", color=TEXT_2)
    ax.set_title("What if the model were retrained as labels arrive? (XGBoost (b), "
                 "mean over 3 seeds)", color=TEXT, loc="left")  # fmt: skip
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=TEXT_2)
    n = mean.loc["frozen", "n_illicit"]
    ax_n.bar(n.index, n.to_numpy(), color=MUTED, width=0.7)
    ax_n.set_ylabel("Illicit\ntransactions", color=TEXT_2)
    ax_n.set_xlabel("Time step", color=TEXT_2)
    ax_n.set_xticks(n.index)
    for a in (ax, ax_n):
        a.grid(axis="y", color="#e6e5e0", lw=0.8)
        _style(a)
    _save(fig, path)
