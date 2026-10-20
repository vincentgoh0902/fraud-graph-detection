"""README figures built from logged MLflow artefacts (no number is typed by hand)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no display; also safe in CI
import matplotlib.pyplot as plt  # noqa: E402
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
