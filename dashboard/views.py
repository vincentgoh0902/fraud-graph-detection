"""Charts and the network view for the dashboard.

Colours come from the validated reference palette (light theme only, see
.streamlit/config.toml): a one-hue blue ramp for risk (ordinal check passed) and the
red/blue diverging pair for SHAP signs. Colour never carries meaning alone: flagged
nodes also change shape, bars carry their values, and every chart has a table next to it.
"""

import altair as alt
import pandas as pd
from pyvis.network import Network

INK, INK_MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
# Risk bands, light to dark: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.
RISK_EDGES = [0.2, 0.4, 0.6, 0.8]
RISK_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
TOWARDS_ILLICIT, TOWARDS_LICIT = "#e34948", "#2a78d6"
SHUTDOWN = 42.5  # between steps 42 and 43 (Weber et al., 2019)


def risk_band(score: float) -> int:
    return sum(score >= e for e in RISK_EDGES)


def risk_legend_html() -> str:
    labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1"]
    swatches = " ".join(
        f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;'
        f'background:{c};vertical-align:middle"></span> {lab}&nbsp;&nbsp;'
        for c, lab in zip(RISK_COLORS, labels, strict=True)
    )
    return (
        f'<div style="color:{INK_MUTED};font-size:0.85rem">Risk score: {swatches}<br>'
        "Shape: ★ the transaction you opened · ◆ flagged (score ≥ threshold) · ● not flagged."
        " Arrows show the direction of payment.</div>"
    )


def network_html(neighbourhood: dict, height: int = 560) -> str:
    """The neighbourhood as an interactive vis.js graph (pyvis), self-contained HTML."""
    focal = neighbourhood["tx_id"]
    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor=SURFACE,
                  font_color=INK, cdn_resources="in_line")  # fmt: skip
    for n in neighbourhood["nodes"]:
        is_focal = n["tx_id"] == focal
        label = n["label"] or "not in this demo"
        tip = (f"Transaction {n['tx_id']}\nRisk {n['risk_score']:.3f}\nHop {n['hop']}\n"
               f"{'Flagged' if n['flagged'] else 'Not flagged'}\nLabel: {label}")  # fmt: skip
        net.add_node(
            n["tx_id"],
            label=str(n["tx_id"]) if is_focal or n["flagged"] else " ",
            title=tip,
            color={"background": RISK_COLORS[risk_band(n["risk_score"])], "border": INK},
            shape="star" if is_focal else "diamond" if n["flagged"] else "dot",
            size=26 if is_focal else 14 if n["flagged"] else 9,
            borderWidth=2 if is_focal else 1,
        )
    for e in neighbourhood["edges"]:
        net.add_edge(e["src"], e["dst"], color=GRID, width=1)
    net.set_options(
        '{"physics": {"barnesHut": {"springLength": 90}, "stabilization": {"iterations": 200}},'
        ' "interaction": {"hover": true, "tooltipDelay": 80}}'
    )
    return net.generate_html()


def shap_chart(transaction: dict) -> alt.Chart:
    """Top SHAP contributions as horizontal bars, signed, with their values written on."""
    df = pd.DataFrame(transaction["top_features"])
    df["direction"] = df["shap"].map(lambda v: "towards illicit" if v > 0 else "towards licit")
    df["value"] = df["shap"].map(lambda v: f"{v:+.2f}")
    order = df["feature"].tolist()
    # Every feature name shown: Vega-Lite otherwise hides labels it thinks overlap.
    y = alt.Y(
        "feature:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)
    )
    base = alt.Chart(df).encode(
        y=y, tooltip=["feature", "group", alt.Tooltip("shap:Q", format="+.3f")]
    )
    bars = base.mark_bar(height=12, cornerRadiusEnd=4).encode(
        x=alt.X("shap:Q", title="SHAP value (log-odds)"),
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(domain=["towards illicit", "towards licit"],
                            range=[TOWARDS_ILLICIT, TOWARDS_LICIT]),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )  # fmt: skip
    # Each value sits just past the end of its bar, on the side the bar points to.
    pos = (
        base.transform_filter("datum.shap > 0")
        .mark_text(color=INK_MUTED, align="left", dx=4, baseline="middle")
        .encode(x="shap:Q", text="value:N")
    )
    neg = (
        base.transform_filter("datum.shap <= 0")
        .mark_text(color=INK_MUTED, align="right", dx=-4, baseline="middle")
        .encode(x="shap:Q", text="value:N")
    )
    return _style((bars + pos + neg).properties(height=30 * len(df)))


def f1_chart(steps: list[dict]) -> alt.Chart:
    """Logged F1 per test step, with the dark-market shutdown marked."""
    df = pd.DataFrame(steps).dropna(subset=["f1"])
    line = alt.Chart(df).mark_line(color=TOWARDS_LICIT, strokeWidth=2, point=alt.OverlayMarkDef(
        size=60, color=TOWARDS_LICIT, filled=True, stroke=SURFACE, strokeWidth=2)).encode(
        x=alt.X("time_step:Q", title="Time step", scale=alt.Scale(domain=[35, 49]),
                axis=alt.Axis(values=list(range(35, 50)), format="d")),
        y=alt.Y("f1:Q", title="Illicit F1", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("time_step:Q", title="Step"), alt.Tooltip("f1:Q", format=".2f"),
                 alt.Tooltip("n_illicit:Q", title="Illicit (labelled)")],
    )  # fmt: skip
    rule_df = pd.DataFrame({"x": [SHUTDOWN], "text": ["dark-market shutdown"]})
    # Every layer shares the step axis; without its own domain a layer pulls it back to 0.
    x = alt.X("x:Q", scale=alt.Scale(domain=[35, 49]))
    rule = alt.Chart(rule_df).mark_rule(color=INK_MUTED, strokeDash=[4, 4]).encode(x=x)
    note = (
        alt.Chart(rule_df)
        .mark_text(color=INK_MUTED, align="left", dx=6, dy=-6)
        .encode(x=x, y=alt.value(12), text="text:N")
    )
    return _style((rule + note + line).properties(height=260))


def _style(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            labelColor=INK_MUTED, titleColor=INK_MUTED, gridColor=GRID, domainColor=GRID
        )
        .configure_legend(labelColor=INK, titleColor=INK)
    )
