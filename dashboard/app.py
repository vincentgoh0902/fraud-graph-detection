"""Investigator dashboard (brief section 6): alert queue, SHAP explanation, risk graph.

Run:
    API_URL=http://localhost:8000 uv run streamlit run dashboard/app.py   # via the API
    uv run streamlit run dashboard/app.py                                 # in-process

With API_URL set it calls the API (docker compose, graph queries in Neo4j). Without it,
it reads exported files in-process (ARTIFACTS_DIR, default artifacts/space): that is the
Hugging Face Space, which has no Neo4j (D-008).
"""

import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import views

from fraudgraph.config import load_config, resolve
from fraudgraph.serving.client import ApiClient, LocalClient
from fraudgraph.serving.service import NotFound, Unavailable

REPO = "https://github.com/vincentgoh0902/fraud-graph-detection"
COLUMNS = {
    "tx_id": st.column_config.NumberColumn("transaction", format="%d"),
    "time_step": st.column_config.NumberColumn("step", format="%d"),
    "risk_score": st.column_config.NumberColumn("risk score", format="%.4f"),
    "community_id": st.column_config.NumberColumn("community", format="%d"),
    "n_flagged": "flagged",
    "share_flagged": st.column_config.NumberColumn("share flagged", format="%.3f"),
    "mean_risk": st.column_config.NumberColumn("mean risk", format="%.4f"),
    "n_flagged_payers": "flagged payers",
}
PAGES = ["Start here", "Alert queue", "Transaction", "Patterns"]

st.set_page_config(page_title="Fraud graph investigator", layout="wide")


@st.cache_resource
def get_client():
    url = os.environ.get("API_URL")
    if url:
        return ApiClient(url)
    from fraudgraph.serving.artifacts import load_artifacts
    from fraudgraph.serving.graph import InMemoryStore
    from fraudgraph.serving.service import Limits, ServingService

    cfg = load_config("serving")
    data = load_artifacts(resolve(os.environ.get("ARTIFACTS_DIR", cfg["demo"]["output_dir"])))
    store = InMemoryStore(data.edges, data.transactions)
    return LocalClient(ServingService(data, store, Limits(**cfg["limits"])))


def go(page: str, **state) -> None:
    st.session_state.update(page=page, **state)


def backend_note(meta: dict) -> None:
    """Which graph backend answers the queries (the Neo4j note)."""
    if meta["backend"] == "neo4j":
        st.caption("Graph queries run in **Neo4j** (Cypher) through the FastAPI service.")
    else:
        st.caption(
            "This demo runs on exported files, without a database. The full stack "
            f"([`docker compose up`]({REPO}#serving)) runs the same queries in **Neo4j** with "
            "Cypher, including paths and fan-in across the whole graph."
        )


def in_range(meta: dict, step: int) -> int:
    """The guided tour uses steps 40 and 43; keep them inside whatever this export serves."""
    first, last = meta["steps"]
    return min(max(step, first), last)


def label_text(label: str | None) -> str:
    return label if label is not None else "not in this demo"


# --- pages ------------------------------------------------------------------------------


def start_page(client, meta: dict) -> None:
    st.title("Investigating Bitcoin transactions flagged as illicit")
    first, last = meta["steps"]
    st.write(
        f"An XGBoost model scores every transaction in time steps {first}–{last} of the "
        "Elliptic dataset, a period it never trained on. An investigator reviews the riskiest "
        "transactions of each two-week step. This page walks you through one alert, then "
        "shows where the model stops working."
    )
    steps = client.steps()["steps"]
    by_step = {s["time_step"]: s for s in steps}
    st.subheader("How well the model catches illicit transactions, step by step")
    st.altair_chart(views.f1_chart(steps), use_container_width=True)
    st.caption(
        "Illicit-class F1 at the validation threshold, from the logged test metrics. "
        "After the dark-market shutdown at step 43 the model misses almost every illicit "
        "transaction: the new ones look licit to every feature it has."
    )

    st.subheader("Try it")
    before, after_step = in_range(meta, 40), in_range(meta, 43)
    alert = client.alerts(before, 1)["alerts"][0]
    after = by_step.get(after_step, {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Review an alert queue**")
        st.write(f"The riskiest transactions in step {before}, before the shutdown.")
        st.button(f"Open step {before}'s queue", on_click=go, args=("Alert queue",),
                  kwargs={"step": before}, key="tour_queue")  # fmt: skip
    with c2:
        st.markdown("**2. Explain the top alert**")
        st.write(
            f"Transaction {alert['tx_id']} scores {alert['risk_score']:.4f}. See which "
            "features push it towards illicit and who it pays."
        )
        st.button("Open the top alert", on_click=go, args=("Transaction",),
                  kwargs={"tx": alert["tx_id"]}, key="tour_alert")  # fmt: skip
    with c3:
        st.markdown("**3. Look after the shutdown**")
        if after.get("n_illicit") is not None and after.get("recall") is not None:
            n = int(after["n_illicit"])
            caught = round(after["recall"] * n)
            st.write(
                f"Step {after_step} has {n} labelled illicit transactions. The model "
                f"catches {caught} of them, and its {after['n_flagged']} flags go elsewhere."
            )
        st.button(f"Open step {after_step}'s queue", on_click=go, args=("Alert queue",),
                  kwargs={"step": after_step}, key="tour_after")  # fmt: skip
    with st.expander("Per-step numbers (table)"):
        st.dataframe(pd.DataFrame(steps), hide_index=True, use_container_width=True)


def alerts_page(client, meta: dict) -> None:
    first, last = meta["steps"]
    st.title("Alert queue")
    c1, c2 = st.columns([1, 2])
    step = c1.selectbox("Time step", list(range(first, last + 1)),
                        index=in_range(meta, st.session_state.get("step", 40)) - first)  # fmt: skip
    k = c2.slider("Alerts to review", 5, min(100, meta["max_k"]), 20)
    st.session_state["step"] = step
    data = client.alerts(step, k)
    df = pd.DataFrame(data["alerts"])
    df.insert(0, "rank", range(1, len(df) + 1))
    df["flagged"] = df["flagged"].map({True: "flagged", False: "—"})
    df["label"] = df["label"].map(label_text)
    st.caption(
        f"Ranked by risk score. Flagged means score ≥ {data['threshold']:.3f}, the threshold "
        "tuned on validation. Select a row to open it."
    )
    event = st.dataframe(
        df[["rank", "tx_id", "risk_score", "flagged", "label"]],
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={"risk_score": st.column_config.NumberColumn("risk score", format="%.4f"),
                       "tx_id": st.column_config.NumberColumn("transaction", format="%d")},
    )  # fmt: skip
    if event.selection.rows:
        go("Transaction", tx=int(df.iloc[event.selection.rows[0]]["tx_id"]))
        st.rerun()


def transaction_page(client, meta: dict) -> None:
    st.title("Transaction")
    default = (
        st.session_state.get("tx") or client.alerts(in_range(meta, 40), 1)["alerts"][0]["tx_id"]
    )
    tx_id = int(st.number_input("Transaction id", value=int(default), step=1, format="%d"))
    st.session_state["tx"] = tx_id
    try:
        tx = client.transaction(tx_id)
    except NotFound as err:
        st.error(str(err))
        return

    c = st.columns(4)
    c[0].metric("Risk score", f"{tx['risk_score']:.4f}")
    c[1].metric("Threshold", f"{tx['threshold']:.3f}")
    c[2].metric("Decision", "Flagged" if tx["flagged"] else "Not flagged")
    c[3].metric("Label", label_text(tx["label"]))
    st.caption(f"Time step {tx['time_step']}.")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Why this score")
        st.altair_chart(views.shap_chart(tx), use_container_width=True)
        st.caption(
            f"The {len(tx['top_features'])} largest SHAP contributions, in log-odds. The "
            f"model starts from {tx['base_value']:+.2f} and adds every feature's contribution; "
            "+1 multiplies the odds of illicit by about 2.7. Features are anonymised in the "
            "dataset, so only their names and groups are known."
        )
        with st.expander("Contributions (table)"):
            st.dataframe(pd.DataFrame(tx["top_features"]), hide_index=True)

    with right:
        st.subheader("Who it transacts with")
        hops = st.radio("Hops", list(range(1, meta["max_hops"] + 1)), index=0, horizontal=True)
        try:
            nb = client.neighbourhood(tx_id, hops)
        except Unavailable as err:
            st.info(str(err))
            backend_note(meta)
            return
        if nb["truncated"]:
            st.warning(
                f"Showing the nearest {len(nb['nodes'])} transactions, riskiest first within "
                "each hop; the full neighbourhood is larger."
            )
        components.html(views.network_html(nb), height=580)
        st.markdown(views.risk_legend_html(), unsafe_allow_html=True)
        with st.expander("Neighbourhood (table)"):
            nodes = pd.DataFrame(nb["nodes"])
            nodes["label"] = nodes["label"].map(label_text)
            st.dataframe(nodes, hide_index=True, use_container_width=True)

    if meta["has_full_graph"]:
        st.subheader("Paths to another transaction")
        other = st.number_input("Target transaction id", value=0, step=1, format="%d")
        if other:
            try:
                paths = client.paths(tx_id, int(other))["paths"]
            except NotFound as err:
                st.error(str(err))
            else:
                if not paths:
                    st.write("No path within 6 hops (transactions only connect inside a step).")
                for p in paths:
                    st.code(" → ".join(str(t) for t in p))


def patterns_page(client, meta: dict) -> None:
    first, last = meta["steps"]
    st.title("Patterns in a step")
    step = st.selectbox("Time step", list(range(first, last + 1)),
                        index=in_range(meta, st.session_state.get("step", 40)) - first)  # fmt: skip
    st.subheader("Communities with the most flagged transactions")
    comm = client.communities(step, 5, 10)["items"]
    st.dataframe(pd.DataFrame(comm), hide_index=True, use_container_width=True,
                 column_config=COLUMNS)  # fmt: skip
    st.caption(
        "Louvain communities. On this data they are close to whole steps (the largest holds "
        "most of its step), so few communities stand out."
    )
    st.subheader("Fan-in: transactions paid by several flagged transactions")
    try:
        fan = client.fan_in(step, 2, 20)["items"]
    except Unavailable as err:
        st.info(str(err))
        backend_note(meta)
        return
    if fan:
        df = pd.DataFrame(fan)
        df["label"] = df["label"].map(label_text)
        st.dataframe(df, hide_index=True, use_container_width=True, column_config=COLUMNS)
    else:
        st.write("No transaction in this step is paid by two or more flagged transactions.")


def main() -> None:
    client = get_client()
    meta = client.meta()
    st.session_state.setdefault("page", PAGES[0])
    with st.sidebar:
        st.radio("View", PAGES, key="page")
        st.divider()
        st.caption(f"Model `{meta['model_version']}`")
        backend_note(meta)
        st.caption(f"[Code and results]({REPO})")
    {"Start here": start_page, "Alert queue": alerts_page, "Transaction": transaction_page,
     "Patterns": patterns_page}[st.session_state["page"]](client, meta)  # fmt: skip


main()
