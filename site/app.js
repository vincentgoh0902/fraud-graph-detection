// The static demo (Phase 7): the dashboard's views over prebuilt JSON (D-043 to D-045).
// Hash routes keep every URL inside the page, so the site works under any base path:
//   #/                     start here (guided tour)
//   #/alerts/40?k=20       alert queue
//   #/tx/96928334?step=40  transaction detail (step optional: looked up when missing)
//   #/patterns/40          communities

import * as data from "./data.js";
import { RISK_COLORS, RISK_LABELS, f1Chart, networkChart, shapChart } from "./charts.js";
import { fixed, integer, signed } from "./format.js";

const REPO = "https://github.com/vincentgoh0902/fraud-graph-detection";
const main = document.getElementById("main");

// Small DOM builder: text is always set as text, never parsed as HTML.
function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) node.append(c);
  return node;
}
const link = (href, text, props = {}) => h("a", { href, ...props }, text);
const labelText = (label) => label ?? "not in this demo";
// Room a chart gets: main's content width, or the left column (2 of 5 parts) when the
// transaction view shows two columns (wider than the 820px breakpoint in style.css).
const contentWidth = () => main.clientWidth - 32;
const leftColumnWidth = () => (window.innerWidth > 820 ? (contentWidth() - 24) * 0.4 : contentWidth());

function table(columns, rows) {
  return h("div", { class: "table-wrap" },
    h("table", {},
      h("thead", {}, h("tr", {}, columns.map((c) => h("th", { scope: "col" }, c.title)))),
      h("tbody", {}, rows.map((r) => h("tr", {}, columns.map((c) => h("td", { class: c.num ? "num" : "" }, c.cell(r))))))));
}

function details(summary, ...content) {
  return h("details", {}, h("summary", {}, summary), ...content);
}

function stepSelect(meta, current, onChange) {
  const [first, last] = meta.steps;
  const sel = h("select", { id: "step", onchange: (e) => onChange(Number(e.target.value)) });
  for (let t = first; t <= last; t++) {
    const opt = h("option", { value: t }, `Step ${t}`);
    if (t === current) opt.selected = true;
    sel.append(opt);
  }
  return h("label", { class: "field" }, "Time step ", sel);
}

const inRange = (meta, t) => Math.min(Math.max(t, meta.steps[0]), meta.steps[1]);

function neo4jNote() {
  return h("p", { class: "note" },
    "This demo runs on prebuilt files, without a database. The full stack (",
    link(`${REPO}#serving-phase-6`, "docker compose up"),
    ") runs the same queries in Neo4j with Cypher through a FastAPI service, including paths and fan-in across the whole graph.");
}

// --- views -------------------------------------------------------------------------------

async function startView() {
  const { meta, steps, tour } = await data.index();
  const after = steps.find((s) => s.time_step === tour.after) ?? {};
  const n = after.n_illicit;
  const caught = n != null && after.recall != null ? Math.round(after.recall * n) : null;
  const [first, last] = meta.steps;
  return [
    h("h1", {}, "Investigating Bitcoin transactions flagged as illicit"),
    h("p", { class: "lead" },
      `An XGBoost model scores every transaction in time steps ${first}–${last} of the Elliptic dataset, ` +
      "a period it never trained on. An investigator reviews the riskiest transactions of each two-week " +
      "step. This page walks you through one alert, then shows where the model stops working."),
    h("h2", {}, "How well the model catches illicit transactions, step by step"),
    h("figure", {}, f1Chart(steps, contentWidth()),
      h("figcaption", {},
        "Illicit-class F1 at the validation threshold, from the logged test metrics. After the dark-market " +
        "shutdown at step 43 the model misses almost every illicit transaction: the new ones look licit to " +
        "every feature it has.")),
    h("h2", {}, "Try it"),
    h("div", { class: "cards" },
      h("section", { class: "card" }, h("h3", {}, "1. Review an alert queue"),
        h("p", {}, `The riskiest transactions in step ${tour.before}, before the shutdown.`),
        link(`#/alerts/${tour.before}`, `Open step ${tour.before}'s queue`, { class: "button" })),
      h("section", { class: "card" }, h("h3", {}, "2. Explain the top alert"),
        h("p", {}, `Transaction ${tour.alert.tx_id} scores ${fixed(tour.alert.risk_score, 4)}. See which ` +
          "features push it towards illicit and who it pays."),
        link(`#/tx/${tour.alert.tx_id}?step=${tour.before}`, "Open the top alert", { class: "button" })),
      h("section", { class: "card" }, h("h3", {}, "3. Look after the shutdown"),
        caught !== null && h("p", {}, `Step ${tour.after} has ${integer(n)} labelled illicit transactions. ` +
          `The model catches ${integer(caught)} of them, and its ${integer(after.n_flagged)} flags go elsewhere.`),
        link(`#/alerts/${tour.after}`, `Open step ${tour.after}'s queue`, { class: "button" }))),
    details("Per-step numbers (table)", table([
      { title: "Step", cell: (r) => String(r.time_step), num: true },
      { title: "Transactions", cell: (r) => integer(r.n_transactions), num: true },
      { title: "Flagged", cell: (r) => integer(r.n_flagged), num: true },
      { title: "Illicit (labelled)", cell: (r) => (r.n_illicit == null ? "–" : integer(r.n_illicit)), num: true },
      { title: "F1", cell: (r) => (r.f1 == null ? "–" : fixed(r.f1, 3)), num: true },
      { title: "Recall", cell: (r) => (r.recall == null ? "–" : fixed(r.recall, 3)), num: true },
      { title: "PR-AUC", cell: (r) => (r.pr_auc == null ? "–" : fixed(r.pr_auc, 3)), num: true },
    ], steps)),
  ];
}

async function alertsView(params, query) {
  const { meta } = await data.index();
  const t = inRange(meta, Number(params[0] ?? 40));
  const k = Math.min(Number(query.get("k") ?? 20), meta.alerts_k);
  const stepData = await data.step(t);
  const rows = stepData.alerts.slice(0, k); // the service's top k is the prefix (tested)
  const kSelect = h("select", { id: "k", onchange: (e) => (location.hash = `#/alerts/${t}?k=${e.target.value}`) },
    [5, 10, 20, 50, 100].filter((v) => v <= meta.alerts_k).map((v) => {
      const o = h("option", { value: v }, String(v));
      if (v === k) o.selected = true;
      return o;
    }));
  return [
    h("h1", {}, "Alert queue"),
    h("div", { class: "controls" },
      stepSelect(meta, t, (s) => (location.hash = `#/alerts/${s}?k=${k}`)),
      h("label", { class: "field" }, "Alerts to review ", kSelect)),
    h("p", { class: "note" }, `Ranked by risk score. Flagged means score ≥ ${fixed(meta.threshold, 3)}, ` +
      "the threshold tuned on validation. Open a transaction to see why."),
    table([
      { title: "Rank", cell: (r) => String(rows.indexOf(r) + 1), num: true },
      { title: "Transaction", cell: (r) => link(`#/tx/${r.tx_id}?step=${t}`, String(r.tx_id)) },
      { title: "Risk score", cell: (r) => fixed(r.risk_score, 4), num: true },
      { title: "Flagged", cell: (r) => (r.flagged ? "◆ flagged" : "—") },
      { title: "Label", cell: (r) => labelText(r.label) },
    ], rows),
  ];
}

function riskLegend() {
  return h("p", { class: "note legend" }, "Risk score: ",
    RISK_COLORS.map((c, i) => h("span", { class: "swatch-item" },
      h("span", { class: "swatch", style: `background:${c}` }), ` ${RISK_LABELS[i]} `)),
    h("br"), "Shape: ★ the transaction you opened · ◆ flagged (score ≥ threshold) · ● not flagged. " +
    "Arrows show the direction of payment. Tap a node for its details.");
}

async function neighbourhoodSection(tx, meta, hops) {
  const roots = meta.neighbourhood_roots ?? [];
  const section = [h("h2", {}, "Who it transacts with")];
  if (!roots.includes(tx.tx_id)) {
    section.push(h("p", { class: "info" },
      "This demo ships neighbourhoods for the top alerts only: run the full stack"), neo4jNote());
    return section;
  }
  const stepData = await data.step(tx.time_step);
  const nb = stepData.neighbourhoods[String(tx.tx_id)][String(hops)];
  const picked = h("p", { class: "note", "aria-live": "polite" }, "");
  const select = (n) => {
    picked.replaceChildren(`Transaction ${n.tx_id}: risk ${fixed(n.risk_score, 3)}, hop ${n.hop}, ` +
      `${n.flagged ? "flagged" : "not flagged"}. `, link(`#/tx/${n.tx_id}?step=${tx.time_step}`, "Open it"));
  };
  const hopLinks = h("div", { class: "segmented", role: "group", "aria-label": "Hops" },
    Array.from({ length: meta.max_hops }, (_, i) => i + 1).map((v) =>
      link(`#/tx/${tx.tx_id}?step=${tx.time_step}&hops=${v}`, `${v} hop${v > 1 ? "s" : ""}`,
        v === hops ? { "aria-current": "true" } : {})));
  section.push(hopLinks);
  if (nb.truncated) {
    section.push(h("p", { class: "warning" }, `Showing the nearest ${nb.nodes.length} transactions, ` +
      "riskiest first within each hop; the full neighbourhood is larger."));
  }
  section.push(h("figure", { class: "graph" }, networkChart(nb, select)), picked, riskLegend(),
    details("Neighbourhood (table)", table([
      { title: "Transaction", cell: (r) => link(`#/tx/${r.tx_id}?step=${tx.time_step}`, String(r.tx_id)) },
      { title: "Hop", cell: (r) => String(r.hop), num: true },
      { title: "Risk score", cell: (r) => fixed(r.risk_score, 4), num: true },
      { title: "Flagged", cell: (r) => (r.flagged ? "◆ flagged" : "—") },
    ], nb.nodes)));
  return section;
}

async function transactionView(params, query) {
  const { meta } = await data.index();
  const txId = Number(params[0] ?? 0);
  const step = query.get("step") ? Number(query.get("step")) : null;
  const hops = Math.min(Math.max(Number(query.get("hops") ?? 1), 1), meta.max_hops);
  const search = h("form", { class: "controls", onsubmit: (e) => {
    e.preventDefault();
    location.hash = `#/tx/${e.target.elements.id.value.trim()}`;
  } }, h("label", { class: "field" }, "Transaction id ",
    h("input", { name: "id", inputmode: "numeric", pattern: "[0-9]+", value: txId || "", required: "" })),
  h("button", { type: "submit" }, "Open"));
  const tx = txId ? await data.transaction(txId, step) : null;
  if (!tx) {
    return [h("h1", {}, "Transaction"), search,
      txId ? h("p", { class: "info" }, `Transaction ${txId} is not in the served period`) : null];
  }
  const tile = (label, value) => h("div", { class: "tile" }, h("div", { class: "tile-label" }, label),
    h("div", { class: "tile-value" }, value));
  return [
    h("h1", {}, "Transaction"), search,
    h("div", { class: "tiles" },
      tile("Risk score", fixed(tx.risk_score, 4)), tile("Threshold", fixed(tx.threshold, 3)),
      tile("Decision", tx.flagged ? "◆ Flagged" : "Not flagged"), tile("Label", labelText(tx.label))),
    h("p", { class: "note" }, `Time step ${tx.time_step}.`),
    h("div", { class: "columns" },
      h("section", {}, h("h2", {}, "Why this score"), h("figure", {}, shapChart(tx, leftColumnWidth())),
        h("p", { class: "note" }, `The ${tx.top_features.length} largest SHAP contributions, in log-odds. ` +
          `The model starts from ${signed(tx.base_value, 2)} and adds every feature's ` +
          "contribution; +1 multiplies the odds of illicit by about 2.7. Features are anonymised in the " +
          "dataset, so only their names and groups are known."),
        details("Contributions (table)", table([
          { title: "Feature", cell: (r) => r.feature },
          { title: "Group", cell: (r) => r.group },
          { title: "SHAP", cell: (r) => fixed(r.shap, 4), num: true },
        ], tx.top_features))),
      h("section", {}, ...(await neighbourhoodSection(tx, meta, hops)))),
  ];
}

async function patternsView(params) {
  const { meta } = await data.index();
  const t = inRange(meta, Number(params[0] ?? 40));
  const { communities } = await data.step(t);
  return [
    h("h1", {}, "Patterns in a step"),
    h("div", { class: "controls" }, stepSelect(meta, t, (s) => (location.hash = `#/patterns/${s}`))),
    h("h2", {}, "Communities with the most flagged transactions"),
    table([
      { title: "Community", cell: (r) => String(r.community_id), num: true },
      { title: "Size", cell: (r) => integer(r.size), num: true },
      { title: "Flagged", cell: (r) => integer(r.n_flagged), num: true },
      { title: "Share flagged", cell: (r) => fixed(r.share_flagged, 3), num: true },
      { title: "Mean risk", cell: (r) => fixed(r.mean_risk, 4), num: true },
    ], communities.items),
    h("p", { class: "note" }, "Louvain communities. On this data they are close to whole steps (the largest " +
      "holds most of its step), so few communities stand out."),
    h("h2", {}, "Fan-in: transactions paid by several flagged transactions"),
    h("p", { class: "info" }, "Fan-in needs the whole graph: run the full stack (Neo4j)"), neo4jNote(),
  ];
}

// --- shell -------------------------------------------------------------------------------

const ROUTES = { "": startView, alerts: alertsView, tx: transactionView, patterns: patternsView };

async function render() {
  const [path, qs = ""] = location.hash.replace(/^#\/?/, "").split("?");
  const [name, ...params] = path.split("/");
  const view = ROUTES[name] ?? startView;
  document.querySelectorAll("nav a").forEach((a) =>
    a.toggleAttribute("aria-current", a.dataset.route === (ROUTES[name] ? name : "")));
  main.setAttribute("aria-busy", "true");
  try {
    main.replaceChildren(...(await view(params, new URLSearchParams(qs))));
  } catch (err) {
    main.replaceChildren(h("h1", {}, "Something went wrong"), h("p", { class: "info" }, String(err.message)));
  }
  main.setAttribute("aria-busy", "false");
  // Switching hops redraws the same transaction: keep the reader where they are.
  const page = location.hash.replace(/[?&]hops=\d+/, "");
  if (page !== lastPage) window.scrollTo(0, 0);
  lastPage = page;
}
let lastPage = null;

async function footer() {
  const { meta } = await data.index();
  const rev = meta.dataset_revision;
  document.getElementById("model").textContent = meta.model_version;
  const ds = document.getElementById("dataset");
  ds.href = `https://huggingface.co/datasets/${meta.dataset_repo}/tree/${rev}`;
  ds.textContent = `${meta.dataset_repo} @ ${rev.slice(0, 7)}`;
}

window.addEventListener("hashchange", render);
render();
footer();
