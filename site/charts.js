// SVG charts in the dashboard's validated palette (dashboard/views.py, D-045). Colour never
// carries meaning alone: values are written on the bars, flagged nodes change shape, and
// every chart has a table view next to it. Elements are built with DOM APIs, never from
// HTML strings, so no data is ever parsed as markup.

import { fixed, signed } from "./format.js";

export const INK = "#0b0b0b";
export const MUTED = "#52514e";
export const SURFACE = "#fcfcfb";
export const GRID = "#e4e3df";
// Risk bands, light to dark: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1 (ordinal check passed).
export const RISK_EDGES = [0.2, 0.4, 0.6, 0.8];
export const RISK_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"];
export const RISK_LABELS = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1"];
export const TOWARDS_ILLICIT = "#e34948";
export const TOWARDS_LICIT = "#2a78d6";
const SHUTDOWN = 42.5; // between steps 42 and 43 (Weber et al., 2019)
const NS = "http://www.w3.org/2000/svg";

export const riskBand = (s) => RISK_EDGES.filter((e) => s >= e).length;

function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const c of children) node.append(c);
  return node;
}

const title = (text) => svg("title", {}, text);

function root(width, height, label) {
  return svg("svg", {
    viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": label,
    width: "100%", preserveAspectRatio: "xMidYMid meet",
  });
}

// Illicit F1 per test step, with the dark-market shutdown marked.
// Charts are drawn at the width they will get on screen, so text stays at its real size on
// a phone instead of shrinking with the SVG.
export function f1Chart(steps, width = 720) {
  const W = Math.max(320, Math.min(width, 820)), H = 280, L = 44, R = 16, T = 26, B = 40;
  const x = (t) => L + ((t - 35) / 14) * (W - L - R);
  const y = (v) => T + (1 - v) * (H - T - B);
  const s = root(W, H, "Illicit F1 per time step, 35 to 49, falling to near zero after step 42");
  for (let v = 0; v <= 1.0001; v += 0.2) {
    s.append(svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: GRID }));
    s.append(svg("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end", class: "tick" }, fixed(v, 1)));
  }
  for (let t = 35; t <= 49; t++) {
    s.append(svg("text", { x: x(t), y: H - B + 18, "text-anchor": "middle", class: "tick" }, String(t)));
  }
  s.append(svg("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", class: "axis" }, "Time step"));
  s.append(svg("text", { x: 12, y: T + (H - T - B) / 2, class: "axis",
    transform: `rotate(-90 12 ${T + (H - T - B) / 2})`, "text-anchor": "middle" }, "Illicit F1"));
  s.append(svg("line", { x1: x(SHUTDOWN), x2: x(SHUTDOWN), y1: T - 8, y2: H - B,
    stroke: MUTED, "stroke-dasharray": "4 4" }));
  // Left of the line: the right of it is too narrow on a phone.
  s.append(svg("text", { x: x(SHUTDOWN) - 6, y: T - 12, "text-anchor": "end", class: "note" },
    "dark-market shutdown"));
  const pts = steps.filter((d) => d.f1 !== null);
  s.append(svg("path", { d: pts.map((d, i) => `${i ? "L" : "M"}${x(d.time_step)},${y(d.f1)}`).join(""),
    fill: "none", stroke: TOWARDS_LICIT, "stroke-width": 2 }));
  for (const d of pts) {
    s.append(svg("circle", { cx: x(d.time_step), cy: y(d.f1), r: 5, fill: TOWARDS_LICIT,
      stroke: SURFACE, "stroke-width": 2 },
      title(`Step ${d.time_step}: F1 ${fixed(d.f1, 2)}, ${d.n_illicit} illicit (labelled)`)));
  }
  return s;
}

// Top SHAP contributions, signed, with each value written past the end of its bar.
export function shapChart(tx, width = 560) {
  const rows = tx.top_features;
  const W = Math.max(320, Math.min(width, 640)), L = W < 420 ? 128 : 150, R = 48, T = 34, ROW = 28;
  const H = T + rows.length * ROW + 34;
  const vals = rows.map((r) => r.shap);
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  // Room past each bar end for its written value.
  const pad = (hi - lo || 1) * 0.2;
  const d0 = lo < 0 ? lo - pad : 0, d1 = hi > 0 ? hi + pad : 0;
  const x = (v) => L + ((v - d0) / (d1 - d0 || 1)) * (W - L - R);
  const s = root(W, H, "SHAP contributions of the top features, in log-odds");
  s.classList.add("shap");
  const legend = [["towards illicit", TOWARDS_ILLICIT], ["towards licit", TOWARDS_LICIT]];
  legend.forEach(([text, color], i) => {
    s.append(svg("rect", { x: L + i * 140, y: 8, width: 12, height: 12, rx: 2, fill: color }));
    s.append(svg("text", { x: L + i * 140 + 18, y: 18, class: "legend" }, text));
  });
  s.append(svg("line", { x1: x(0), x2: x(0), y1: T - 4, y2: H - 30, stroke: GRID }));
  rows.forEach((r, i) => {
    const cy = T + i * ROW + ROW / 2;
    const color = r.shap > 0 ? TOWARDS_ILLICIT : TOWARDS_LICIT;
    s.append(svg("text", { x: L - 8, y: cy + 4, "text-anchor": "end", class: "label" }, r.feature));
    const x0 = Math.min(x(0), x(r.shap)), w = Math.max(Math.abs(x(r.shap) - x(0)), 1);
    s.append(svg("rect", { x: x0, y: cy - 6, width: w, height: 12, rx: 3, fill: color },
      title(`${r.feature} (${r.group}): ${signed(r.shap, 3)}`)));
    const pos = r.shap > 0;
    s.append(svg("text", { x: x(r.shap) + (pos ? 4 : -4), y: cy + 4,
      "text-anchor": pos ? "start" : "end", class: "value" }, signed(r.shap, 2)));
  });
  s.append(svg("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle", class: "axis" },
    "SHAP value (log-odds)"));
  return s;
}

function star(cx, cy, r) {
  const pts = [];
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5, rr = i % 2 ? r * 0.45 : r;
    pts.push(`${cx + rr * Math.cos(a)},${cy + rr * Math.sin(a)}`);
  }
  return pts.join(" ");
}

// The neighbourhood with build-time positions (D-045). onSelect(node) on tap or Enter.
export function networkChart(nb, onSelect) {
  const S = 1000;
  const s = root(S, S, `Payment neighbourhood of transaction ${nb.tx_id}: ${nb.nodes.length} transactions`);
  s.classList.add("network");
  s.append(svg("defs", {}, svg("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 22, refY: 5,
    markerWidth: 8, markerHeight: 8, orient: "auto-start-reverse" },
    svg("path", { d: "M0,0 L10,5 L0,10 z", fill: "#b9b8b2" }))));
  const pos = new Map(nb.nodes.map((n, i) => [n.tx_id, nb.positions[i]]));
  for (const e of nb.edges) {
    const [x1, y1] = pos.get(e.src), [x2, y2] = pos.get(e.dst);
    s.append(svg("line", { x1, y1, x2, y2, stroke: "#c9c8c2", "stroke-width": 2,
      "marker-end": "url(#arrow)" }));
  }
  nb.nodes.forEach((n, i) => {
    const [cx, cy] = nb.positions[i];
    const focal = n.tx_id === nb.tx_id;
    const fill = RISK_COLORS[riskBand(n.risk_score)];
    const g = svg("g", { class: "node", tabindex: 0, role: "button",
      "aria-label": `Transaction ${n.tx_id}, risk ${fixed(n.risk_score, 3)}, ${n.flagged ? "flagged" : "not flagged"}, hop ${n.hop}` });
    const shape = focal
      ? svg("polygon", { points: star(cx, cy, 30), fill, stroke: INK, "stroke-width": 3 })
      : n.flagged
        ? svg("polygon", { points: `${cx},${cy - 17} ${cx + 17},${cy} ${cx},${cy + 17} ${cx - 17},${cy}`,
          fill, stroke: INK, "stroke-width": 1.5 })
        : svg("circle", { cx, cy, r: 10, fill, stroke: INK, "stroke-width": 1.5 });
    g.append(shape, title(`Transaction ${n.tx_id}\nRisk ${fixed(n.risk_score, 3)}\nHop ${n.hop}\n` +
      `${n.flagged ? "Flagged" : "Not flagged"}`));
    if (focal || n.flagged) {
      g.append(svg("text", { x: cx, y: cy + (focal ? 52 : 40), "text-anchor": "middle",
        class: "node-label" }, String(n.tx_id)));
    }
    const pick = () => onSelect(n);
    g.addEventListener("click", pick);
    g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pick(); } });
    s.append(g);
  });
  return s;
}
