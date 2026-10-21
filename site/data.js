// Fetching the site's JSON. URLs are relative to the page, so the site works under the
// GitHub Pages sub-path and on a custom domain alike (D-045). Each file is fetched once.

const cache = new Map();

export function load(path) {
  if (!cache.has(path)) {
    const url = new URL(path, document.baseURI);
    cache.set(
      path,
      fetch(url).then((r) => {
        if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
        return r.json();
      }),
    );
  }
  return cache.get(path);
}

export const index = () => load("data/index.json");
export const step = (t) => load(`data/steps/${t}.json`);
export const lookup = () => load("data/lookup.json");

// Transaction detail: the step's shard, then the ServingService.transaction shape, built
// exactly as fraudgraph.site.build.expand_transaction does (tests check that function).
export async function transaction(txId, timeStep) {
  const { meta } = await index();
  if (timeStep == null) {
    const steps = await lookup();
    timeStep = Number(Object.keys(steps).find((t) => steps[t].includes(txId)));
    if (!timeStep) return null;
  }
  const shard = await load(`data/shap/${timeStep}-${txId % meta.shards_per_step}.json`);
  const entry = shard.transactions.find((e) => e.tx_id === txId);
  if (!entry) return null;
  return {
    tx_id: entry.tx_id,
    time_step: shard.time_step,
    risk_score: entry.risk_score,
    flagged: entry.flagged,
    label: null,
    threshold: meta.threshold,
    base_value: meta.base_value,
    top_features: entry.top.map(([i, v]) => ({
      feature: meta.features[i],
      group: meta.groups[i],
      shap: v,
    })),
  };
}
