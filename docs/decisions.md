# Decision log

One entry per non-obvious choice: what we decided and why, so every decision can
be defended. Entries are never deleted; a reversed decision gets a new entry.

## D-001: Neighbour risk uses out-of-fold predictions, not training labels
**Context.** Elliptic edges only connect transactions within the same time step,
so train-period and test-period graphs are disjoint. A test node's neighbours are
all test nodes, whose labels are unknown at prediction time.
**Problem.** "Neighbour illicit-rate from training labels" is informative in
training but always 0 or missing at test. The model learns to depend on a signal
that disappears at test time: a train/serve skew, a form of leakage.
**Decision.** Two-stage stacking. Stage 1: a tabular XGBoost model produces
predicted risk for every node (including unknowns), using GroupKFold grouped by
time step on steps 1–34. Because graphs are disjoint per step, a node and its whole
neighbourhood fall in the same fold, so every training node's neighbour-risk is
truly out-of-fold. For test nodes, stage 1 is trained on all of 1–34. Stage 2
uses mean/max neighbour predicted risk as features.
**Why this is valid.** The feature is computed the same way at train and test
time, from information that exists at prediction time.
**Known limitation.** Out-of-fold models see slightly less data than the full
stage-1 model, so the feature's distribution may differ a little between train
and test. We'll check this by comparing the distributions.
**Verification.** A test asserts no node's neighbour-risk was computed by a model
that trained on that node's time step. EDA compares feature distributions.

## D-002: Louvain community properties, not community IDs
**Problem.** Louvain IDs are arbitrary integers with no meaning across runs or
time steps. As a feature they're meaningless numbers that tree models will still
split on.
**Decision.** Use community-level properties: size, internal edge density, and
mean stage-1 predicted risk of members (computed as in D-001).
**Verification.** No raw ID column appears in any feature matrix (feature-builder test).

## D-003: Feature set (c) is compared against (b), not only (a)
**Context.** The 72 "aggregated" Elliptic features are already one-hop
neighbourhood statistics, so (a) vs (b) is already a graph-information test.
**Decision.** Report three comparisons: (a)→(b) (value of the provided neighbourhood
aggregates), (a)→(c) (value of our graph features in place of them), and
(b)→(b+graph) (value added on top). Graph features only "help" if they beat (b).
**Why.** Otherwise we'd credit Neo4j with information the dataset already provides.

## D-004: Precision@K uses K chosen from the data, plus recall@K
**Problem.** Some test steps have fewer illicit transactions than 100, so
precision@100 has a low maximum and is misleading.
**Decision.** After EDA (Phase 1), pick K from the per-step illicit counts in the
*training* period (likely 20 or 50), fixed before any test evaluation. Report
recall@K and the per-step illicit count alongside it, so the reader sees the maximum
possible score.

## D-005: Validation uncertainty is reported
**Problem.** Steps 30–34 contain relatively few illicit examples, so a threshold
tuned there is noisy.
**Decision.** Bootstrap validation nodes to get a confidence interval for the tuned
threshold and validation F1. Report it next to test results.

## D-006: Seeds only for stochastic models
Logistic regression (lbfgs) is deterministic, so it runs once and reports no std.
RF, XGBoost and the GNNs run with ≥3 seeds and report mean ± std.

## D-007: Elliptic only; no second dataset
**Considered.** IBM AML (synthetic, readable features, labelled laundering
patterns) and Elliptic++ (wallet addresses, heterogeneous graph).
**Decision.** Elliptic only.
**Why.** The research question needs real data with real temporal drift, and only
Elliptic has a documented drift event (the dark market shutdown) and published
baselines to compare with. Synthetic data lets models learn simulator quirks. A
second dataset would roughly double the work without strengthening the main answer;
depth on one dataset (rolling retraining, calibration, drift analysis) is worth more.

## D-008: Showcase via README + Hugging Face Space, not public homelab hosting
**Decision.** The GitHub README carries the results (generated table, per-step F1
plot, findings, dashboard GIF). A Hugging Face Space serves a lightweight demo from
precomputed artefacts. The full stack (Neo4j + API + dashboard) runs locally with
`docker compose up`.
**Why.** Recruiters spend about a minute and rarely run code, so the README matters
most. A public Neo4j-backed app on a personal server adds attack surface and uptime
work for little extra visibility. To support both modes, graph access goes through
one interface with two implementations: Neo4j (full stack) and in-memory from
exported files (demo).

## D-009: Validation on steps 30–34, no refit
**Decision.** Train on steps 1–29, select models and tune thresholds on 30–34, and
evaluate *that same model* on test steps 35–49. No retraining on 1–34 afterwards.
**Why.** A threshold tuned on validation is only valid for the model it was tuned
on. Refitting on 1–34 would produce a different model with different score
distributions, so the tuned threshold might no longer fit. This also keeps to the
rule "report the checkpoint you actually keep".
**Cost.** The model never sees the five most recent training steps. The value of
recent data is measured separately by the rolling-retraining experiment (Phase 5).

## D-010: Decisions use training steps only
Every data-driven choice (K for precision@K, scaling, feature selection) uses
training steps only. EDA may show the full 49-step timeline for context, marked as
such, but nothing in it drives a decision. Otherwise, information about the test
period would leak into the design.

## D-011: Label encoding and data validation
**Decision.** Raw labels "1" (illicit), "2" (licit), "unknown" are mapped once in
the loader to 1, 0, −1, so illicit is the positive class everywhere. Every load
validates: unique tx_ids, no missing features, every edge endpoint exists, no edge
crosses time steps, and counts equal the published figures (203,769 nodes, 234,355
edges, 4,545 illicit, 42,019 licit).
**Why.** Treating "2" as positive would invert every metric without any error.
The same-step edge check is the factual basis of D-001, so it is enforced in
code, not taken from the paper.

## D-012: Licence (CC BY-NC-ND 4.0) and what the public demo may contain
**Context.** The dataset is licensed Attribution-NonCommercial-NoDerivatives.
**Decision.** Raw data is never committed or redistributed; the README links to the
Kaggle source and cites Elliptic and Weber et al. (2019). The Hugging Face demo ships
only model outputs (transaction IDs, risk scores, SHAP feature names and values) and
small subgraphs (IDs and edges), never the feature table itself.
**Why.** A personal portfolio project is non-commercial, but re-publishing the
features (even reformatted) could count as redistributing a derivative. This is a
cautious reading, not legal advice.

## D-013: time_step is not a model feature
**Problem.** Test steps 35–49 never appear in training. A tree model would treat
every test transaction as "time ≥ 29", and any pattern it learned from time in
training (for example "illicit was common in step 20") cannot transfer to the future.
**Decision.** time_step is used only to split and to group results, never as an input.

## D-004 update: K = 20 (Phase 1 EDA)
**Evidence (training period, steps 1–34; notebooks/01_eda.ipynb, section 2).** Illicit
transactions per step: median 84, min 5, max 342. Share of steps with fewer than K
illicit, where precision@K is capped below 1 even for a perfect model: K=10: 6%,
K=20: 21%, K=50: 35%, K=100: 59%.
**Decision.** Primary metric precision@20 per step: an investigator reviewing 20
alerts per two-week step. K=100 (the brief's example) would be capped in most steps,
so it would mostly measure how few illicit transactions there are. Alongside it we
report recall@20, the per-step illicit count (so the cap is visible), and precision
at K ∈ {10, 50, 100} as a robustness check. K was fixed before any model was evaluated.

## D-014: Tuning budget and selection rule (Phase 2 onwards)
**Decision.** Each model gets a small fixed grid (logistic regression 6 settings,
random forest 8, XGBoost 8; `configs/baselines.yaml`). Every setting runs with every
seed; the setting with the best *mean validation PR-AUC* wins. The winner is then
refit per seed and only those refits are scored on test. Exact ties keep the setting
listed first, so grids list simpler settings first.
**Budget across phases.** The budget does not grow for later models. Phase 3's
XGBoost on feature set (c) reuses this exact XGBoost grid, so any difference comes
from the features alone. Phase 4's GNNs get a grid of comparable size (about 8
settings). If any model's budget is ever raised, the baselines are re-run with the
same increase, because a tuned graph model against an untuned baseline would
measure tuning effort, not graph structure.
**Why mean over seeds.** Selecting on one seed rewards configurations that got a
lucky seed on a validation set with few illicit examples (D-005).
**Refit check.** Refitting the winner (instead of keeping every candidate model in
memory) is only valid if the refit is the same model, so the protocol asserts that
each refit reproduces its selection-time validation PR-AUC, and raises otherwise.
**Amendment (after the first run).** Logistic regression's grid was originally
C ∈ {0.01, 0.1, 1, 10}; C = 10 won and validation PR-AUC was still rising steeply
(local: 0.38, 0.42, 0.56, 0.78). An optimum on the grid edge means the grid was too
narrow, so it was widened to add 100 and 1000 (6 settings). This was decided after
test numbers had been seen; the trigger and the choice used validation only. A
check beyond the grid (C = 10⁴, 10⁶) showed validation PR-AUC flat and noisy
(local 0.80–0.84, all 0.77) with lbfgs converged, so C = 1000 is effectively
unregularised and widening further would not help. The very large coefficients
(|coef| up to ~200) reflect the near-duplicate feature pairs found in EDA.
**Not done.** Random or Bayesian search over large spaces. A larger search would mostly
fit noise in the 30–34 validation steps.

## D-015: Precision@K counts labelled nodes only; ties scored by expectation
**Labelled only.** Top-K is taken among *labelled* test transactions in each step,
because unknown ones cannot be scored right or wrong. A real investigator's queue
would also contain unknowns, so absolute precision@K is optimistic relative to
deployment; comparisons between models are still fair, since every model is
measured the same way.
**Ties.** When several transactions share the score at the K-th place (common with
random forest probabilities), hits are counted as their expected value under random
tie-breaking. A plain sort would let row order decide who is reviewed.
**Averaging.** Per-step values are averaged over test steps (macro), so each
two-week batch of review work counts equally; steps with no illicit transactions
are skipped for recall (undefined) but kept for precision.

## D-016: PR-AUC is computed as average precision
Average precision (scikit-learn's `average_precision_score`) sums precision at each
recall step. The trapezoidal rule interpolates linearly between precision-recall
points, which is not achievable by any threshold and overstates the area. Average
precision is also what most papers mean by PR-AUC in fraud detection.

## D-017: pyarrow pinned below 26 for MLflow
MLflow 3.x requires `pyarrow<26`. The Phase 1 lower bound (`>=26`) was only what
`uv add` wrote, not a real requirement; pyarrow is used for Parquet and fast CSV
reading, which 25 handles. Loosened to `pyarrow>=24`.

## D-001 update: stage-1 folds cover steps 1–29, not 1–34 (Phase 3)
D-001 was written before D-009 moved training to steps 1–29. The scheme now is:
train-period nodes (1–29) are scored out-of-fold, by a stage-1 model trained on the
other folds (GroupKFold, 5 folds, grouped by time step); validation (30–34) and test
(35–49) nodes are scored by one stage-1 model trained on all labelled nodes in 1–29.
That is the same model the protocol keeps (no refit, D-009), so validation and test
features are built the same way. Unknown-label nodes are scored too, because they are
neighbours of labelled ones. Fold models trained on later steps do score earlier
training steps; that uses no validation or test information, only other training data.

## D-018: Stage-1 uses the same base features as the comparison
**Decision.** For feature set (c) (local + graph), the stage-1 model that produces
predicted risk uses local features only. For (b) + graph it uses all features. Its
settings are the Phase 2 XGBoost winner for that feature set, read from MLflow, with one
fixed seed (0).
**Why.** If (c)'s stage-1 used the aggregated features, the neighbour-risk columns would
carry them into (c), and the (a)→(c) comparison would credit our graph features with
information from the 72 aggregated ones. Reusing the Phase 2 winner means stage-1 needs
no tuning of its own.
**Limitation.** A single stage-1 seed makes the graph features one fixed table, so the
seed spread reported for stage 2 does not include variation from stage 1.

## D-019: Neo4j GDS computes the features; NetworkX/NumPy cross-check them
**Decision.** Structural features (degree, PageRank, clustering, Louvain) come from Neo4j
GDS and are saved to `data/processed/graph_features.parquet`, so experiments run without
Neo4j. A second implementation (NetworkX, plus a small NumPy PageRank that uses the same
formula as GDS) computes the same features. CI has no Neo4j, so CI tests use the second
implementation; a local integration test checks that the two agree on the real graph.
**Why.** The brief wants Neo4j doing real work, but results that can only be reproduced
with a running database are fragile. Two implementations that agree are also a check on
each other. This is the first piece of the two-backend graph interface in D-008.
**PageRank detail.** GDS uses the unnormalised form PR = (1 − d) + d · Σ PR(in)/outdeg(in)
and does not redistribute the rank of nodes without outgoing edges. NetworkX's
`pagerank` normalises over the whole graph and spreads that rank over *all* nodes, which
would couple different time steps slightly. The NumPy version follows GDS, so the scores
of one time step never depend on another.
**No labels in Neo4j.** Only `tx_id`, `time_step` and edges are loaded, so no graph
algorithm can see a label.

## D-020: Louvain runs single-threaded
GDS Louvain can return different communities between runs when it uses several threads.
It runs with `concurrency: 1` so the features are reproducible; the integration test runs
it twice and compares. The NetworkX fallback uses a fixed seed. The two implementations
are not expected to find identical communities, so they are compared only loosely.

## D-021: Structure-only ablation
**Decision.** Besides (c) = local + all graph features and (b) + graph, run local +
structure only: degree, PageRank, clustering, community size and density, without any
feature built from stage-1 predicted risk.
**Why.** Neighbour risk and community risk feed one model's predictions into another,
which is a form of stacking. If (c) beats (a), the ablation shows how much of the gain
comes from the graph's shape and how much from passing predicted risk along edges.

## D-022: No random forest in Phase 3
**Decision.** Phase 3 trains only XGBoost on the graph feature sets. Random forest is not
re-run with graph features. (Decided by the project owner when approving the Phase 3 plan.)
**Why.**
- The brief's Phase 3 criterion is "XGBoost with graph features is evaluated and compared".
- One model keeps the question about features: every Phase 3 difference against Phase 2
  XGBoost comes from the feature set, at the same grid and seeds (D-014).
- In Phase 2, random forest and XGBoost were within 0.005 test PR-AUC of each other on
  both feature sets (0.781 vs 0.778 on (a), 0.786 vs 0.791 on (b)), so a second tree
  ensemble would likely repeat XGBoost's answer while doubling the runs.
- XGBoost is also the model Phase 4's hybrid (GNN embeddings + XGBoost) builds on, so it
  is the comparison that carries forward.
**Cost.** We cannot say whether random forest would gain more or less from graph features
than XGBoost. If that question matters later, it can be run with the same config.
