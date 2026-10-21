# Project Brief: Graph-Based Fraud Detection on Bitcoin Transactions

## 1. Goal

Build an end-to-end fraud detection system that classifies Bitcoin transactions as
illicit or licit, compares tabular ML against graph neural networks under a proper
temporal split, and gives an investigator a way to explore suspicious transaction
networks in Neo4j.

The project should answer one research question clearly:

> Does graph structure (GNN embeddings or graph features) improve illicit-transaction
> detection over tabular features alone, and does that improvement hold over time?

Target roles: Data Scientist, ML Engineer (fintech, risk, trust and safety).

## 2. Dataset

**Elliptic Bitcoin Dataset** (Weber et al., 2019), available on Kaggle.

- ~200k transactions (nodes), ~234k payment flows (directed edges)
- 49 time steps (~2 weeks apart); edges only connect transactions within the same step
- Labels: illicit (~2%), licit (~21%), unknown (rest)
- 166 features per node: column 1 is the time step, then 93 local features, then 72
  aggregated neighbourhood features

Before using it:
- Verify the download source and licence terms, and record them in the README.
- Do NOT commit raw data to git. Add `data/` to `.gitignore` and document the download
  step in the README (or a `scripts/download_data.py`).

Optional extension: **Elliptic++** adds wallet addresses and address-transaction edges,
which enables a heterogeneous graph. Only start this after the core project is done.

## 3. Evaluation protocol (non-negotiable)

These rules prevent the leakage problems from my previous project.

- **Temporal split:** train on time steps 1–34, test on 35–49 (the split used in the
  original paper). Use a validation set carved from the end of the training period
  (e.g. steps 30–34) for model selection and threshold tuning. Never tune on test.
- **Unknown labels:** exclude unknown nodes from the loss and metrics, but keep them in
  the graph for message passing.
- **Graph per split:** because edges stay within a time step, training-period graphs and
  test-period graphs are disjoint. Confirm this in code with an assertion.
- **Feature sets compared:** (a) local features only, (b) all features (local +
  aggregated), (c) local features + graph features/embeddings. Report each.
- **Metrics:** illicit-class precision, recall and F1; PR-AUC (primary, because of class
  imbalance); precision@K (e.g. top 100 flagged per time step, mimicking an
  investigator's review capacity). Do not lead with accuracy.
- **Report the checkpoint you actually keep:** test metrics must come from the model
  selected on validation, never from the last epoch.
- **Seeds:** run every model with at least 3 seeds and report mean ± std.
- **Per-time-step analysis:** plot illicit F1 per test time step. The original paper
  found performance collapses in later steps (a dark market shutdown); investigate and
  explain whether my models show the same distribution shift.

## 4. Models

Baselines (tabular):
1. Logistic regression
2. Random forest
3. XGBoost (or LightGBM)

Graph models (PyTorch Geometric):
4. GCN
5. GraphSAGE
6. Hybrid: GNN node embeddings concatenated with tabular features, fed to XGBoost

Graph features from Neo4j (Graph Data Science library, or NetworkX if GDS is unavailable):
- in/out degree, PageRank, clustering coefficient, community ID (Louvain), and
  neighbour illicit-rate computed using **training labels only**
- Feed these into XGBoost as an extra feature set

Imbalance handling to compare: class weights vs focal loss (GNN) vs scale_pos_weight
(XGBoost). Tune the decision threshold on validation to maximise F1, and also report a
cost-based threshold (e.g. a missed illicit transaction costs 10x a false alarm).

## 5. Why Neo4j (the graph must do real work)

Neo4j is used for two things that a dataframe cannot do well:

1. **Graph feature engineering** via GDS algorithms (section 4).
2. **Investigation queries** at serving time. Examples to implement as Cypher:
   - k-hop neighbourhood (k = 1–3) around a flagged transaction
   - paths connecting two flagged transactions
   - clusters with a high share of predicted-illicit nodes ("suspicious communities")
   - transactions that receive funds from many flagged transactions (fan-in patterns)

Store model predictions back into Neo4j as node properties (`risk_score`,
`predicted_label`, `model_version`) so investigation queries can filter on them.

## 6. Serving and UI

- **FastAPI**
  - `GET /transactions/{tx_id}`: risk score, top contributing features (SHAP), label
  - `GET /transactions/{tx_id}/neighbourhood?hops=2`: subgraph from Neo4j with risk scores
  - `GET /alerts?time_step=40&k=100`: top-K highest-risk transactions for a time step
  - Load models and precompute scores at startup; never recompute embeddings per request
- **Streamlit dashboard** for an investigator
  - Alert queue for a selected time step
  - Transaction detail with SHAP explanation
  - Interactive neighbourhood graph (e.g. pyvis or streamlit-agraph) coloured by risk
- **Measure latency:** time each endpoint over 100 requests and report p50 and p95.

## 7. Engineering standards

- Python 3.11+, dependency management with `uv` (or poetry), `pyproject.toml`
- `docker-compose.yml` with three services: `neo4j`, `api`, `dashboard`
- **MLflow** for experiment tracking (params, metrics, artefacts for every run)
- **pytest** tests for: data loading, temporal split (no leakage), feature builders,
  API endpoints
- **GitHub Actions CI**: lint (ruff), tests, and a small smoke-training run on a sample
- Configuration in YAML (`configs/`), not hard-coded constants
- Secrets (Neo4j password etc.) in `.env`, which is git-ignored; provide `.env.example`
- Commit in small, meaningful steps with descriptive messages

## 8. Suggested repo structure

```
fraud-graph-detection/
├── configs/              # YAML configs per experiment
├── data/                 # git-ignored
├── notebooks/            # EDA only, not core logic
├── scripts/              # download_data.py, load_neo4j.py, run_experiment.py
├── src/fraudgraph/
│   ├── data/             # loading, temporal split
│   ├── features/         # tabular + graph features
│   ├── models/           # baselines, gnn, hybrid
│   ├── evaluation/       # metrics, per-time-step analysis, plots
│   ├── graph/            # Neo4j client, Cypher queries
│   └── api/              # FastAPI app
├── dashboard/            # Streamlit app
├── tests/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## 9. Phases and acceptance criteria

**Phase 0 – Setup**
- Repo scaffold, pyproject, docker-compose with Neo4j running, CI passing on an empty test
- Done when: `docker compose up` starts Neo4j and `pytest` passes in CI

**Phase 1 – Data and EDA**
- Download script, loader, temporal split with leakage assertions
- EDA notebook: class balance per time step, feature distributions, graph stats
- Done when: split tests pass and EDA findings are summarised in the README

**Phase 2 – Tabular baselines**
- LR, RF, XGBoost on feature sets (a) and (b), 3 seeds, logged in MLflow
- Done when: a results table with PR-AUC, F1, precision@100 (mean ± std) exists

**Phase 3 – Neo4j graph features**
- Load nodes and edges into Neo4j, compute GDS features, add as feature set (c)
- Done when: XGBoost with graph features is evaluated and compared

**Phase 4 – GNNs and hybrid**
- GCN, GraphSAGE, hybrid embeddings + XGBoost
- Done when: all models are in one comparison table plus a per-time-step F1 plot

**Phase 5 – Analysis**
- Distribution shift analysis on later time steps, SHAP for the best model, error
  analysis on false negatives
- Done when: README has a "Findings" section with plots and honest conclusions,
  including negative results

**Phase 6 – Serving**
- FastAPI endpoints, predictions written to Neo4j, Streamlit dashboard, latency numbers
- Done when: `docker compose up` gives a working demo end to end

## 10. Rules for Claude Code

- Work one phase at a time. Propose a plan for the phase before writing code, and wait
  for my approval.
- Never invent or estimate metrics. Every number in the README must come from a logged
  MLflow run, and the README results table should be generated by a script.
- If a result is worse than expected, report it as is and help me investigate.
- Explain non-obvious design choices in short comments or in the README, because I need
  to be able to defend every decision in an interview.
- Never write secrets into code or commit `.env`.

## 11. Resume bullet targets (fill in only with real results)

- Built a fraud detection pipeline on ~200k Bitcoin transactions with a temporal split,
  comparing XGBoost and GNNs (GCN, GraphSAGE); [best model] achieved [X] PR-AUC vs [Y]
  for a tabular baseline
- Engineered graph features with Neo4j Graph Data Science (PageRank, Louvain
  communities, leakage-free neighbour risk) and showed with paired bootstrap intervals
  that neither they nor GNNs improved on tabular XGBoost (PR-AUC change [A], 95%
  interval [B]); traced the post-shutdown failure to new illicit transactions that look
  licit in both their own features and their neighbourhoods
- Diagnosed a [Z]% drop in recall after a dark market shutdown through per-time-step
  analysis, and served risk scores and investigator subgraph queries via FastAPI,
  Neo4j and Streamlit (p95 latency [N] ms)
