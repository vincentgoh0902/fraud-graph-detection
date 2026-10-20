# fraud-graph-detection

Detecting illicit Bitcoin transactions with graph machine learning, evaluated
under a strict temporal split.

**Research question:** does graph structure (GNN embeddings or graph features)
improve illicit-transaction detection over tabular features alone, and does that
improvement hold over time?

> Status: Phase 4 (GNNs and hybrid). Results, plots and a live demo will appear here as the
> project progresses. Every number in this README will be generated from logged
> experiment runs, not typed by hand.

## Approach (planned)

- **Data:** Elliptic Bitcoin dataset (Weber et al., 2019): ~200k transactions,
  49 time steps, ~2% labelled illicit.
- **Evaluation:** train on time steps 1–29, select models and thresholds on 30–34,
  test on 35–49; PR-AUC as the primary metric because of class imbalance;
  per-time-step analysis of drift.
- **Models:** logistic regression, random forest, XGBoost, GCN, GraphSAGE, and a
  hybrid (GNN embeddings + XGBoost).
- **Graph features and investigation:** Neo4j Graph Data Science (PageRank,
  Louvain communities, neighbour risk) and Cypher queries for an investigator
  dashboard.

Design decisions and their reasoning are recorded in [`docs/decisions.md`](docs/decisions.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync                      # install dependencies
cp .env.example .env         # then set NEO4J_PASSWORD
docker compose up -d neo4j   # Neo4j browser at http://localhost:7474
uv run pytest
```

## Data

[Elliptic Bitcoin dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
from Kaggle, published by Elliptic: 203,769 transactions (nodes), 234,355 payment
flows (edges), 49 time steps about two weeks apart; 4,545 illicit and 42,019 licit
labels, the rest unknown. Features are anonymised.

**Licence:** [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
The data is not included in this repository and must be downloaded separately.

**Citation:** M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei,
T. Robinson, C. E. Leiserson, "Anti-Money Laundering in Bitcoin: Experimenting with
Graph Convolutional Networks for Financial Forensics", KDD '19 Workshop on Anomaly
Detection in Finance, 2019. Data: Elliptic, www.elliptic.co.

**Download** (needs a Kaggle API token in `~/.kaggle/`):

```bash
uv run python scripts/download_data.py   # ~150 MB download; verifies SHA-256 checksums
```

The loader (`src/fraudgraph/data/load.py`) validates every load against the published
counts and checks that no edge crosses time steps, which the temporal split relies on.

## EDA findings

From [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). All findings use the training
period (steps 1–34) except the timeline, which is shown for context only.

![Labels per time step](docs/figures/labels_per_step.png)

- **The illicit share varies a lot between steps,** from under 1% to 36% of labelled
  transactions within the training period. Illicit counts per step range from 5 to 342
  (median 84). Late in the timeline, illicit labels almost vanish (from 239 in step 42
  to between 2 and 24 in steps 43–47), consistent with the dark market shutdown reported
  by Weber et al. Phase 5 investigates this.
- **Precision@K uses K = 20.** With K = 100, 59% of training steps have fewer than 100
  illicit transactions, so even a perfect model could not reach precision 1 there
  ([D-004](docs/decisions.md)).
- **The features are already standardised** (column means ≈ 0, standard deviations ≈ 1)
  but heavily skewed: 85 of 165 have |skew| > 5. There are no constant columns, but 45
  feature pairs are near-duplicates (|r| > 0.99). This matters for logistic regression
  (unstable coefficients), not for tree models.

  ![Feature skew](docs/figures/feature_skew.png)
  ![Feature correlation](docs/figures/feature_correlation.png)
- **The graph is sparse and each time step is one connected component** (verified for
  all 34 training steps). Each step has about 4,000 transactions and 4,600 edges;
  78% of transactions have only 1 or 2 connections, and the busiest has 473.

  ![Degree distribution](docs/figures/degree_distribution.png)
- **There is strong homophily.** Between labelled transactions, illicit–illicit edges
  occur 2.7× more often than random pairing predicts. An illicit transaction's labelled
  neighbours are on average 55% illicit, a licit one's only 2%. So the graph carries
  signal, but 21% of labelled transactions have no labelled neighbour, and labels can't
  be used at prediction time anyway (D-001). Graph models have to work from features
  and predicted risk.

  ![Homophily](docs/figures/homophily.png)
  ![Example neighbourhoods](docs/figures/example_neighbourhoods.png)

## Results

### Phase 2: tabular baselines

Logistic regression, random forest and XGBoost on feature sets (a) local and
(b) local + aggregated. Each model's settings were chosen from a small grid by mean
validation PR-AUC (steps 30–34), and the decision thresholds were tuned on validation
too; test steps 35–49 were scored once, with the chosen model
([D-009](docs/decisions.md), [D-014](docs/decisions.md)). Generated by
`scripts/make_results_table.py` from the MLflow runs.

<!-- results:start -->
**Ranking quality** (test steps 35–49; precision/recall@K are per-step means)

| Model | Features | Seeds | Val PR-AUC | Test PR-AUC | Precision@20 | Recall@20 |
|---|---|---|---|---|---|---|
| Logistic regression | (a) local | 1 | 0.836 | 0.267 | 0.133 | 0.083 |
| Random forest | (a) local | 3 | 0.989 ± 0.000 | 0.781 ± 0.001 | 0.543 ± 0.000 | 0.176 ± 0.000 |
| XGBoost | (a) local | 3 | 0.988 ± 0.000 | 0.778 ± 0.003 | 0.550 ± 0.000 | 0.181 ± 0.001 |
| Logistic regression | (b) local + aggregated | 1 | 0.770 | 0.231 | 0.107 | 0.074 |
| Random forest | (b) local + aggregated | 3 | 0.990 ± 0.000 | 0.786 ± 0.004 | 0.545 ± 0.002 | 0.177 ± 0.001 |
| XGBoost | (b) local + aggregated | 3 | 0.990 ± 0.001 | 0.791 ± 0.000 | 0.553 ± 0.003 | 0.182 ± 0.002 |

**Decisions at the validation-tuned threshold** (left: max-F1 threshold; right: cost threshold, a miss costs 10× a false alarm)

| Model | Features | Seeds | Precision | Recall | F1 | Precision (cost) | Recall (cost) | F1 (cost) |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | (a) local | 1 | 0.442 | 0.641 | 0.523 | 0.199 | 0.804 | 0.319 |
| Random forest | (a) local | 3 | 0.782 ± 0.031 | 0.725 ± 0.007 | 0.752 ± 0.011 | 0.573 ± 0.022 | 0.748 ± 0.002 | 0.649 ± 0.014 |
| XGBoost | (a) local | 3 | 0.775 ± 0.012 | 0.713 ± 0.003 | 0.742 ± 0.007 | 0.459 ± 0.089 | 0.763 ± 0.018 | 0.569 ± 0.065 |
| Logistic regression | (b) local + aggregated | 1 | 0.276 | 0.706 | 0.396 | 0.177 | 0.835 | 0.292 |
| Random forest | (b) local + aggregated | 3 | 0.886 ± 0.032 | 0.710 ± 0.006 | 0.788 ± 0.014 | 0.592 ± 0.048 | 0.740 ± 0.001 | 0.657 ± 0.030 |
| XGBoost | (b) local + aggregated | 3 | 0.841 ± 0.009 | 0.731 ± 0.001 | 0.782 ± 0.003 | 0.688 ± 0.109 | 0.743 ± 0.007 | 0.711 ± 0.055 |

Mean ± sample standard deviation over seeds.
<!-- results:end -->

Reproduce with `uv run python scripts/run_experiment.py`, then
`uv run python scripts/make_results_table.py`. Browse runs with
`uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

### Phase 3: Neo4j graph features

XGBoost (the same 8-setting grid and 3 seeds as Phase 2, [D-014](docs/decisions.md)) on
three graph feature sets ([D-003](docs/decisions.md), [D-021](docs/decisions.md)):

- **(c) local + graph:** the 93 local features plus in/out degree, PageRank, clustering
  coefficient, Louvain community size and density (computed by Neo4j Graph Data Science),
  and community and neighbour risk.
- **(b) + graph:** all 165 Elliptic features plus the same graph features. Graph features
  only help if this beats (b), because the 72 aggregated features are already
  neighbourhood statistics.
- **Local + structure only:** the graph's shape without any risk feature, to separate
  structure from risk passed along edges.

Neighbour and community risk use a stage-1 XGBoost model's *predicted* risk, not labels:
test transactions only connect to other test transactions, whose labels are unknown at
prediction time. Training transactions are scored out-of-fold by time step, so no
transaction's risk comes from a model that saw its time step
([D-001](docs/decisions.md), [D-018](docs/decisions.md)). Louvain communities turned out to
cover most of a time step (median 5 per step, the largest holding 81% of the step on
average), so the community features say more about the step than about a local group
([D-023](docs/decisions.md)).

<!-- results-phase3:start -->
**Ranking quality** (test steps 35–49; precision/recall@K are per-step means)

| Model | Features | Seeds | Val PR-AUC | Test PR-AUC | Precision@20 | Recall@20 |
|---|---|---|---|---|---|---|
| XGBoost | (c) local + graph | 3 | 0.993 ± 0.001 | 0.788 ± 0.001 | 0.552 ± 0.002 | 0.180 ± 0.001 |
| XGBoost | (b) + graph | 3 | 0.994 ± 0.000 | 0.795 ± 0.001 | 0.562 ± 0.002 | 0.187 ± 0.001 |
| XGBoost | local + structure only | 3 | 0.990 ± 0.000 | 0.776 ± 0.001 | 0.550 ± 0.003 | 0.179 ± 0.002 |

**Decisions at the validation-tuned threshold** (left: max-F1 threshold; right: cost threshold, a miss costs 10× a false alarm)

| Model | Features | Seeds | Precision | Recall | F1 | Precision (cost) | Recall (cost) | F1 (cost) |
|---|---|---|---|---|---|---|---|---|
| XGBoost | (c) local + graph | 3 | 0.808 ± 0.036 | 0.718 ± 0.009 | 0.760 ± 0.011 | 0.607 ± 0.029 | 0.745 ± 0.004 | 0.669 ± 0.016 |
| XGBoost | (b) + graph | 3 | 0.893 ± 0.021 | 0.715 ± 0.005 | 0.794 ± 0.005 | 0.705 ± 0.079 | 0.737 ± 0.009 | 0.719 ± 0.039 |
| XGBoost | local + structure only | 3 | 0.769 ± 0.019 | 0.705 ± 0.002 | 0.736 ± 0.007 | 0.547 ± 0.072 | 0.741 ± 0.013 | 0.627 ± 0.045 |

Mean ± sample standard deviation over seeds.

**XGBoost: what each feature set adds** (test steps 35–49; change = difference of seed means)

| From | To | test PR-AUC: from → to (change) | precision@20: from → to (change) | F1 (max-F1 threshold): from → to (change) |
|---|---|---|---|---|
| (a) local | (b) local + aggregated | 0.778 → 0.791 (+0.013) | 0.550 → 0.553 (+0.003) | 0.742 → 0.782 (+0.039) |
| (a) local | (c) local + graph | 0.778 → 0.788 (+0.010) | 0.550 → 0.552 (+0.002) | 0.742 → 0.760 (+0.018) |
| (a) local | local + structure only | 0.778 → 0.776 (-0.002) | 0.550 → 0.550 (+0.000) | 0.742 → 0.736 (-0.007) |
| (b) local + aggregated | (b) + graph | 0.791 → 0.795 (+0.004) | 0.553 → 0.562 (+0.009) | 0.782 → 0.794 (+0.012) |

**How sure is each change?** Paired bootstrap over test steps: 1000 resamples of the 15 test steps, both feature sets scored on the same resample ([D-024](docs/decisions.md)).

| From | To | test PR-AUC change [95% interval] | precision@20 change [95% interval] | share of resamples where To has higher PR-AUC |
|---|---|---|---|---|
| (a) local | (b) local + aggregated | +0.013 [+0.003, +0.025] | +0.003 [-0.014, +0.024] | 99% |
| (a) local | (c) local + graph | +0.010 [+0.003, +0.017] | +0.002 [-0.014, +0.019] | 100% |
| (a) local | local + structure only | -0.002 [-0.007, +0.003] | +0.000 [-0.010, +0.010] | 33% |
| (b) local + aggregated | (b) + graph | +0.004 [-0.005, +0.013] | +0.009 [+0.000, +0.027] | 85% |
<!-- results-phase3:end -->

**Reading it.** On top of all Elliptic features, our graph features give no improvement
that can be told apart from zero: the bootstrap interval for the PR-AUC change includes
zero, and the precision@20 interval only just reaches it. Graph shape on its own adds
nothing. Graph information does help in one form, neighbourhood statistics: the dataset's
aggregated features clearly improve on local features, and our graph features, which
include risk averaged over neighbours, improve on local features by a similar amount
when used *instead* of them. So the graph information is real but already captured by
Elliptic's aggregated features. The intervals are approximate with only 15 test steps
([D-024](docs/decisions.md)), and whether any effect changes after the late-timeline
collapse is analysed per step in Phase 5. GNNs (Phase 4) use the graph differently, so
this result is about engineered graph features, not graph learning in general.

Reproduce with `docker compose up -d neo4j`, `uv run python scripts/load_neo4j.py`,
`uv run python scripts/build_graph_features.py` (or `--backend networkx` without Neo4j),
then `uv run python scripts/run_experiment.py --config graph_features` and
`uv run python scripts/make_results_table.py`.

### Phase 4: GNNs and hybrid

GCN and GraphSAGE (two message-passing layers each, PyTorch Geometric) on feature sets (a)
and (b), and a hybrid: XGBoost on the same features plus the embeddings of the better GNN
(chosen on validation). Everything goes through the same protocol as Phases 2 and 3: an
8-setting grid per model, 3 seeds, selection and thresholds on validation, one test score
per kept model ([D-014](docs/decisions.md), [D-027](docs/decisions.md)).

- **Graph.** Edges in both directions; unknown-label transactions are in the graph but not
  in the loss or the metrics. Each period is its own subgraph, because no edge crosses
  periods ([D-026](docs/decisions.md)).
- **Checkpoint.** The kept model is the epoch with the best validation PR-AUC, not the
  last epoch ([D-027](docs/decisions.md)).
- **Imbalance.** Class-weighted cross-entropy and focal loss are both in the grid and
  compared on validation only ([D-029](docs/decisions.md)).
- **Hybrid caveat.** The embeddings of training transactions come from a GNN trained on
  their labels, so they are cleaner than those of test transactions. That can only hurt
  the hybrid's test score; the train vs validation PR-AUC of each GNN is shown so the gap
  is visible ([D-028](docs/decisions.md)).
- **Reproducible.** GNNs train on CPU with deterministic algorithms, so a refit
  reproduces its validation score exactly, as the protocol checks
  ([D-025](docs/decisions.md)).

<!-- results-phase4:start -->
**Ranking quality** (test steps 35–49; precision/recall@K are per-step means)

| Model | Features | Seeds | Val PR-AUC | Test PR-AUC | Precision@20 | Recall@20 |
|---|---|---|---|---|---|---|
| GCN | (a) local | 3 | 0.909 ± 0.007 | 0.482 ± 0.030 | 0.431 ± 0.013 | 0.100 ± 0.002 |
| GraphSAGE | (a) local | 3 | 0.956 ± 0.003 | 0.577 ± 0.011 | 0.532 ± 0.008 | 0.173 ± 0.004 |
| GCN | (b) local + aggregated | 3 | 0.897 ± 0.009 | 0.464 ± 0.044 | 0.424 ± 0.014 | 0.129 ± 0.003 |
| GraphSAGE | (b) local + aggregated | 3 | 0.957 ± 0.008 | 0.526 ± 0.031 | 0.446 ± 0.003 | 0.131 ± 0.018 |
| XGBoost | (a) + GNN embeddings (hybrid) | 3 | 0.975 ± 0.000 | 0.673 ± 0.018 | 0.540 ± 0.012 | 0.177 ± 0.012 |
| XGBoost | (b) + GNN embeddings (hybrid) | 3 | 0.966 ± 0.002 | 0.649 ± 0.028 | 0.504 ± 0.017 | 0.160 ± 0.008 |

**Decisions at the validation-tuned threshold** (left: max-F1 threshold; right: cost threshold, a miss costs 10× a false alarm)

| Model | Features | Seeds | Precision | Recall | F1 | Precision (cost) | Recall (cost) | F1 (cost) |
|---|---|---|---|---|---|---|---|---|
| GCN | (a) local | 3 | 0.551 ± 0.032 | 0.402 ± 0.039 | 0.464 ± 0.023 | 0.261 ± 0.027 | 0.654 ± 0.057 | 0.372 ± 0.020 |
| GraphSAGE | (a) local | 3 | 0.685 ± 0.010 | 0.578 ± 0.014 | 0.627 ± 0.011 | 0.345 ± 0.014 | 0.694 ± 0.013 | 0.460 ± 0.011 |
| GCN | (b) local + aggregated | 3 | 0.588 ± 0.040 | 0.460 ± 0.039 | 0.515 ± 0.027 | 0.409 ± 0.099 | 0.624 ± 0.060 | 0.485 ± 0.061 |
| GraphSAGE | (b) local + aggregated | 3 | 0.609 ± 0.070 | 0.576 ± 0.006 | 0.591 ± 0.031 | 0.480 ± 0.047 | 0.631 ± 0.016 | 0.544 ± 0.026 |
| XGBoost | (a) + GNN embeddings (hybrid) | 3 | 0.661 ± 0.052 | 0.602 ± 0.038 | 0.628 ± 0.007 | 0.345 ± 0.009 | 0.708 ± 0.020 | 0.463 ± 0.003 |
| XGBoost | (b) + GNN embeddings (hybrid) | 3 | 0.671 ± 0.050 | 0.566 ± 0.001 | 0.614 ± 0.020 | 0.459 ± 0.094 | 0.681 ± 0.046 | 0.542 ± 0.057 |

Mean ± sample standard deviation over seeds.

**All models** (test steps 35–49; F1 at the max-F1 and the cost threshold, both tuned on validation)

| Model | Features | Seeds | Val PR-AUC | Test PR-AUC | Precision@20 | Recall@20 | F1 | F1 (cost) |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | (a) local | 1 | 0.836 | 0.267 | 0.133 | 0.083 | 0.523 | 0.319 |
| Random forest | (a) local | 3 | 0.989 ± 0.000 | 0.781 ± 0.001 | 0.543 ± 0.000 | 0.176 ± 0.000 | 0.752 ± 0.011 | 0.649 ± 0.014 |
| XGBoost | (a) local | 3 | 0.988 ± 0.000 | 0.778 ± 0.003 | 0.550 ± 0.000 | 0.181 ± 0.001 | 0.742 ± 0.007 | 0.569 ± 0.065 |
| Logistic regression | (b) local + aggregated | 1 | 0.770 | 0.231 | 0.107 | 0.074 | 0.396 | 0.292 |
| Random forest | (b) local + aggregated | 3 | 0.990 ± 0.000 | 0.786 ± 0.004 | 0.545 ± 0.002 | 0.177 ± 0.001 | 0.788 ± 0.014 | 0.657 ± 0.030 |
| XGBoost | (b) local + aggregated | 3 | 0.990 ± 0.001 | 0.791 ± 0.000 | 0.553 ± 0.003 | 0.182 ± 0.002 | 0.782 ± 0.003 | 0.711 ± 0.055 |
| XGBoost | (c) local + graph | 3 | 0.993 ± 0.001 | 0.788 ± 0.001 | 0.552 ± 0.002 | 0.180 ± 0.001 | 0.760 ± 0.011 | 0.669 ± 0.016 |
| XGBoost | local + structure only | 3 | 0.990 ± 0.000 | 0.776 ± 0.001 | 0.550 ± 0.003 | 0.179 ± 0.002 | 0.736 ± 0.007 | 0.627 ± 0.045 |
| XGBoost | (b) + graph | 3 | 0.994 ± 0.000 | 0.795 ± 0.001 | 0.562 ± 0.002 | 0.187 ± 0.001 | 0.794 ± 0.005 | 0.719 ± 0.039 |
| GCN | (a) local | 3 | 0.909 ± 0.007 | 0.482 ± 0.030 | 0.431 ± 0.013 | 0.100 ± 0.002 | 0.464 ± 0.023 | 0.372 ± 0.020 |
| GCN | (b) local + aggregated | 3 | 0.897 ± 0.009 | 0.464 ± 0.044 | 0.424 ± 0.014 | 0.129 ± 0.003 | 0.515 ± 0.027 | 0.485 ± 0.061 |
| GraphSAGE | (a) local | 3 | 0.956 ± 0.003 | 0.577 ± 0.011 | 0.532 ± 0.008 | 0.173 ± 0.004 | 0.627 ± 0.011 | 0.460 ± 0.011 |
| GraphSAGE | (b) local + aggregated | 3 | 0.957 ± 0.008 | 0.526 ± 0.031 | 0.446 ± 0.003 | 0.131 ± 0.018 | 0.591 ± 0.031 | 0.544 ± 0.026 |
| XGBoost | (a) + GNN embeddings (hybrid) | 3 | 0.975 ± 0.000 | 0.673 ± 0.018 | 0.540 ± 0.012 | 0.177 ± 0.012 | 0.628 ± 0.007 | 0.463 ± 0.003 |
| XGBoost | (b) + GNN embeddings (hybrid) | 3 | 0.966 ± 0.002 | 0.649 ± 0.028 | 0.504 ± 0.017 | 0.160 ± 0.008 | 0.614 ± 0.020 | 0.542 ± 0.057 |

**Imbalance handling: weighted cross-entropy vs focal loss** (validation only; best grid setting per loss, mean over seeds; [D-029](docs/decisions.md))

| Model | Features | Weighted CE: best val PR-AUC | Focal: best val PR-AUC | Selected |
|---|---|---|---|---|
| GCN | (a) local | 0.909 | 0.905 | weighted CE |
| GCN | (b) local + aggregated | 0.896 | 0.897 | focal |
| GraphSAGE | (a) local | 0.956 | 0.956 | weighted CE |
| GraphSAGE | (b) local + aggregated | 0.957 | 0.957 | weighted CE |

**Kept GNN checkpoints** (epoch per seed, and PR-AUC on training vs validation nodes; [D-027](docs/decisions.md), [D-028](docs/decisions.md))

| Model | Features | Seeds | Kept epoch | Train PR-AUC | Val PR-AUC |
|---|---|---|---|---|---|
| GCN | (a) local | 3 | 148, 192, 228 | 0.937 ± 0.009 | 0.909 ± 0.007 |
| GCN | (b) local + aggregated | 3 | 126, 95, 107 | 0.970 ± 0.006 | 0.897 ± 0.009 |
| GraphSAGE | (a) local | 3 | 192, 378, 181 | 0.989 ± 0.003 | 0.956 ± 0.003 |
| GraphSAGE | (b) local + aggregated | 3 | 299, 166, 337 | 0.998 ± 0.001 | 0.957 ± 0.008 |

**How sure is each change?** Paired bootstrap over test steps: 1000 resamples of the 15 test steps ([D-024](docs/decisions.md)). Best GNN by validation PR-AUC: (a) local: GraphSAGE, (b) local + aggregated: GraphSAGE.

| From | To | test PR-AUC change [95% interval] | precision@20 change [95% interval] | share of resamples where To has higher PR-AUC |
|---|---|---|---|---|
| XGBoost, (a) local | GraphSAGE, (a) local | -0.202 [-0.402, -0.067] | -0.018 [-0.046, +0.016] | 0% |
| XGBoost, (b) local + aggregated | GraphSAGE, (b) local + aggregated | -0.265 [-0.383, -0.174] | -0.107 [-0.216, -0.006] | 0% |
| XGBoost, (a) local | XGBoost, (a) + GNN embeddings (hybrid) | -0.105 [-0.211, -0.045] | -0.010 [-0.033, +0.017] | 0% |
| XGBoost, (b) local + aggregated | XGBoost, (b) + GNN embeddings (hybrid) | -0.142 [-0.247, -0.083] | -0.049 [-0.119, +0.007] | 0% |
| GCN, (a) local | GraphSAGE, (a) local | +0.095 [-0.064, +0.245] | +0.101 [+0.031, +0.193] | 87% |
| GCN, (b) local + aggregated | GraphSAGE, (b) local + aggregated | +0.062 [+0.012, +0.115] | +0.022 [-0.047, +0.084] | 100% |

![Illicit F1 per test step](docs/figures/per_step_f1.png)

<details><summary>The figure's numbers (illicit F1 per test step, mean over seeds)</summary>

| Step | Illicit | XGBoost (b) | XGBoost (b) + graph features | GCN (b) | GraphSAGE (b) | Hybrid: XGBoost (b) + embeddings |
|---|---|---|---|---|---|---|
| 35 | 182 | 0.958 | 0.964 | 0.775 | 0.841 | 0.861 |
| 36 | 33 | 0.842 | 0.850 | 0.238 | 0.397 | 0.418 |
| 37 | 40 | 0.776 | 0.761 | 0.452 | 0.566 | 0.571 |
| 38 | 111 | 0.931 | 0.935 | 0.509 | 0.683 | 0.716 |
| 39 | 81 | 0.933 | 0.908 | 0.553 | 0.584 | 0.565 |
| 40 | 112 | 0.759 | 0.763 | 0.469 | 0.546 | 0.554 |
| 41 | 116 | 0.952 | 0.919 | 0.522 | 0.604 | 0.626 |
| 42 | 239 | 0.843 | 0.871 | 0.713 | 0.750 | 0.767 |
| 43 | 24 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 44 | 24 | 0.037 | 0.044 | 0.049 | 0.047 | 0.056 |
| 45 | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 46 | 2 | 0.144 | 0.207 | 0.411 | 0.056 | 0.000 |
| 47 | 22 | 0.000 | 0.000 | 0.000 | 0.025 | 0.000 |
| 48 | 36 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 49 | 56 | 0.030 | 0.032 | 0.009 | 0.011 | 0.011 |

</details>
<!-- results-phase4:end -->

**Reading it.** This is a negative result, and a clear one.
- **GNNs lose to XGBoost.** The best GNN, GraphSAGE on local features, reaches test PR-AUC
  0.577 against XGBoost's 0.778 on the same features. On both feature sets the bootstrap
  interval for the change is well below zero.
- **The hybrid is worse than XGBoost alone.** Adding GraphSAGE embeddings lowers test
  PR-AUC from 0.791 to 0.649 on (b), and from 0.778 to 0.673 on (a). It is also lower on
  *validation* (0.966 vs 0.990 on (b)), so the protocol would never have chosen it; it is
  scored on test only because the comparison was fixed in advance
  ([D-031](docs/decisions.md)). A likely cause is the in-sample embeddings of
  [D-028](docs/decisions.md): GraphSAGE's PR-AUC on its own training nodes is 0.998 on (b)
  but 0.957 on validation, so XGBoost learns from embeddings that are cleaner than the
  ones it is scored on. That explanation fits the numbers but has not been tested.
- **The GNNs drop much more from validation to test.** GraphSAGE goes from 0.956 to 0.577,
  XGBoost from 0.988 to 0.778. The gap opens *before* the late collapse: in steps 36–41
  GraphSAGE's F1 is 0.40–0.68 where XGBoost's is 0.76–0.95. After step 42 every model
  collapses alike, which Phase 5 investigates.
- **GraphSAGE beats GCN**, clearly on (b) (+0.062 test PR-AUC, interval above zero), not
  clearly on (a). GraphSAGE keeps a separate weight for a transaction's own features while
  GCN averages them with its neighbours'. That may matter when most neighbours are
  unlabelled or licit, but it is a hypothesis, not a tested result.
- **Aggregated features do not help the GNNs** (GraphSAGE 0.577 on (a), 0.526 on (b)), which
  is plausible because a GNN computes its own neighbourhood aggregates. This pair was not
  in the planned bootstrap comparisons, so treat it as an observation.
- **Focal loss vs weighted cross-entropy makes no difference** on validation: the best
  setting of each is within 0.004.

Together with Phase 3, the answer to the research question so far is no. On Elliptic's
temporal split, neither engineered graph features nor graph neural networks improve on
XGBoost with the dataset's own features. The original Elliptic paper (Weber et al., 2019)
also found random forest ahead of GCN. One limitation: one GCN setting was still improving
at the 500-epoch cap ([D-032](docs/decisions.md)). That affects only which GCN setting
wins, not GCN's place behind XGBoost.

Reproduce with `uv run python scripts/run_experiment.py --config gnn`, then
`--config hybrid` (a separate process: torch and XGBoost cannot share one,
[D-030](docs/decisions.md)), then `uv run python scripts/make_results_table.py`.
