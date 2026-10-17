# fraud-graph-detection

Detecting illicit Bitcoin transactions with graph machine learning, evaluated
under a strict temporal split.

**Research question:** does graph structure (GNN embeddings or graph features)
improve illicit-transaction detection over tabular features alone, and does that
improvement hold over time?

> Status: Phase 0 (setup). Results, plots and a live demo will appear here as the
> project progresses. Every number in this README will be generated from logged
> experiment runs, not typed by hand.

## Approach (planned)

- **Data:** Elliptic Bitcoin dataset (Weber et al., 2019): ~200k transactions,
  49 time steps, ~2% labelled illicit.
- **Evaluation:** train on time steps 1–34, test on 35–49; PR-AUC as the primary
  metric because of class imbalance; per-time-step analysis of drift.
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
