---
license: cc-by-nc-nd-4.0
pretty_name: Elliptic fraud-graph demo subset (model outputs)
tags:
  - fraud-detection
  - bitcoin
  - graph
  - elliptic
size_categories:
  - 10K<n<100K
---

# Elliptic fraud-graph demo subset

The data behind the public demo of
[fraud-graph-detection](https://github.com/vincentgoh0902/fraud-graph-detection): an
investigator dashboard for illicit-transaction detection on the Elliptic Bitcoin dataset.

## Contents

| File | What it holds |
|---|---|
| `transactions.parquet` | Every transaction in time steps 35–49: ID, time step, risk score, flag at the model's validation threshold, Louvain community |
| `shap.parquet` | The 10 largest SHAP contributions per transaction (feature name, group, value in log-odds) |
| `edges.parquet` | Payment edges inside the 2-hop neighbourhoods of each step's top 20 alerts |
| `per_step.csv` | The served model's logged test metrics per time step |
| `meta.json` | Model version, threshold, SHAP base value, and what this export contains |

Scores, flags, communities and SHAP values are outputs of this project's model (XGBoost,
seed 0, selected on validation). Transaction IDs, time steps and payment edges are
unmodified excerpts of the Elliptic Data Set. **Not included:** transaction features,
labels and the full edge list.

## Licence and attribution

The source data is the [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
by Elliptic, licensed [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
This subset is shared under the same licence, for non-commercial use only.

M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson, C. E. Leiserson,
"Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for
Financial Forensics", KDD '19 Workshop on Anomaly Detection in Finance, 2019.
