---
title: Fraud Graph Investigator
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Fraud graph investigator

An investigator's view of illicit-transaction detection on the Elliptic Bitcoin dataset:
an alert queue per time step, a SHAP explanation for each score, and the transaction's
neighbourhood coloured by risk. It walks through where the model works and where it
stops working, after the dark-market shutdown at step 43.

Code, method and full results: https://github.com/vincentgoh0902/fraud-graph-detection

**What this demo contains.** Model outputs only: transaction IDs, risk scores, flags and
SHAP values for time steps 35–49, plus the payment edges around each step's top 20 alerts.
It contains no transaction features and no full edge list. The full stack, with Neo4j and
the whole graph, runs locally with `docker compose up`.

**Data.** [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
by Elliptic, licensed [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
The transaction IDs and payment edges shown here are unmodified excerpts; scores, flags and
SHAP values are this project's model outputs. M. Weber et al., "Anti-Money Laundering in
Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics", KDD '19
Workshop on Anomaly Detection in Finance, 2019. This is a non-commercial portfolio project.
