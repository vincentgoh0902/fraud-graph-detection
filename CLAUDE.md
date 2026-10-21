# fraud-graph-detection

Graph-based illicit-transaction detection on the Elliptic Bitcoin dataset.
Research question: does graph structure improve detection over tabular features,
and does the improvement hold over time?

- Spec: `PROJECT_BRIEF.md` (authoritative for scope, evaluation protocol, phases)
- Design decisions and their justification: `docs/decisions.md`
- Scope: Elliptic only. The IBM AML and Elliptic++ extensions are out of scope (D-007).
- Visibility: public GitHub repo. Live demo: static site on GitHub Pages
  (https://vincentgoh0902.github.io/fraud-graph-detection/, D-043 to D-045), built by
  `.github/workflows/pages.yml` from the public HF dataset repo
  `vincentgoh-kaiswee/fraud-graph-detection-demo` at the revision pinned in
  `configs/serving.yaml`. Optional: the Streamlit dashboard on Community Cloud (D-042). Full stack runs locally via docker compose.
  Not hosted publicly on the homelab (D-008).

## Stack
Python 3.12, uv, pandas, scikit-learn, XGBoost, PyTorch Geometric (training only),
Neo4j 5 Community + GDS plugin, MLflow (local store), FastAPI, Streamlit, pytest, ruff.

## Ports and resources
- API: `PORT` env var, default 8000. Dashboard: `PORT`, default 8501.
- Neo4j: 7474 (browser) / 7687 (bolt), local compose only, bound to 127.0.0.1.
- Memory (measured in Phase 6, `docker stats`): Neo4j ~2.1 GB, API ~300 MB, dashboard
  ~50 MB calling the API; the in-process demo needs ~300 MB once its data is loaded. Host ports overridable via API_HOST_PORT /
  DASHBOARD_HOST_PORT.
- Serving images are torch-free: embeddings, scores and SHAP values are precomputed offline.

## Commands
- `uv sync` — install deps
- `uv run pytest` — tests
- `uv run ruff check . && uv run ruff format --check .` — lint
- `docker compose up -d neo4j` — start Neo4j (needs `.env`, see `.env.example`)

## Working agreement
- One phase at a time. Propose the plan for a phase, wait for approval, then build.
- Claude writes all code; the user does not write code. The user's goal is to
  understand and defend every piece, so explain each file: what it does, why, and
  the non-obvious lines. Write tests first for evaluation-critical code.
- One branch and pull request per phase; merge to `main` only when CI is green.
- End each phase with a short "what you should be able to explain" checklist.
- Every non-obvious choice gets an entry in `docs/decisions.md`.

## Evaluation rules (summary; full version in the brief)
- Temporal split: train 1–34 (validation 30–34), test 35–49. Never tune on test.
- Unknown-label nodes: excluded from loss/metrics, kept in the graph.
- Primary metric PR-AUC; also illicit P/R/F1, precision@K and recall@K per step.
- Test metrics come from the validation-selected checkpoint. ≥3 seeds for
  stochastic models, reported as mean ± std.
- No feature may use labels that would be unavailable at prediction time.

## Hard rules
- Never invent or estimate metrics. README numbers come from MLflow via a script.
- Report bad or negative results as they are.
- Never commit `.env`, data, MLflow runs or model files. Run `git status` before commits.
- No Claude attribution in commits or PRs (no `Co-Authored-By: Claude`, no
  "Generated with Claude Code").
