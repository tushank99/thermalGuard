# ThermalGuard: Build Plan (free tier, start small and scale)

Predictive maintenance for industrial assets: it predicts Remaining Useful Life (RUL) from sensor
degradation data and answers maintenance questions from manuals and logs.

## Guiding principles
- **Free tier only.** Every component has a $0 path. Paid or trial services (Snowflake) are used
  inside one focused window, and screenshots are recorded.
- **Local first, cloud second.** The same code runs locally (DuckDB, local MLflow) and in the cloud
  (Snowflake, Lambda). We switch with a config target, not a rewrite.
- **Thin tests that matter.** dbt data tests gate training, plus one API smoke test in CI. No
  unit-test sprawl.

## Architecture

```
NASA C-MAPSS raw .txt
   │  scripts/ingest.py  (--target duckdb | snowflake)
   ▼
RAW schema ──► dbt (staging → marts: labels + rolling features) ──► dbt tests (ranges/nulls/schema)
   │                                                                   │ fail = block training
   ▼                                                                   ▼
Training (sklearn baseline → XGBoost → PyTorch LSTM) ──► MLflow tracking + registry (@champion)
   │                                                   │
   │ export ONNX                                       └─► model cards (governance/)
   ▼
FastAPI + Mangum ──► AWS Lambda (zip, onnxruntime) ◄── Terraform ◄── GitHub Actions
   ▲                                    ▲
Streamlit dashboard            Evidently drift report ──► retrain workflow (scheduled / on drift)

LangGraph RAG (manuals + maintenance logs) ──► /ask endpoint + Streamlit tab ──► RAGAS eval
```

## Free-tier choices

| Need | Choice | Why it is free |
|---|---|---|
| Warehouse (dev) | DuckDB + dbt-duckdb | local, forever free |
| Warehouse (showcase) | Snowflake 30-day trial + dbt-snowflake | same dbt models, `--target snowflake` |
| Experiment tracking + registry | MLflow (local SQLite) → DagsHub-hosted MLflow | DagsHub free tier gives a remote registry CI can reach |
| Serving | FastAPI + Mangum on Lambda, **ONNX Runtime** zip | no ECR or container needed; fits within the Lambda always-free 1M req/mo |
| IaC | Terraform, local state → S3 backend | S3 cost is a few KB |
| CI/CD | GitHub Actions (public repo) | unlimited minutes for public repos |
| Drift | Evidently OSS | static HTML reports |
| LLM | Groq or Gemini free-tier API | your API key, when we reach Phase 5 |
| Embeddings + vector DB | sentence-transformers (local) + Chroma/FAISS | local |
| Dashboard | Streamlit Community Cloud | free hosting |

## Phases

| # | Phase | Output | Status |
|---|---|---|---|
| 1 | **Data foundation**: ingest, dbt on DuckDB, data-quality tests | `warehouse.duckdb`, `dbt build` passes (29 checks) | ✅ done |
| 2 | **Models v1**: Ridge baseline + XGBoost with MLflow tracking + registry | Ridge 20.65 → XGBoost 17.88 RMSE, registered | ✅ done |
| 3 | **LSTM + governance**: PyTorch LSTM, ONNX export, model cards, lineage notes | LSTM RMSE 13.27 (−35.7%), ONNX, model cards | ✅ done |
| 4 | **Serving**: FastAPI local → Lambda via Terraform → GitHub Actions deploy | FastAPI app, Lambda zip build, Terraform, deploy workflow | ✅ done |
| 5 | **GenAI**: LangGraph RAG over manuals and synthetic maintenance logs, RAGAS eval | assistant + RAGAS eval script | ✅ done |
| 6 | **MLOps loop**: Evidently drift report + scheduled retrain workflow + champion promotion | drift HTML, retrain Action | ⏳ |
| 7 | **UI**: Streamlit operator dashboard (fleet health, RUL, drift, assistant) | Streamlit Cloud URL | ⏳ |
| 8 | **Snowflake sprint** (start the trial only when phases 1–3 are stable): same pipeline on Snowflake, screenshots | `docs/snowflake/*.png` | ⏳ needs trial |
| T1 | **PySpark + Delta lakehouse** (Docker, free): bronze → silver (+quarantine) → gold rolling features (5/10/30), schema enforcement, time travel, data dictionary, DQ report, Spark-vs-pandas benchmark; training reads gold | 266,363 bronze → 265,256 silver (+400 quarantined) → 265,256 gold × 109 cols | ✅ done |
| T2 | **Unsupervised anomaly detection**: Isolation Forest + PyTorch autoencoder on healthy cycles, MLflow, lead-time metric | 100% of held-out engines alerted, median **~58 cycles before failure** | ✅ done |
| T3 | **Chatbot + self-serve**: Streamlit chat (LangGraph, memory, citations, engine lookup) + no-SQL engine explorer | 4-page Streamlit app, Groq free tier | ✅ done |

**Two pipelines, on purpose:** dbt on DuckDB/Snowflake (the analytics-engineering path) and PySpark/Delta (the
lakehouse path) build the same features. Training defaults to the lakehouse gold table (`TG_FEATURE_SOURCE`).

**Scale-up path, after v1:** add all four FD subsets (multiple operating regimes) → GitHub OIDC to AWS
(no static keys) → S3 Terraform backend → provisioned concurrency off / API Gateway throttling →
DagsHub registry → scheduled drift on simulated "live" batches.

## Repo layout
```
scripts/          ingest + utility scripts
dbt/              dbt project (profiles.yml driven by env vars, duckdb + snowflake targets)
src/thermalguard/ config, data access, training, serving, rag
governance/       model cards, data lineage, GDPR note
infra/            Terraform
.github/workflows CI / deploy / retrain
app/              Streamlit
```

## Key numbers (from real runs)
- Sensor readings: **~5.6M** (265k cycles × 21 sensors, all 4 subsets), assets: **1,416 engines**
- RMSE reduction vs baseline: **35.7%** (LSTM 13.27 vs Ridge 20.65, FD001 test set)
- Anomaly lead time: median **~58 cycles** before failure, 100% of held-out engines alerted
