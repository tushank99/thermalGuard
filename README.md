# ThermalGuard: Predictive Maintenance & Asset Intelligence

ThermalGuard predicts the **remaining useful life (RUL)** of industrial assets from sensor-degradation
data and answers maintenance questions from equipment manuals and maintenance logs.

**Stack:** Python · PySpark · Delta Lake · dbt · DuckDB (Snowflake-ready dbt target) · scikit-learn · XGBoost ·
PyTorch · MLflow · FastAPI · AWS Lambda · Terraform · GitHub Actions · LangGraph · RAGAS · Evidently · Streamlit

## Architecture

```
NASA C-MAPSS raw files
   ├─► PySpark + Delta lakehouse: bronze ─► silver (+ quarantine) ─► gold rolling features ──┐
   └─► dbt (DuckDB / Snowflake): staging ─► marts, gated by data tests ───────────────────────┤
                                                                                              ▼
              Ridge · XGBoost · PyTorch LSTM ──► MLflow tracking + registry ──► model cards (governance/)
              Isolation Forest · autoencoder (anomaly alerts)
                         │ ONNX export
                         ▼
   FastAPI + Mangum on AWS Lambda (Terraform, GitHub Actions)

   Evidently drift check ──► weekly retrain workflow
   LangGraph RAG (manual + maintenance logs) ──► RAGAS evaluation
   Streamlit dashboard: fleet health · engine explorer · anomaly monitor · maintenance assistant
```

## Quickstart (local)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt; pip install -e .
copy .env.example .env

# NASA C-MAPSS → data/raw/cmapss  (see scripts/download_data.ps1)
python scripts/run_pipeline.py            # ingest → dbt build (quality gate) → train
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Data quality gate
`dbt build` runs the models and their tests together: schema (`has_columns`), nulls, uniqueness,
accepted values, physical ranges for all 21 sensors (`tests/assert_sensor_ranges.sql`) and label
ranges. If any test fails, `run_pipeline.py` stops and **training is blocked for that batch**.

## Results (FD001 official test set, 100 engines)

| Model | Test RMSE (cycles) | NASA score ↓ | vs baseline |
|---|---|---|---|
| Ridge (scikit-learn baseline) | 20.65 | 1346 | n/a |
| XGBoost | 17.88 | 961 | −13.4% |
| **PyTorch LSTM** (served via ONNX) | **13.27** | **305** | **−35.7%** |

Data: 5.57M sensor readings (265k cycles × 21 sensors) from 1,416 engines, gated by 29 dbt checks.

The table uses the dbt features (window 10). Retrained on the richer lakehouse gold features (windows 5/10/30),
the Ridge baseline improves to 15.33 and XGBoost to 15.23, so the LSTM is **13.4% better than the strongest baseline**.

| Lakehouse (PySpark + Delta) | Value |
|---|---|
| Rows: bronze → silver → gold | 266,363 → 265,256 (+707 RUL, **400 quarantined**) → 265,256 × 109 cols |
| Quarantine reasons (synthetic fault batch) | 160 out-of-range · 80 bad token count · 80 unparseable · 80 duplicates |
| PySpark vs pandas, same transform (1x / 5x) | 24.8 s vs 8.1 s / 56.3 s vs 34.7 s (single laptop; Spark's time grows more slowly) |

| Anomaly detection (held-out run-to-failure engines) | Isolation Forest | Autoencoder |
|---|---|---|
| Engines alerted before failure | 100% | 100% |
| Median lead time (cycles before failure) | **57.5** | **58.5** |
| Early-alarm rate | 5% | 5% |

| Drift monitoring (Evidently, reference = half of FD001 engines) | Sensors drifted | Retrain triggered |
|---|---|---|
| Same fleet, held-out engines | 0 / 14 | no |
| New operating regimes (FD002) | 14 / 14 | yes |

| Assistant quality (RAGAS, 15-question golden set, judge: gpt-oss-20b) | Score |
|---|---|
| Faithfulness (answers supported by the retrieved sources) | **0.89** |
| Answer relevancy | 0.69 |
| Context precision | 0.63 |

Faithfulness is high because the assistant answers only from cited sources and says so when it can't. Relevancy and
precision are limited by retrieval over a small document set; details in [reports/ragas.json](reports/ragas.json).

Model cards: [governance/model_cards/](governance/model_cards/).

## Lakehouse (PySpark + Delta Lake, runs in Docker)
```powershell
docker build -t thermalguard-spark -f lakehouse/Dockerfile .
python scripts/make_fault_batch.py                     # synthetic corrupted batch → exercises quarantine
docker run --rm -v "${PWD}:/app" thermalguard-spark python -m thermalguard.lakehouse.pipeline
docker run --rm -v "${PWD}:/app" thermalguard-spark python -m thermalguard.lakehouse.benchmark
```
Bronze (raw lines + lineage) → Silver (explicit schema, bad rows → `silver/quarantine`) → Gold (RUL + rolling
mean/std over 5/10/30 cycles). Reports: [data quality](governance/data_quality_report.md),
[data dictionary](governance/data_dictionary.md).

## Anomaly detection, fleet scoring, assistant, dashboard
```powershell
python -m thermalguard.anomaly        # Isolation Forest + autoencoder → reports/anomaly.json
python -m thermalguard.fleet          # per-engine RUL + alerts for the dashboard
python scripts/make_maintenance_logs.py; python -m thermalguard.rag.index
python -m thermalguard.rag.evaluate   # RAGAS: faithfulness, answer relevancy, context precision
streamlit run app/Home.py             # fleet · explorer · anomaly monitor · assistant
```
The assistant needs a free Groq key (`GROQ_API_KEY` in `.env`, see `.env.example`).

## Drift monitoring and scheduled retraining
```powershell
python -m thermalguard.drift          # Evidently report -> reports/drift.json + reports/drift_<batch>.html
```
Normed Wasserstein per sensor; the threshold (0.3) is calibrated on a same-fleet holdout, where healthy
engine-to-engine variation scores up to 0.16 and a new operating regime scores 4 or more.
[retrain.yml](.github/workflows/retrain.yml) runs weekly: dbt quality gate, then the drift check, then a retrain of
the classic and LSTM models only when drift is detected. Reports and the candidate model are uploaded as run artifacts.

## Hosted dashboard (Streamlit Community Cloud)
The hosted app reads small precomputed files committed under `artifacts/` (fleet scores, anomaly thresholds,
RAG index, a slim sensor export of gold), so the cloud needs no Spark, torch or raw data. After re-running the
pipeline, refresh them with `python -m thermalguard.fleet`, `python -m thermalguard.rag.index` and
`python scripts/export_dashboard_data.py`. Dependencies: [app/requirements.txt](app/requirements.txt).
The Groq key goes in the app's **Secrets** as `GROQ_API_KEY = "..."`.

## Serve locally
```powershell
pip install -r requirements-serve.txt
uvicorn thermalguard.serving.app:app --reload   # http://127.0.0.1:8000/docs
```
