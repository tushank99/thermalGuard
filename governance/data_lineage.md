# Data Lineage

| Layer | Object | Produced by | Key logic |
|---|---|---|---|
| Source | NASA C-MAPSS `train_/test_/RUL_FD00x.txt` | NASA Prognostics Data Repository (public domain) | Simulated turbofan run-to-failure telemetry |
| Raw | `raw.cmapss_train`, `raw.cmapss_test`, `raw.cmapss_rul` | `scripts/ingest.py` | Loaded as-is, plus `batch_id` and `loaded_at` per load |
| Staging | `staging.stg_cycles` | dbt view | Union train/test, typed columns, `asset_id` and `cycle_key` |
| Staging | `staging.stg_rul_truth` | dbt view | Ground-truth RUL for test engines |
| Mart | `marts.fct_asset_cycles` | dbt table | RUL label (train: last−cycle, test: truth + last−cycle), `rul_capped` = min(RUL, 125) |
| Mart | `marts.fct_rul_features` | dbt table | Rolling mean, std and delta (window 10) for 14 informative sensors |
| Model | `rul-ridge`, `rul-xgboost`, `rul-lstm` (MLflow registry) | `src/thermalguard/train_*.py` | Trained on `fct_rul_features`; each run tagged with its data source |
| Serving | `artifacts/*/model.onnx` + `preprocess.json` | export step in training | Loaded by the FastAPI service |

**Quality gate.** `dbt build` runs the tests at each layer (schema, nulls, uniqueness, accepted values,
sensor and label ranges). A failing test stops `scripts/run_pipeline.py` before training, so no model
is ever trained on a rejected batch.

**Traceability.** batch_id (raw row) → dbt run (`dbt/target/run_results.json`) → MLflow run (params,
metrics, data_source tag) → registered model version → deployed artifact.

dbt docs (`dbt docs generate && dbt docs serve`) render the full lineage graph.
