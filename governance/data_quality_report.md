# Data-Quality Report: Lakehouse
_Generated 2026-11-23T19:41:43+00:00 by `thermalguard.lakehouse.pipeline`._

| Layer | Table | Rows |
|---|---|---|
| Bronze | `bronze/cmapss_lines` | 266,363 |
| Silver | `silver/cycles` | 265,256 |
| Silver | `silver/rul_truth` | 707 |
| Silver | `silver/quarantine` | 400 |
| Gold | `gold/rul_features` (109 cols, 1,416 assets) | 265,256 |

Reconciliation: bronze lines = silver cycles + silver RUL rows + quarantined rows → 266,363 = 266,363

## Quarantined rows by reason
All quarantined rows come from the **synthetic fault-injection batch** (`scripts/make_fault_batch.py`). The NASA source files themselves are clean.

| Reason | Rows |
|---|---|
| `bad_token_count` | 80 |
| `duplicate_key` | 80 |
| `sensor_out_of_range` | 160 |
| `unparseable_value` | 80 |

## Delta Lake features
- **Schema enforcement:** appending a row with `sensor_2` as a string plus an extra column was **rejected** (`AnalysisException`).
- **Time travel:** `bronze/cmapss_lines` version 0 has 121,836 rows; the latest version has 266,363 (after the batch-2 append).

| Version | Operation | Rows written |
|---|---|---|
| 0 | WRITE | 121836 |
| 1 | WRITE | 144527 |
