# Data Dictionary: ThermalGuard Lakehouse

Storage: Delta Lake tables under `data/lakehouse/<layer>/<table>`, written by `thermalguard.lakehouse.pipeline`
(PySpark 3.5 + Delta 3.2). Grain, keys and owners are listed per table. Source: NASA C-MAPSS (public domain,
simulated). Contains **no personal data**.

## Bronze: `bronze/cmapss_lines` (append-only, raw)
Grain: one raw text line of one source file. Never modified after ingest.

| Column | Type | Description |
|---|---|---|
| value | string | Raw line exactly as received (space-separated tokens) |
| line_number | long | 1-based line position in the source file (= unit id for RUL files) |
| source_file | string | File name, e.g. `train_FD001.txt`, `RUL_FD003.txt`, `train_FD001_injected.txt` |
| batch_id | int | Ingest batch (1 = FD001–FD002, 2 = FD003–FD004 + fault-injection batch) |
| ingest_ts | timestamp | UTC time the batch was ingested |

## Silver: `silver/cycles` (validated, typed, explicit schema, all columns NOT NULL)
Grain: one asset × operating cycle. Key: (`asset_id`, `cycle`), unique.

| Column | Type | Description |
|---|---|---|
| asset_id | string | `<subset>_<split>_<unit:03d>`, e.g. `FD001_test_031` |
| subset | string | FD001–FD004 (operating conditions × fault modes) |
| split | string | `train` = run-to-failure, `test` = truncated, in service |
| unit_id | int | Engine number within subset and split |
| cycle | int | Operating cycle (≥ 1) |
| op_setting_1..3 | double | Operating condition: altitude, Mach and throttle resolver angle (op_setting_3 ∈ [60, 100]) |
| sensor_1..21 | double | Sensor channels (see manual §1, e.g. sensor_3 = T30, HPC outlet temperature). Valid range [0, 10000] |
| source_file, batch_id, ingest_ts | | Lineage back to bronze |

## Silver: `silver/rul_truth`
| Column | Type | Description |
|---|---|---|
| asset_id | string | Test engine id |
| rul_true | int | True RUL at the engine's last observed cycle |

## Silver: `silver/quarantine`
Rows rejected by validation, kept for audit and never used for training.

| Column | Type | Description |
|---|---|---|
| value, source_file, line_number, batch_id, ingest_ts | | Original bronze record |
| dq_reason | string | `bad_token_count`, `unparseable_value`, `invalid_key`, `op_setting_out_of_range`, `sensor_out_of_range`, `duplicate_key` |
| quarantined_at | timestamp | When the row was quarantined |

## Gold: `gold/rul_features` (partitioned by `subset`)
Grain: one asset × cycle. Consumed by training (`--source lakehouse`), anomaly detection and the dashboard.

| Column | Type | Description |
|---|---|---|
| asset_id, subset, split, unit_id, cycle | | Keys (see silver) |
| rul | int | Cycles until failure (train: last − cycle; test: rul_true + last − cycle) |
| rul_capped | int | min(rul, 125). Training target (piecewise-linear RUL) |
| is_last_cycle | bool | Last observed cycle of the asset (used for official test scoring) |
| op_setting_1..3 | double | Operating condition |
| sensor_{2,3,4,7,8,9,11,12,13,14,15,17,20,21} | double | 14 informative raw sensors |
| sensor_X_mean_{5,10,30} | double | Rolling mean over the last w cycles (fewer at start of life) |
| sensor_X_std_{5,10,30} | double | Rolling sample std over the last w cycles (0 when only 1 cycle) |

**Versioning:** every write creates a Delta version, so any table can be read as of an earlier version
(`versionAsOf`) for audit and reproducibility. See `governance/data_quality_report.md`.
