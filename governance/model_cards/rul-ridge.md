# Model Card: rul-ridge

_Generated 2026-11-24 by `scripts/make_model_cards.py` from `reports/`._

## Intended use
Predict **remaining useful life (RUL, in operating cycles)** of an asset from its recent sensor history, so
maintenance can be planned before failure. Output feeds a health status: critical (<30), warning (<60),
healthy. **Decision support only.** A qualified engineer confirms maintenance actions.

## Architecture
scikit-learn Pipeline(StandardScaler → Ridge, alpha=1.0) on 98 engineered features (14 sensors × raw + rolling mean/std over 5, 10 and 30 cycles) from the lakehouse gold table. Serves as the baseline.

## Training data
- NASA C-MAPSS turbofan degradation, subset FD001 (one operating condition, one fault mode: HPC degradation).
- 100 run-to-failure training engines, split 80/20 **by engine** into train/validation (no leakage).
- Target: piecewise-linear RUL capped at 125 cycles (early life is treated as healthy).
- Source: lakehouse Delta table `gold/rul_features` (bronze → silver with quarantine → gold; see
  `governance/data_dictionary.md` and `governance/data_quality_report.md`). The dbt mart `marts.fct_rul_features`
  is an equivalent alternative source (`--source dbt`).

## Evaluation (official FD001 test set: 100 engines, RUL at the last observed cycle)
| Metric | Validation | Test |
|---|---|---|
| RMSE (cycles) | 14.76 | 15.33 |
| NASA PHM08 score (lower is better; penalises late predictions) | 13281 | 387 |

## Limitations
- Trained on **simulated** turbofan data with a single operating regime. Performance on real assets, other
  operating regimes (FD002/FD004) or other fault modes is not validated.
- RUL above 125 cycles is not distinguishable by design (the cap).
- Assumes all 14 sensors are present and calibrated as in training. Sensor drift degrades accuracy, which is
  monitored with Evidently (Phase 6).
- Point estimate only, with no calibrated uncertainty interval yet.

## Ethical & governance notes
No personal data is used (see `governance/gdpr_data_handling.md`). Model versions, parameters and metrics are
tracked in the MLflow registry (`rul-ridge`).
