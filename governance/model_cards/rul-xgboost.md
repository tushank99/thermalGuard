# Model Card: rul-xgboost

_Generated 2026-11-24 by `scripts/make_model_cards.py` from `reports/`._

## Intended use
Predict **remaining useful life (RUL, in operating cycles)** of an asset from its recent sensor history, so
maintenance can be planned before failure. Output feeds a health status: critical (<30), warning (<60),
healthy. **Decision support only.** A qualified engineer confirms maintenance actions.

## Architecture
XGBoost gradient-boosted trees (max_depth 5, lr 0.03, up to 1000 trees with early stopping on validation RMSE) on the same 98 engineered features.

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
| RMSE (cycles) | 13.33 | 15.23 |
| NASA PHM08 score (lower is better; penalises late predictions) | 14583 | 532 |

RMSE reduction vs Ridge baseline: **0.7%**

## Limitations
- Trained on **simulated** turbofan data with a single operating regime. Performance on real assets, other
  operating regimes (FD002/FD004) or other fault modes is not validated.
- RUL above 125 cycles is not distinguishable by design (the cap).
- Assumes all 14 sensors are present and calibrated as in training. Sensor drift degrades accuracy, which is
  monitored with Evidently (Phase 6).
- Point estimate only, with no calibrated uncertainty interval yet.

## Ethical & governance notes
No personal data is used (see `governance/gdpr_data_handling.md`). Model versions, parameters and metrics are
tracked in the MLflow registry (`rul-xgboost`).
