# Model Card: rul-lstm

_Generated 2026-11-24 by `scripts/make_model_cards.py` from `reports/`._

## Intended use
Predict **remaining useful life (RUL, in operating cycles)** of an asset from its recent sensor history, so
maintenance can be planned before failure. Output feeds a health status: critical (<30), warning (<60),
healthy. **Decision support only.** A qualified engineer confirms maintenance actions.

## Architecture
PyTorch 2-layer LSTM (hidden 64, dropout 0.2) → MLP head, over sliding windows of the last 30 cycles × 14 min-max-scaled sensors. Exported to ONNX for serving with onnxruntime.

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
| RMSE (cycles) | 12.48 | 13.27 |
| NASA PHM08 score (lower is better; penalises late predictions) | 8905 | 305 |

RMSE reduction vs Ridge baseline: **13.4%**

## Limitations
- Trained on **simulated** turbofan data with a single operating regime. Performance on real assets, other
  operating regimes (FD002/FD004) or other fault modes is not validated.
- RUL above 125 cycles is not distinguishable by design (the cap).
- Assumes all 14 sensors are present and calibrated as in training. Sensor drift degrades accuracy, which is
  monitored with Evidently (`python -m thermalguard.drift`).
- Point estimate only, with no calibrated uncertainty interval yet.

## Ethical & governance notes
No personal data is used (see `governance/gdpr_data_handling.md`). Model versions, parameters and metrics are
tracked in the MLflow registry (`rul-lstm`).
