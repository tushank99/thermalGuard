"""Generate governance/model_cards/<model>.md from reports/metrics_*.json (re-run after each retrain)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "governance" / "model_cards"

ARCH = {
    "ridge": "scikit-learn Pipeline(StandardScaler → Ridge, alpha=1.0) on 98 engineered features "
             "(14 sensors × raw + rolling mean/std over 5, 10 and 30 cycles) from the lakehouse gold table. "
             "Serves as the baseline.",
    "xgboost": "XGBoost gradient-boosted trees (max_depth 5, lr 0.03, up to 1000 trees with early stopping "
               "on validation RMSE) on the same 98 engineered features.",
    "lstm": "PyTorch 2-layer LSTM (hidden 64, dropout 0.2) → MLP head, over sliding windows of the last 30 "
            "cycles × 14 min-max-scaled sensors. Exported to ONNX for serving with onnxruntime.",
}

TEMPLATE = """# Model Card: rul-{name}

_Generated {today} by `scripts/make_model_cards.py` from `reports/`._

## Intended use
Predict **remaining useful life (RUL, in operating cycles)** of an asset from its recent sensor history, so
maintenance can be planned before failure. Output feeds a health status: critical (<30), warning (<60),
healthy. **Decision support only.** A qualified engineer confirms maintenance actions.

## Architecture
{arch}

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
| RMSE (cycles) | {val_rmse:.2f} | {test_rmse:.2f} |
| NASA PHM08 score (lower is better; penalises late predictions) | {val_nasa:.0f} | {test_nasa:.0f} |
{extra}
## Limitations
- Trained on **simulated** turbofan data with a single operating regime. Performance on real assets, other
  operating regimes (FD002/FD004) or other fault modes is not validated.
- RUL above 125 cycles is not distinguishable by design (the cap).
- Assumes all 14 sensors are present and calibrated as in training. Sensor drift degrades accuracy, which is
  monitored with Evidently (Phase 6).
- Point estimate only, with no calibrated uncertainty interval yet.

## Ethical & governance notes
No personal data is used (see `governance/gdpr_data_handling.md`). Model versions, parameters and metrics are
tracked in the MLflow registry (`rul-{name}`).
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for f in sorted((ROOT / "reports").glob("metrics_*.json")):
        metrics.update(json.loads(f.read_text()))
    base = metrics.get("ridge", {}).get("test", {}).get("rmse")
    for name, m in metrics.items():
        extra = ""
        if base and name != "ridge":
            extra = f"\nRMSE reduction vs Ridge baseline: **{100 * (base - m['test']['rmse']) / base:.1f}%**\n"
        (OUT / f"rul-{name}.md").write_text(TEMPLATE.format(
            name=name, today=date.today(), arch=ARCH[name], extra=extra,
            val_rmse=m["val"]["rmse"], test_rmse=m["test"]["rmse"],
            val_nasa=m["val"]["nasa_score"], test_nasa=m["test"]["nasa_score"]), encoding="utf-8")
        print(f"wrote model card rul-{name}")


if __name__ == "__main__":
    main()
