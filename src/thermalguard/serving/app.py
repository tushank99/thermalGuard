"""FastAPI RUL service. Uses only onnxruntime + numpy, so it fits in a Lambda zip (no torch, no container).

Local:   uvicorn thermalguard.serving.app:app --reload
Lambda:  handler = thermalguard.serving.app.handler   (Mangum adapter)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from mangum import Mangum
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[3] / "artifacts" / "lstm"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "local")

_pre = json.loads((MODEL_DIR / "preprocess.json").read_text())
_sess = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
_lo, _hi = np.array(_pre["min"], dtype=np.float32), np.array(_pre["max"], dtype=np.float32)
SENSORS, SEQ_LEN, RUL_CAP = _pre["sensors"], _pre["seq_len"], _pre["rul_cap"]


class PredictRequest(BaseModel):
    asset_id: str = Field(examples=["FD001_test_031"])
    readings: list[dict[str, float]] = Field(
        min_length=1, description=f"Per-cycle sensor readings, oldest first; last {SEQ_LEN} are used.")


class PredictResponse(BaseModel):
    asset_id: str
    rul_cycles: float
    status: str
    model_version: str


def health_status(rul: float) -> str:
    return "critical" if rul < 30 else "warning" if rul < 60 else "healthy"


def to_window(readings: list[dict[str, float]]) -> np.ndarray:
    missing = [s for s in SENSORS if s not in readings[0]]
    if missing:
        raise HTTPException(422, f"missing sensors: {missing}")
    x = np.array([[r[s] for s in SENSORS] for r in readings[-SEQ_LEN:]], dtype=np.float32)
    x = (x - _lo) / (_hi - _lo + 1e-9)
    if len(x) < SEQ_LEN:  # front-pad short histories exactly like training
        x = np.vstack([np.repeat(x[:1], SEQ_LEN - len(x), axis=0), x])
    return x[None, ...]


app = FastAPI(title="ThermalGuard RUL API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "sensors": SENSORS, "seq_len": SEQ_LEN}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    rul = float(np.clip(_sess.run(None, {"window": to_window(req.readings)})[0][0], 0, RUL_CAP))
    return PredictResponse(asset_id=req.asset_id, rul_cycles=round(rul, 1),
                           status=health_status(rul), model_version=MODEL_VERSION)


handler = Mangum(app)
