"""FastAPI service exposing the trained heart-disease classifier.

Endpoints
---------
GET  /           -> service metadata
GET  /health     -> readiness / model-loaded probe
POST /predict    -> single-patient inference
POST /predict/batch -> multi-patient inference
GET  /metrics    -> Prometheus scrape endpoint
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src import config
from src.api.schemas import HealthResponse, PatientFeatures, PredictionResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("heart-api")

# ---------------------------------------------------------------------------
# App + Prometheus instrumentation
# ---------------------------------------------------------------------------
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
)

PREDICTION_COUNTER = Counter(
    "heart_predictions_total",
    "Total number of predictions served, labeled by outcome",
    ["label"],
)
PREDICTION_LATENCY = Histogram(
    "heart_prediction_latency_seconds",
    "Time spent in the /predict handler",
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(config.MODEL_FILE)))
METRICS_PATH = Path(os.getenv("METRICS_PATH", str(config.METRICS_FILE)))


class ModelWrapper:
    """Lazily-loaded wrapper around the trained sklearn Pipeline."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model = None
        self._version: str | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            log.warning("Model file %s does not exist yet", self.model_path)
            return
        log.info("Loading model from %s", self.model_path)
        self._model = joblib.load(self.model_path)
        self._version = self._compute_version()

    def _compute_version(self) -> str:
        if METRICS_PATH.exists():
            try:
                with METRICS_PATH.open() as fh:
                    payload = json.load(fh)
                return f"{payload.get('best_model', 'unknown')}@{payload.get('generated_at', '')}"
            except json.JSONDecodeError:
                pass
        stat = self.model_path.stat()
        return f"model@{int(stat.st_mtime)}"

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str | None:
        return self._version

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        proba = self._model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)
        return preds, proba


MODEL = ModelWrapper(MODEL_PATH)


def reload_model(model_path: str | Path | None = None) -> None:
    """Point ``MODEL`` at a new artefact and reload it (used by tests)."""
    global MODEL
    if model_path is not None:
        MODEL = ModelWrapper(Path(model_path))
    MODEL.load()


@app.on_event("startup")
def _startup() -> None:
    MODEL.load()


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000.0
    log.info(
        "request path=%s method=%s status=%s duration_ms=%.2f client=%s",
        request.url.path,
        request.method,
        response.status_code,
        duration_ms,
        request.client.host if request.client else "-",
    )
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _features_to_frame(records: Iterable[PatientFeatures]) -> pd.DataFrame:
    """Convert pydantic objects into a DataFrame with the training-time columns."""
    rows = [rec.model_dump() for rec in records]
    df = pd.DataFrame(rows)
    for col in config.FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[config.FEATURE_COLUMNS]


def _make_response(pred: int, proba: float) -> PredictionResponse:
    label = "disease" if pred == 1 else "no_disease"
    confidence = float(proba if pred == 1 else 1.0 - proba)
    PREDICTION_COUNTER.labels(label=label).inc()
    return PredictionResponse(
        prediction=int(pred),
        label=label,
        probability=float(proba),
        confidence=confidence,
        model_version=MODEL.version or "unknown",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "name": config.API_TITLE,
        "version": config.API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if MODEL.ready else "degraded",
        model_loaded=MODEL.ready,
        model_version=MODEL.version,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    if not MODEL.ready:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    with PREDICTION_LATENCY.time():
        X = _features_to_frame([features])
        try:
            preds, probas = MODEL.predict(X)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return _make_response(int(preds[0]), float(probas[0]))


@app.post("/predict/batch")
def predict_batch(features: list[PatientFeatures]) -> JSONResponse:
    if not MODEL.ready:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    if not features:
        raise HTTPException(status_code=400, detail="Empty batch")
    with PREDICTION_LATENCY.time():
        X = _features_to_frame(features)
        preds, probas = MODEL.predict(X)
    payload = [
        _make_response(int(p), float(pr)).model_dump()
        for p, pr in zip(preds, probas, strict=False)
    ]
    return JSONResponse(content={"predictions": payload, "count": len(payload)})
