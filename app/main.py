"""API REST con FastAPI para servir un modelo de clasificación."""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.schemas import BatchRequest, BatchResponse, IrisFeatures, PredictionResponse

# ---------------------------------------------------------------- LOGGING ---
Path("logs").mkdir(exist_ok=True)
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = RotatingFileHandler("logs/api.log", maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console)

# ------------------------------------------------------- MÉTRICAS SIMPLES ---
metrics = {"requests": 0, "predictions": 0, "errors": 0, "total_time_ms": 0.0}
state: dict = {}
MODEL_PATH = "models/iris_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo UNA sola vez, al iniciar el servidor."""
    try:
        state["bundle"] = joblib.load(MODEL_PATH)
        logger.info("Modelo cargado correctamente desde %s", MODEL_PATH)
    except Exception:
        logger.exception("No se pudo cargar el modelo")
        state["bundle"] = None
    yield
    logger.info("Servidor detenido")


app = FastAPI(
    title="API de Clasificación de Iris",
    description="Servicio web que predice la especie de una flor Iris.",
    version="1.0.0",
    lifespan=lifespan,
)


# -------------------------------------------------------------- MIDDLEWARE ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Registra cada petición: id, método, ruta, estado y duración."""
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        metrics["errors"] += 1
        logger.exception("[%s] Error no controlado en %s", request_id, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Error interno del servidor"})
    elapsed = (time.perf_counter() - start) * 1000
    metrics["requests"] += 1
    metrics["total_time_ms"] += elapsed
    if response.status_code >= 400:
        metrics["errors"] += 1
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(level, "[%s] %s %s -> %s (%.1f ms)", request_id, request.method,
               request.url.path, response.status_code, elapsed)
    return response


# --------------------------------------------------------------- UTILIDAD ---
def _predict(samples: list[IrisFeatures]) -> list[PredictionResponse]:
    bundle = state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    X = np.array([[s.sepal_length, s.sepal_width, s.petal_length, s.petal_width] for s in samples])
    probs = bundle["model"].predict_proba(X)
    classes = bundle["classes"]
    results = []
    for p in probs:
        idx = int(np.argmax(p))
        results.append(PredictionResponse(
            prediction=classes[idx],
            confidence=round(float(p[idx]), 4),
            probabilities={c: round(float(v), 4) for c, v in zip(classes, p)},
        ))
    metrics["predictions"] += len(results)
    return results


# --------------------------------------------------------------- ENDPOINTS ---
@app.get("/", tags=["General"])
def root():
    return {"mensaje": "API de clasificación activa. Documentación en /docs"}


@app.get("/health", tags=["Monitoreo"])
def health():
    """Indica si el servicio y el modelo están operativos."""
    ok = state.get("bundle") is not None
    return {"status": "ok" if ok else "degradado", "modelo_cargado": ok}


@app.get("/model-info", tags=["General"])
def model_info():
    bundle = state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return {"tipo": type(bundle["model"]).__name__, "clases": bundle["classes"],
            "accuracy_test": bundle["accuracy"]}


@app.get("/metrics", tags=["Monitoreo"])
def get_metrics():
    n = metrics["requests"]
    return {**metrics, "avg_latency_ms": round(metrics["total_time_ms"] / n, 2) if n else 0}


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(features: IrisFeatures):
    """Predice la especie de UNA flor."""
    result = _predict([features])[0]
    logger.info("Predicción: %s (confianza %.2f)", result.prediction, result.confidence)
    return result


@app.post("/predict/batch", response_model=BatchResponse, tags=["Predicción"])
def predict_batch(batch: BatchRequest):
    """Predice la especie de VARIAS flores en una sola llamada."""
    return BatchResponse(predictions=_predict(batch.samples))
