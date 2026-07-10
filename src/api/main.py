"""FastAPI application — prediction API + premium dashboard."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.dashboard import get_algorithm_comparison, get_overview
from src.api.predictor import predictor_service
from src.api.schemas import (
    AlgorithmComparison,
    DashboardOverview,
    ExampleComparison,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def _format_currency(value: float) -> str:
    return f"${value:,.0f}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        predictor_service.load()
    except (FileNotFoundError, AttributeError, Exception) as exc:
        import logging

        logging.getLogger("uvicorn.error").warning(
            "Model not loaded at startup (%s). Predictions will fail until pipeline is reproduced.",
            exc,
        )
    yield


app = FastAPI(
    title="House Price Prediction Engine",
    description="MLOps pipeline — Ames Housing price prediction with MLflow registry",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    meta = predictor_service.metadata if predictor_service.is_ready else {}
    return HealthResponse(
        status="healthy" if predictor_service.is_ready else "degraded",
        model_loaded=predictor_service.is_ready,
        preprocessor_loaded=predictor_service.is_ready,
        model_name=meta.get("model_name"),
        model_version=meta.get("model_version"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: PredictionRequest) -> PredictionResponse:
    if not predictor_service.is_ready:
        try:
            predictor_service.load()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    payload = request.model_dump(by_alias=True)
    result = predictor_service.predict_with_interval(payload)
    meta = predictor_service.metadata

    return PredictionResponse(
        predicted_price=result["price"],
        predicted_price_formatted=_format_currency(result["price"]),
        confidence_interval_low=result["low"],
        confidence_interval_high=result["high"],
        model_name=meta.get("model_name", "house-price-predictor"),
        log_scale=bool(meta.get("log_target")),
    )


@app.get("/api/v1/dashboard/overview", response_model=DashboardOverview, tags=["Dashboard"])
def dashboard_overview() -> DashboardOverview:
    return DashboardOverview(**get_overview())


@app.get("/api/v1/dashboard/algorithms", response_model=AlgorithmComparison, tags=["Dashboard"])
def dashboard_algorithms() -> AlgorithmComparison:
    payload = get_algorithm_comparison()
    return AlgorithmComparison(
        algorithms=payload["algorithms"],
        significance_tests=payload["significance_tests"],
    )


@app.get("/api/v1/dashboard/examples", response_model=ExampleComparison, tags=["Dashboard"])
def dashboard_examples() -> ExampleComparison:
    if not predictor_service.is_ready:
        try:
            predictor_service.load()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ExampleComparison(
        examples=predictor_service.get_holdout_examples(n=6),
        scatter=predictor_service.get_scatter_sample(limit=80),
    )


@app.get("/", include_in_schema=False)
def serve_dashboard():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Dashboard not found. API docs at /docs"}


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
