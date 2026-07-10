"""Aggregate project metrics for the premium dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

ALGORITHM_LABELS = {
    "baseline_naive": "Baseline Naive",
    "baseline_engineered": "Baseline Engineered",
    "linear_regression": "Linear Regression",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elasticnet": "Elastic Net",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_overview() -> dict[str, Any]:
    training = _read_json(REPORTS_DIR / "training_summary.json") or {}
    baseline = _read_json(REPORTS_DIR / "baseline_metrics.json") or {}
    registry = _read_json(REPORTS_DIR / "registry_summary.json")

    best = training.get("best_run", {})
    naive_metrics = baseline.get("naive", {}).get("metrics", {})
    engineered_metrics = baseline.get("engineered", {}).get("metrics", {})
    return {
        "project_name": "House Price Prediction Engine",
        "tagline": "Pipeline MLOps end-to-end — Ames Housing",
        "best_model": {
            "run_name": best.get("run_name"),
            "algorithm": best.get("algorithm"),
            "mae": best.get("metrics", {}).get("mae"),
            "rmse": best.get("metrics", {}).get("rmse"),
            "r2": best.get("metrics", {}).get("r2"),
            "cv_mae": best.get("metrics", {}).get("cv_mae"),
            "n_features": best.get("metrics", {}).get("n_features"),
        },
        "registry": registry,
        "baseline": {
            "naive_mae": naive_metrics.get("mae"),
            "naive_rmse": naive_metrics.get("rmse"),
            "engineered_mae": engineered_metrics.get("mae"),
            "engineered_rmse": engineered_metrics.get("rmse"),
            "improvement_pct": baseline.get("summary", {}).get("mae_improvement_pct"),
            "conclusion": baseline.get("summary", {}).get("conclusion"),
        },
        "pipeline_stages": get_pipeline_stages(),
        "stats": {
            "total_experiments": training.get("total_runs", 0),
            "algorithms_count": 6,
            "dataset_rows": 1460,
            "features_engineered": 277,
            "dvc_stages": 5,
        },
    }


def get_pipeline_stages() -> list[dict[str, Any]]:
    return [
        {
            "id": "prepare",
            "name": "Prepare",
            "phase": "Fase 2",
            "description": "Validação, limpeza e versionamento DVC dos dados brutos.",
            "outputs": ["data/interim/clean.csv", "data_profile.json"],
            "status": "complete",
        },
        {
            "id": "featurize",
            "name": "Featurize",
            "phase": "Fase 3",
            "description": "8 features derivadas, imputação, OneHotEncoder e scaling.",
            "outputs": ["X_train.csv", "preprocessor.pkl", "277 features"],
            "status": "complete",
        },
        {
            "id": "baseline",
            "name": "Baseline",
            "phase": "Fase 4",
            "description": "Comparação naive vs pipeline completo de features.",
            "outputs": ["baseline_metrics.json", "baseline_comparison.md"],
            "status": "complete",
        },
        {
            "id": "train",
            "name": "Train",
            "phase": "Fase 5",
            "description": "33 experimentos MLflow — 6 algoritmos com grid search.",
            "outputs": ["best_model.pkl", "training_summary.json", "mlflow.db"],
            "status": "complete",
        },
        {
            "id": "evaluate",
            "name": "Evaluate",
            "phase": "Fase 6/7",
            "description": "Testes t pareados + promoção ao Model Registry Production.",
            "outputs": ["model_comparison.md", "registry_summary.json"],
            "status": "complete",
        },
        {
            "id": "serve",
            "name": "Serve",
            "phase": "Fase 8",
            "description": "FastAPI + dashboard premium para inferência e visualização.",
            "outputs": ["/predict", "/docs", "Dashboard UI"],
            "status": "active",
        },
    ]


def get_algorithm_comparison() -> dict[str, Any]:
    training = _read_json(REPORTS_DIR / "training_summary.json") or {}
    significance = _read_json(REPORTS_DIR / "significance_tests.json") or {}

    runs = training.get("runs", [])
    best_by_algo: dict[str, dict[str, Any]] = {}

    for run in runs:
        algo = run.get("algorithm", "unknown")
        mae = run.get("metrics", {}).get("mae")
        if mae is None:
            continue
        current = best_by_algo.get(algo)
        if current is None or mae < current["mae"]:
            best_by_algo[algo] = {
                "algorithm": algo,
                "label": ALGORITHM_LABELS.get(algo, algo.replace("_", " ").title()),
                "run_name": run.get("run_name"),
                "mae": mae,
                "rmse": run.get("metrics", {}).get("rmse"),
                "r2": run.get("metrics", {}).get("r2"),
                "cv_mae": run.get("metrics", {}).get("cv_mae"),
            }

    order = [
        "baseline_naive",
        "baseline_engineered",
        "linear_regression",
        "ridge",
        "lasso",
        "elasticnet",
        "random_forest",
        "gradient_boosting",
    ]
    algorithms = [best_by_algo[key] for key in order if key in best_by_algo]

    return {
        "algorithms": algorithms,
        "significance_tests": significance.get("tests", []),
        "significance_level": significance.get("significance_level", 0.05),
    }
