"""Baseline linear regression experiments: with vs without feature engineering."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer

from src.models.metrics import compute_regression_metrics, to_dollar_scale
from src.utils.config import load_params

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "SalePrice"
CLEAN_TRAIN_PATH = INTERIM_DIR / "clean.csv"

NAIVE_MODEL_PATH = MODELS_DIR / "baseline_naive.pkl"
ENGINEERED_MODEL_PATH = MODELS_DIR / "baseline_engineered.pkl"
METRICS_PATH = REPORTS_DIR / "baseline_metrics.json"
COMPARISON_PATH = REPORTS_DIR / "baseline_comparison.md"


def _cross_validate_mae(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    log_scale: bool,
) -> float:
    def dollar_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(
            to_dollar_scale(y_true, log_scale=log_scale),
            to_dollar_scale(y_pred, log_scale=log_scale),
        )

    scorer = make_scorer(dollar_mae, greater_is_better=False)
    scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring=scorer)
    return float(np.mean(-scores))


def run_naive_baseline(params: dict) -> dict:
    """Linear regression on numeric columns only — no encoding, scaling or derived features."""
    train_df = pd.read_csv(CLEAN_TRAIN_PATH)
    training = params["training"]
    feature_params = params["features"]

    X = train_df.drop(columns=[TARGET_COLUMN])
    y = train_df[TARGET_COLUMN]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric,
        y,
        test_size=training["test_size"],
        random_state=training["random_state"],
    )

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    log_scale = False
    metrics = compute_regression_metrics(y_test.to_numpy(), predictions, log_scale=log_scale)
    metrics["cv_mae"] = _cross_validate_mae(
        pipeline, X_numeric, y, training["cv_folds"], log_scale=log_scale
    )
    metrics["n_features"] = int(X_numeric.shape[1])
    metrics["experiment"] = "naive_numeric_only"
    metrics["description"] = (
        "LinearRegression on numeric columns only (no categorical encoding, "
        "no scaling, no derived features)"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, NAIVE_MODEL_PATH)

    return {
        "metrics": metrics,
        "model_path": str(NAIVE_MODEL_PATH),
        "feature_strategy": "numeric_only",
    }


def run_engineered_baseline(params: dict) -> dict:
    """Linear regression on fully engineered feature matrices from Phase 3."""
    feature_params = params["features"]
    training = params["training"]
    log_scale = feature_params["log_target"]

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")[TARGET_COLUMN]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")[TARGET_COLUMN]

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = compute_regression_metrics(y_test.to_numpy(), predictions, log_scale=log_scale)

    full_pipeline = Pipeline([("model", model)])
    metrics["cv_mae"] = _cross_validate_mae(
        full_pipeline,
        X_train,
        y_train,
        training["cv_folds"],
        log_scale=log_scale,
    )
    metrics["n_features"] = int(X_train.shape[1])
    metrics["experiment"] = "engineered_pipeline"
    metrics["description"] = (
        "LinearRegression on processed features (derived features, imputation, "
        "one-hot encoding, scaling, log target)"
    )
    metrics["log_target"] = log_scale

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ENGINEERED_MODEL_PATH)

    return {
        "metrics": metrics,
        "model_path": str(ENGINEERED_MODEL_PATH),
        "feature_strategy": "full_pipeline",
    }


def _improvement_pct(naive_value: float, engineered_value: float, *, higher_is_better: bool) -> float:
    if naive_value == 0:
        return 0.0
    if higher_is_better:
        return ((engineered_value - naive_value) / abs(naive_value)) * 100
    return ((naive_value - engineered_value) / abs(naive_value)) * 100


def _render_comparison_markdown(results: dict) -> str:
    naive = results["naive"]["metrics"]
    engineered = results["engineered"]["metrics"]
    summary = results["summary"]

    return f"""# Baseline Comparison — Linear Regression

Comparação entre regressão linear **sem** e **com** feature engineering (Fase 4).

## Resultados no holdout (20%)

| Métrica | Sem FE (numéricas apenas) | Com FE (pipeline completo) | Melhoria |
|---------|---------------------------|----------------------------|----------|
| MAE ($) | ${naive['mae']:,.0f} | ${engineered['mae']:,.0f} | {summary['mae_improvement_pct']:.1f}% |
| RMSE ($) | ${naive['rmse']:,.0f} | ${engineered['rmse']:,.0f} | {summary['rmse_improvement_pct']:.1f}% |
| R² | {naive['r2']:.4f} | {engineered['r2']:.4f} | {summary['r2_improvement_pct']:.1f}% |

## Cross-validation ({results['cv_folds']}-fold)

| Métrica | Sem FE | Com FE |
|---------|--------|--------|
| MAE | ${naive['cv_mae']:,.0f} | ${engineered['cv_mae']:,.0f} |

## Features utilizadas

| Experimento | Nº de features | Estratégia |
|-------------|----------------|------------|
| Sem FE | {naive['n_features']} | Apenas colunas numéricas brutas |
| Com FE | {engineered['n_features']} | Derived + imputation + OHE + scaling |

## Conclusão

{summary['conclusion']}

## Artefatos

- Modelo naive: `models/baseline_naive.pkl`
- Modelo engineered: `models/baseline_engineered.pkl`
- Métricas JSON: `reports/baseline_metrics.json`
"""


def run_baselines(params: dict | None = None) -> dict:
    params = params or load_params()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    naive_result = run_naive_baseline(params)
    engineered_result = run_engineered_baseline(params)

    naive_metrics = naive_result["metrics"]
    engineered_metrics = engineered_result["metrics"]

    mae_improvement = _improvement_pct(naive_metrics["mae"], engineered_metrics["mae"], higher_is_better=False)
    rmse_improvement = _improvement_pct(naive_metrics["rmse"], engineered_metrics["rmse"], higher_is_better=False)
    r2_improvement = _improvement_pct(naive_metrics["r2"], engineered_metrics["r2"], higher_is_better=True)

    if engineered_metrics["r2"] > naive_metrics["r2"] and engineered_metrics["mae"] < naive_metrics["mae"]:
        conclusion = (
            "O pipeline de feature engineering reduziu o erro e aumentou o R² de forma clara. "
            "Regressão linear pura em features numéricas é insuficiente para este dataset — "
            "encoding de categóricas, scaling e features derivadas são necessários."
        )
    else:
        conclusion = (
            "Revise os resultados: o modelo engineered deveria superar o naive neste dataset."
        )

    results = {
        "naive": naive_result,
        "engineered": engineered_result,
        "cv_folds": params["training"]["cv_folds"],
        "summary": {
            "mae_improvement_pct": mae_improvement,
            "rmse_improvement_pct": rmse_improvement,
            "r2_improvement_pct": r2_improvement,
            "conclusion": conclusion,
        },
    }

    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    COMPARISON_PATH.write_text(_render_comparison_markdown(results), encoding="utf-8")

    print(f"Naive baseline      — MAE: ${naive_metrics['mae']:,.0f} | R²: {naive_metrics['r2']:.4f}")
    print(f"Engineered baseline — MAE: ${engineered_metrics['mae']:,.0f} | R²: {engineered_metrics['r2']:.4f}")
    print(f"Saved {METRICS_PATH}")
    print(f"Saved {COMPARISON_PATH}")

    return results


def main() -> None:
    run_baselines()


if __name__ == "__main__":
    main()
