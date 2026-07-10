"""Train models with MLflow experiment tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.models.baseline import (
    CLEAN_TRAIN_PATH,
    PROCESSED_DIR,
    TARGET_COLUMN,
    _cross_validate_mae,
    run_engineered_baseline,
)
from src.models.metrics import compute_regression_metrics
from src.utils.config import load_params
from src.utils.mlflow_utils import log_model_metrics, log_model_params, set_run_tags, setup_mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
TRAINING_SUMMARY_PATH = REPORTS_DIR / "training_summary.json"


def _load_engineered_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")[TARGET_COLUMN]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")[TARGET_COLUMN]
    return X_train, X_test, y_train, y_test


def _build_estimator(algorithm: str, hyperparams: dict[str, Any]):
    if algorithm == "linear_regression":
        return LinearRegression()
    if algorithm == "ridge":
        return Ridge(alpha=hyperparams["alpha"])
    if algorithm == "lasso":
        return Lasso(alpha=hyperparams["alpha"], max_iter=10_000)
    if algorithm == "elasticnet":
        return ElasticNet(
            alpha=hyperparams["alpha"],
            l1_ratio=hyperparams["l1_ratio"],
            max_iter=10_000,
        )
    if algorithm == "random_forest":
        return RandomForestRegressor(
            n_estimators=hyperparams["n_estimators"],
            max_depth=hyperparams["max_depth"],
            random_state=hyperparams.get("random_state", 42),
            n_jobs=-1,
        )
    if algorithm == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=hyperparams["n_estimators"],
            learning_rate=hyperparams["learning_rate"],
            max_depth=hyperparams["max_depth"],
            random_state=hyperparams.get("random_state", 42),
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def build_experiment_grid(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return flattened list of experiment configurations."""
    grid = params["experiments"]
    random_state = params["training"]["random_state"]
    log_target = params["features"]["log_target"]
    experiments: list[dict[str, Any]] = [
        {
            "run_name": "baseline_naive",
            "algorithm": "baseline_naive",
            "feature_set": "numeric_only",
            "hyperparams": {},
            "log_target": False,
            "is_baseline": True,
        },
        {
            "run_name": "baseline_engineered",
            "algorithm": "baseline_engineered",
            "feature_set": "engineered",
            "hyperparams": {},
            "log_target": log_target,
            "is_baseline": True,
        },
        {
            "run_name": "linear_regression_engineered",
            "algorithm": "linear_regression",
            "feature_set": "engineered",
            "hyperparams": {},
            "log_target": log_target,
        },
    ]

    for alpha in grid["ridge"]["alpha"]:
        experiments.append(
            {
                "run_name": f"ridge_alpha_{alpha}",
                "algorithm": "ridge",
                "feature_set": "engineered",
                "hyperparams": {"alpha": alpha},
                "log_target": log_target,
            }
        )

    for alpha in grid["lasso"]["alpha"]:
        experiments.append(
            {
                "run_name": f"lasso_alpha_{alpha}",
                "algorithm": "lasso",
                "feature_set": "engineered",
                "hyperparams": {"alpha": alpha},
                "log_target": log_target,
            }
        )

    for alpha in grid["elasticnet"]["alpha"]:
        for l1_ratio in grid["elasticnet"]["l1_ratio"]:
            experiments.append(
                {
                    "run_name": f"elasticnet_a{alpha}_l1{l1_ratio}",
                    "algorithm": "elasticnet",
                    "feature_set": "engineered",
                    "hyperparams": {"alpha": alpha, "l1_ratio": l1_ratio},
                    "log_target": log_target,
                }
            )

    for n_estimators in grid["random_forest"]["n_estimators"]:
        for max_depth in grid["random_forest"]["max_depth"]:
            depth_label = "none" if max_depth is None else str(max_depth)
            experiments.append(
                {
                    "run_name": f"rf_est{n_estimators}_depth{depth_label}",
                    "algorithm": "random_forest",
                    "feature_set": "engineered",
                    "hyperparams": {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "random_state": random_state,
                    },
                    "log_target": log_target,
                }
            )

    for n_estimators in grid["gradient_boosting"]["n_estimators"]:
        for learning_rate in grid["gradient_boosting"]["learning_rate"]:
            for max_depth in grid["gradient_boosting"]["max_depth"]:
                experiments.append(
                    {
                        "run_name": (
                            f"gbr_est{n_estimators}_lr{learning_rate}_depth{max_depth}"
                        ),
                        "algorithm": "gradient_boosting",
                        "feature_set": "engineered",
                        "hyperparams": {
                            "n_estimators": n_estimators,
                            "learning_rate": learning_rate,
                            "max_depth": max_depth,
                            "random_state": random_state,
                        },
                        "log_target": log_target,
                    }
                )

    return experiments


def _run_naive_baseline_mlflow(params: dict[str, Any]) -> dict[str, Any]:
    train_df = pd.read_csv(CLEAN_TRAIN_PATH)
    training = params["training"]
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

    metrics = compute_regression_metrics(y_test.to_numpy(), predictions, log_scale=False)
    metrics["cv_mae"] = _cross_validate_mae(
        pipeline, X_numeric, y, training["cv_folds"], log_scale=False
    )
    metrics["n_features"] = int(X_numeric.shape[1])
    return {"model": pipeline, "metrics": metrics}


def _run_engineered_experiment(
    experiment: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: dict[str, Any],
) -> dict[str, Any]:
    algorithm = experiment["algorithm"]
    hyperparams = dict(experiment["hyperparams"])

    estimator = _build_estimator(algorithm, hyperparams)
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)

    log_scale = experiment["log_target"]
    metrics = compute_regression_metrics(y_test.to_numpy(), predictions, log_scale=log_scale)
    metrics["cv_mae"] = _cross_validate_mae(
        Pipeline([("model", estimator)]),
        X_train,
        y_train,
        params["training"]["cv_folds"],
        log_scale=log_scale,
    )
    metrics["n_features"] = int(X_train.shape[1])
    return {"model": estimator, "metrics": metrics}


def _log_run(
    experiment: dict[str, Any],
    result: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    metrics = result["metrics"]
    model = result["model"]

    with mlflow.start_run(run_name=experiment["run_name"]) as run:
        set_run_tags(
            algorithm=experiment["algorithm"],
            feature_set=experiment["feature_set"],
            project="house-price-prediction",
            phase="fase-5-mlflow",
        )
        log_model_params(
            {
                "algorithm": experiment["algorithm"],
                "feature_set": experiment["feature_set"],
                "log_target": experiment["log_target"],
                "cv_folds": params["training"]["cv_folds"],
                **experiment.get("hyperparams", {}),
            }
        )
        if "description" in result.get("metrics", {}):
            mlflow.set_tag("description", str(result["metrics"]["description"]))
        log_model_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
        current_run_id = run.info.run_id

    return {
        "run_id": current_run_id,
        "run_name": experiment["run_name"],
        "algorithm": experiment["algorithm"],
        "metrics": metrics,
    }


def run_training(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or load_params()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    experiment_name = setup_mlflow(params)
    experiments = build_experiment_grid(params)
    X_train, X_test, y_train, y_test = _load_engineered_data()

    logged_runs: list[dict[str, Any]] = []

    for experiment in experiments:
        if experiment["algorithm"] == "baseline_naive":
            result = _run_naive_baseline_mlflow(params)
        elif experiment["algorithm"] == "baseline_engineered":
            engineered = run_engineered_baseline(params)
            result = {
                "model": joblib.load(engineered["model_path"]),
                "metrics": engineered["metrics"],
            }
        else:
            result = _run_engineered_experiment(
                experiment, X_train, X_test, y_train, y_test, params
            )

        logged = _log_run(experiment, result, params)
        logged_runs.append(logged)
        print(
            f"Logged {logged['run_name']}: "
            f"MAE=${logged['metrics']['mae']:,.0f} | R²={logged['metrics']['r2']:.4f}"
        )

    engineered_runs = [
        run for run in logged_runs if run["algorithm"] not in {"baseline_naive", "baseline_engineered"}
    ]
    best_run = min(engineered_runs, key=lambda item: item["metrics"]["mae"])

    best_model_uri = f"runs:/{best_run['run_id']}/model"
    best_model = mlflow.sklearn.load_model(best_model_uri)
    joblib.dump(best_model, BEST_MODEL_PATH)

    summary = {
        "experiment_name": experiment_name,
        "total_runs": len(logged_runs),
        "best_run": best_run,
        "best_model_path": str(BEST_MODEL_PATH),
        "runs": logged_runs,
    }
    TRAINING_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nBest run: {best_run['run_name']} — MAE ${best_run['metrics']['mae']:,.0f}")
    print(f"Saved {BEST_MODEL_PATH}")
    print(f"Saved {TRAINING_SUMMARY_PATH}")
    print(f"MLflow UI: mlflow ui --port 5000")

    return summary


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()
