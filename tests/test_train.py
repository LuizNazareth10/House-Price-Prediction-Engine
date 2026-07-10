"""Tests for MLflow training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.models.train import build_experiment_grid, run_training


def test_build_experiment_grid_has_minimum_runs() -> None:
    params = {
        "training": {"random_state": 42},
        "features": {"log_target": True},
        "experiments": {
            "ridge": {"alpha": [0.1, 1.0]},
            "lasso": {"alpha": [0.1]},
            "elasticnet": {"alpha": [0.1], "l1_ratio": [0.5]},
            "random_forest": {"n_estimators": [50], "max_depth": [5]},
            "gradient_boosting": {
                "n_estimators": [50],
                "learning_rate": [0.1],
                "max_depth": [3],
            },
        },
    }

    experiments = build_experiment_grid(params)
    assert len(experiments) >= 8
    algorithms = {item["algorithm"] for item in experiments}
    assert "ridge" in algorithms
    assert "gradient_boosting" in algorithms


def test_run_training_logs_runs_to_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    mlruns = tmp_path / "mlruns"
    for folder in (interim, processed, models, reports, mlruns):
        folder.mkdir()

    clean = pd.DataFrame(
        {
            "GrLivArea": np.linspace(1000, 2500, 20),
            "OverallQual": np.tile([5, 6, 7, 8], 5),
            "YearBuilt": np.linspace(1980, 2010, 20),
            "Neighborhood": ["A", "B", "C", "D"] * 5,
            "SalePrice": np.linspace(120000, 320000, 20),
        }
    )
    clean.to_csv(interim / "clean.csv", index=False)

    rng = np.random.default_rng(42)
    n_features = 4
    X = pd.DataFrame(rng.normal(size=(16, n_features)), columns=[f"f{i}" for i in range(n_features)])
    X_holdout = pd.DataFrame(rng.normal(size=(4, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(np.log1p(np.linspace(130000, 300000, 16)))
    y_holdout = pd.Series(np.log1p(np.linspace(140000, 280000, 4)))

    X.to_csv(processed / "X_train.csv", index=False)
    X_holdout.to_csv(processed / "X_test.csv", index=False)
    y.to_frame("SalePrice").to_csv(processed / "y_train.csv", index=False)
    y_holdout.to_frame("SalePrice").to_csv(processed / "y_test.csv", index=False)

    monkeypatch.setattr("src.models.train.CLEAN_TRAIN_PATH", interim / "clean.csv")
    monkeypatch.setattr("src.models.train.PROCESSED_DIR", processed)
    monkeypatch.setattr("src.models.train.MODELS_DIR", models)
    monkeypatch.setattr("src.models.train.REPORTS_DIR", reports)
    monkeypatch.setattr("src.models.train.BEST_MODEL_PATH", models / "best_model.pkl")
    monkeypatch.setattr("src.models.train.TRAINING_SUMMARY_PATH", reports / "training_summary.json")
    monkeypatch.setattr("src.models.baseline.CLEAN_TRAIN_PATH", interim / "clean.csv")
    monkeypatch.setattr("src.models.baseline.PROCESSED_DIR", processed)
    monkeypatch.setattr("src.models.baseline.MODELS_DIR", models)
    monkeypatch.setattr("src.models.baseline.ENGINEERED_MODEL_PATH", models / "baseline_engineered.pkl")

    params = {
        "data": {"id_column": "Id", "missing_threshold": 0.5, "min_train_rows": 3, "min_test_rows": 2},
        "training": {"test_size": 0.2, "random_state": 42, "cv_folds": 3},
        "features": {"log_target": True},
        "mlflow": {
            "tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            "experiment_name": "test-house-price",
        },
        "experiments": {
            "ridge": {"alpha": [0.1]},
            "lasso": {"alpha": [0.1]},
            "elasticnet": {"alpha": [0.1], "l1_ratio": [0.5]},
            "random_forest": {"n_estimators": [20], "max_depth": [3]},
            "gradient_boosting": {
                "n_estimators": [20],
                "learning_rate": [0.1],
                "max_depth": [2],
            },
        },
    }

    summary = run_training(params)

    assert summary["total_runs"] >= 8
    assert (models / "best_model.pkl").exists()
    assert (reports / "training_summary.json").exists()

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    experiment = mlflow.get_experiment_by_name("test-house-price")
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == summary["total_runs"]

    saved = json.loads((reports / "training_summary.json").read_text(encoding="utf-8"))
    assert "best_run" in saved
