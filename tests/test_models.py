"""Tests for baseline model training and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.models.baseline import run_baselines
from src.models.metrics import compute_regression_metrics


def test_compute_regression_metrics_log_scale() -> None:
    y_true = np.log1p(np.array([200000.0, 150000.0, 300000.0]))
    y_pred = np.log1p(np.array([210000.0, 140000.0, 290000.0]))

    metrics = compute_regression_metrics(y_true, y_pred, log_scale=True)

    assert metrics["mae"] > 0
    assert -1.0 <= metrics["r2"] <= 1.0
    assert metrics["rmse"] >= metrics["mae"]


def test_run_baselines_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    interim.mkdir()
    processed.mkdir()
    models.mkdir()
    reports.mkdir()

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
    n_features = 5
    X = pd.DataFrame(rng.normal(size=(16, n_features)), columns=[f"f{i}" for i in range(n_features)])
    X_holdout = pd.DataFrame(rng.normal(size=(4, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(np.log1p(np.linspace(130000, 300000, 16)))
    y_holdout = pd.Series(np.log1p(np.linspace(140000, 280000, 4)))

    X.to_csv(processed / "X_train.csv", index=False)
    X_holdout.to_csv(processed / "X_test.csv", index=False)
    y.to_frame("SalePrice").to_csv(processed / "y_train.csv", index=False)
    y_holdout.to_frame("SalePrice").to_csv(processed / "y_test.csv", index=False)

    monkeypatch.setattr("src.models.baseline.INTERIM_DIR", interim)
    monkeypatch.setattr("src.models.baseline.PROCESSED_DIR", processed)
    monkeypatch.setattr("src.models.baseline.MODELS_DIR", models)
    monkeypatch.setattr("src.models.baseline.REPORTS_DIR", reports)
    monkeypatch.setattr("src.models.baseline.CLEAN_TRAIN_PATH", interim / "clean.csv")
    monkeypatch.setattr("src.models.baseline.NAIVE_MODEL_PATH", models / "baseline_naive.pkl")
    monkeypatch.setattr("src.models.baseline.ENGINEERED_MODEL_PATH", models / "baseline_engineered.pkl")
    monkeypatch.setattr("src.models.baseline.METRICS_PATH", reports / "baseline_metrics.json")
    monkeypatch.setattr("src.models.baseline.COMPARISON_PATH", reports / "baseline_comparison.md")

    params = {
        "training": {"test_size": 0.2, "random_state": 42, "cv_folds": 3},
        "features": {"log_target": True},
    }

    results = run_baselines(params)

    assert (models / "baseline_naive.pkl").exists()
    assert (models / "baseline_engineered.pkl").exists()
    assert (reports / "baseline_metrics.json").exists()
    assert (reports / "baseline_comparison.md").exists()
    assert results["engineered"]["metrics"]["n_features"] == n_features
    assert "summary" in results

    saved = json.loads((reports / "baseline_metrics.json").read_text(encoding="utf-8"))
    assert "naive" in saved and "engineered" in saved


def test_cross_validate_mae_uses_dollar_scale() -> None:
    from src.models.baseline import _cross_validate_mae

    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    y = pd.Series(np.log1p([100.0, 120.0, 140.0, 160.0, 180.0, 200.0]))
    pipeline = Pipeline([("model", LinearRegression())])

    cv_mae = _cross_validate_mae(pipeline, X, y, cv_folds=3, log_scale=True)
    assert cv_mae > 0
