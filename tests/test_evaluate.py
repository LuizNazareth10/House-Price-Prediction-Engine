"""Tests for model evaluation and registry promotion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import (
    _run_significance_tests,
    _select_top_models,
    run_evaluation,
)
from src.models.registry import register_best_model


def test_select_top_models_excludes_baselines() -> None:
    summary = {
        "runs": [
            {"run_name": "baseline_naive", "algorithm": "baseline_naive", "metrics": {"mae": 1000}},
            {"run_name": "ridge_a", "algorithm": "ridge", "metrics": {"mae": 200}},
            {"run_name": "lasso_a", "algorithm": "lasso", "metrics": {"mae": 150}},
            {"run_name": "gbr_a", "algorithm": "gradient_boosting", "metrics": {"mae": 180}},
        ]
    }

    top = _select_top_models(summary, top_n=2)
    assert len(top) == 2
    assert top[0]["run_name"] == "lasso_a"
    assert top[1]["run_name"] == "gbr_a"


def test_run_significance_tests_returns_pairwise_results() -> None:
    fold_maes = {
        "model_a": [100.0, 102.0, 98.0, 101.0, 99.0],
        "model_b": [120.0, 118.0, 122.0, 119.0, 121.0],
        "model_c": [100.5, 101.0, 99.5, 100.0, 100.5],
    }

    tests = _run_significance_tests(fold_maes, alpha=0.05)
    assert len(tests) == 3
    assert all("p_value" in item for item in tests)

    ab_test = next(item for item in tests if item["model_a"] == "model_a" and item["model_b"] == "model_b")
    ac_test = next(item for item in tests if item["model_a"] == "model_a" and item["model_b"] == "model_c")
    assert ab_test["significant_at_alpha"] is True
    assert ac_test["significant_at_alpha"] is False


def test_run_evaluation_and_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    processed.mkdir()
    reports.mkdir()

    rng = np.random.default_rng(42)
    n_features = 3
    X = pd.DataFrame(rng.normal(size=(12, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(np.log1p(np.linspace(130000, 300000, 12)))
    X.to_csv(processed / "X_train.csv", index=False)
    y.to_frame("SalePrice").to_csv(processed / "y_train.csv", index=False)

    training_summary = {
        "experiment_name": "test-house-price",
        "total_runs": 4,
        "best_run": {
            "run_id": "run-best",
            "run_name": "linear_regression_engineered",
            "algorithm": "linear_regression",
            "metrics": {"mae": 100.0, "rmse": 150.0, "r2": 0.9, "cv_mae": 110.0},
        },
        "runs": [
            {
                "run_id": "run-best",
                "run_name": "linear_regression_engineered",
                "algorithm": "linear_regression",
                "metrics": {"mae": 100.0, "rmse": 150.0, "r2": 0.9, "cv_mae": 110.0},
            },
            {
                "run_id": "run-ridge",
                "run_name": "ridge_alpha_0.1",
                "algorithm": "ridge",
                "metrics": {"mae": 105.0, "rmse": 155.0, "r2": 0.89, "cv_mae": 115.0},
            },
            {
                "run_id": "run-lasso",
                "run_name": "lasso_alpha_0.1",
                "algorithm": "lasso",
                "metrics": {"mae": 108.0, "rmse": 160.0, "r2": 0.88, "cv_mae": 118.0},
            },
        ],
    }
    (reports / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.models.train.PROCESSED_DIR", processed)
    monkeypatch.setattr("src.models.train.TRAINING_SUMMARY_PATH", reports / "training_summary.json")
    monkeypatch.setattr("src.models.evaluate.TRAINING_SUMMARY_PATH", reports / "training_summary.json")
    monkeypatch.setattr("src.models.registry.TRAINING_SUMMARY_PATH", reports / "training_summary.json")
    monkeypatch.setattr("src.models.evaluate.REPORTS_DIR", reports)
    monkeypatch.setattr("src.models.registry.REPORTS_DIR", reports)
    monkeypatch.setattr(
        "src.models.evaluate.SIGNIFICANCE_PATH",
        reports / "significance_tests.json",
    )
    monkeypatch.setattr(
        "src.models.evaluate.COMPARISON_PATH",
        reports / "model_comparison.md",
    )
    monkeypatch.setattr(
        "src.models.evaluate.EVALUATION_METRICS_PATH",
        reports / "evaluation_metrics.json",
    )
    monkeypatch.setattr(
        "src.models.registry.REGISTRY_SUMMARY_PATH",
        reports / "registry_summary.json",
    )

    def fake_register(params=None):
        payload = {
            "registered_model_name": "house-price-predictor",
            "model_version": "1",
            "stage": "Production",
            "source_run_id": "run-best",
            "source_run_name": "linear_regression_engineered",
            "model_uri": "runs:/run-best/model",
            "metrics": training_summary["best_run"]["metrics"],
        }
        (reports / "registry_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    monkeypatch.setattr("src.models.evaluate.register_best_model", fake_register)

    params = {
        "training": {"cv_folds": 3, "random_state": 42},
        "evaluation": {"top_n_models": 2, "significance_level": 0.05},
        "features": {"log_target": True},
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

    result = run_evaluation(params)

    assert len(result["top_models"]) == 2
    assert len(result["significance_tests"]) == 1
    assert (reports / "significance_tests.json").exists()
    assert (reports / "model_comparison.md").exists()
    assert (reports / "evaluation_metrics.json").exists()
    assert "Model Registry" in (reports / "model_comparison.md").read_text(encoding="utf-8")
