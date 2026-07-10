"""Tests for FastAPI prediction service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api import predictor as predictor_module


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "training_summary.json").write_text(
        json.dumps(
            {
                "total_runs": 33,
                "best_run": {
                    "run_name": "linear_regression_engineered",
                    "algorithm": "linear_regression",
                    "metrics": {"mae": 15367, "rmse": 24139, "r2": 0.924, "cv_mae": 17523},
                },
                "runs": [],
            }
        ),
        encoding="utf-8",
    )
    (reports / "baseline_metrics.json").write_text(
        json.dumps(
            {
                "naive": {"metrics": {"mae": 22979, "rmse": 36840}},
                "engineered": {"metrics": {"mae": 15367, "rmse": 24139}},
                "summary": {"mae_improvement_pct": 33.1, "conclusion": "Feature engineering helps."},
            }
        ),
        encoding="utf-8",
    )
    (reports / "registry_summary.json").write_text(
        json.dumps(
            {
                "registered_model_name": "house-price-predictor",
                "model_version": 1,
                "source_run_name": "linear_regression_engineered",
            }
        ),
        encoding="utf-8",
    )
    (reports / "significance_tests.json").write_text(json.dumps({"tests": []}), encoding="utf-8")

    monkeypatch.setattr("src.api.dashboard.REPORTS_DIR", reports)

    class FakeService:
        is_ready = True

        def predict_with_interval(self, features: dict) -> dict:
            area = features.get("GrLivArea", 1500)
            price = area * 120.0
            return {"price": price, "low": price - 15000, "high": price + 15000}

        def get_holdout_examples(self, n: int = 6) -> list:
            return [
                {
                    "id": 1,
                    "neighborhood": "CollgCr",
                    "gr_liv_area": 1710,
                    "overall_qual": 7,
                    "year_built": 2003,
                    "house_style": "2Story",
                    "actual_price": 208500,
                    "predicted_price": 205000,
                    "error": -3500,
                    "error_pct": -1.7,
                }
            ]

        def get_scatter_sample(self, limit: int = 80) -> list:
            return [{"actual": 200000, "predicted": 195000}]

        @property
        def metadata(self) -> dict:
            return {"model_name": "house-price-predictor", "model_version": "1", "log_target": True}

        def load(self) -> None:
            pass

    fake = FakeService()
    monkeypatch.setattr(predictor_module, "predictor_service", fake)
    import src.api.main as main_module

    monkeypatch.setattr(main_module, "predictor_service", fake)
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True


def test_predict_endpoint(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"GrLivArea": 1710, "OverallQual": 7, "YearBuilt": 2003},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_price"] == 1710 * 120.0
    assert payload["predicted_price_formatted"].startswith("$")


def test_dashboard_overview(client: TestClient) -> None:
    response = client.get("/api/v1/dashboard/overview")
    assert response.status_code == 200
    payload = response.json()
    assert payload["project_name"]
    assert len(payload["pipeline_stages"]) >= 5
    assert payload["registry"]["registered_model_name"] == "house-price-predictor"


def test_dashboard_algorithms(client: TestClient) -> None:
    response = client.get("/api/v1/dashboard/algorithms")
    assert response.status_code == 200
    assert "algorithms" in response.json()


def test_dashboard_examples(client: TestClient) -> None:
    response = client.get("/api/v1/dashboard/examples")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["examples"]) >= 1


def test_serve_dashboard(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
