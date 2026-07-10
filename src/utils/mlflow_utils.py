"""MLflow experiment tracking helpers."""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np


def setup_mlflow(params: dict[str, Any]) -> str:
    """Configure tracking URI and experiment. Returns experiment name."""
    mlflow_config = params.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = mlflow_config.get("experiment_name", "house-price-prediction")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return experiment_name


def log_model_metrics(metrics: dict[str, float], *, prefix: str = "") -> None:
    for name, value in metrics.items():
        if not isinstance(value, (int, float, np.integer, np.floating)):
            continue
        key = f"{prefix}{name}" if prefix else name
        mlflow.log_metric(key, float(value))


def log_model_params(params: dict[str, Any]) -> None:
    flat_params = {
        key: "null" if value is None else value
        for key, value in params.items()
    }
    mlflow.log_params(flat_params)


def set_run_tags(**tags: str) -> None:
    for key, value in tags.items():
        mlflow.set_tag(key, value)
