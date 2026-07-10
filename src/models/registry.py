"""Promote the best MLflow run to the Model Registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.config import load_params
from src.utils.mlflow_utils import setup_mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
TRAINING_SUMMARY_PATH = REPORTS_DIR / "training_summary.json"
REGISTRY_SUMMARY_PATH = REPORTS_DIR / "registry_summary.json"


def _load_training_summary() -> dict[str, Any]:
    if not TRAINING_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Training summary not found at {TRAINING_SUMMARY_PATH}. Run train stage first."
        )
    return json.loads(TRAINING_SUMMARY_PATH.read_text(encoding="utf-8"))


def register_best_model(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or load_params()
    setup_mlflow(params)

    summary = _load_training_summary()
    best_run = summary["best_run"]
    model_name = params["mlflow"]["registered_model_name"]
    model_uri = f"runs:/{best_run['run_id']}/model"

    client = MlflowClient()
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)

    try:
        client.set_registered_model_alias(name=model_name, alias="Production", version=registered.version)
        stage = "Production"
    except Exception:
        client.transition_model_version_stage(
            name=model_name,
            version=registered.version,
            stage="Production",
            archive_existing_versions=True,
        )
        stage = "Production"

    result = {
        "registered_model_name": model_name,
        "model_version": registered.version,
        "stage": stage,
        "source_run_id": best_run["run_id"],
        "source_run_name": best_run["run_name"],
        "model_uri": model_uri,
        "metrics": best_run["metrics"],
    }

    REGISTRY_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_SUMMARY_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Registered {model_name} v{registered.version} -> Production")
    print(f"Source run: {best_run['run_name']} (MAE ${best_run['metrics']['mae']:,.0f})")
    print(f"Saved {REGISTRY_SUMMARY_PATH}")

    return result


def main() -> None:
    register_best_model()


if __name__ == "__main__":
    main()
