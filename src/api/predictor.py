"""Model and preprocessor loading with inference helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import load_params

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
CLEAN_TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "clean.csv"

TARGET_COLUMN = "SalePrice"
PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
REGISTRY_SUMMARY_PATH = REPORTS_DIR / "registry_summary.json"


class PredictorService:
    """Lazy-loaded inference service backed by preprocessor + best model."""

    def __init__(self) -> None:
        self._preprocessor = None
        self._model = None
        self._defaults: dict[str, Any] | None = None
        self._log_target = True
        self._model_name = "house-price-predictor"
        self._model_version: str | None = None
        self._mae: float | None = None

    @property
    def is_ready(self) -> bool:
        return self._preprocessor is not None and self._model is not None

    def load(self) -> None:
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}. Run dvc repro featurize.")
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {BEST_MODEL_PATH}. Run dvc repro train.")

        params = load_params()
        self._log_target = bool(params["features"]["log_target"])
        self._preprocessor = joblib.load(PREPROCESSOR_PATH)
        self._model = joblib.load(BEST_MODEL_PATH)
        self._defaults = self._build_defaults()
        self._load_registry_metadata()
        self._load_mae_from_reports()

    def _build_defaults(self) -> dict[str, Any]:
        df = pd.read_csv(CLEAN_TRAIN_PATH).drop(columns=[TARGET_COLUMN], errors="ignore")
        defaults: dict[str, Any] = {}
        for column in df.columns:
            series = df[column]
            if pd.api.types.is_numeric_dtype(series):
                defaults[column] = float(series.median()) if series.notna().any() else 0.0
            else:
                mode = series.mode()
                defaults[column] = mode.iloc[0] if not mode.empty else "NA"
        return defaults

    def _load_registry_metadata(self) -> None:
        if not REGISTRY_SUMMARY_PATH.exists():
            return
        payload = json.loads(REGISTRY_SUMMARY_PATH.read_text(encoding="utf-8"))
        self._model_name = payload.get("registered_model_name", self._model_name)
        self._model_version = str(payload.get("model_version", ""))

    def _load_mae_from_reports(self) -> None:
        summary_path = REPORTS_DIR / "training_summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self._mae = float(payload["best_run"]["metrics"]["mae"])

    def _to_dataframe(self, features: dict[str, Any]) -> pd.DataFrame:
        row = dict(self._defaults or {})
        row.update(features)
        if row.get("YearRemodAdd") is None:
            row["YearRemodAdd"] = row.get("YearBuilt", 2000)
        if row.get("GarageYrBlt") is None:
            row["GarageYrBlt"] = row.get("YearBuilt", 2000)
        return pd.DataFrame([row])

    def predict_row(self, features: dict[str, Any]) -> float:
        if not self.is_ready:
            self.load()
        assert self._preprocessor is not None and self._model is not None

        matrix = self._preprocessor.transform(self._to_dataframe(features))
        raw = float(self._model.predict(matrix)[0])
        if self._log_target:
            return float(np.expm1(raw))
        return raw

    def predict_with_interval(self, features: dict[str, Any]) -> dict[str, float]:
        price = self.predict_row(features)
        low, high = price, price
        if self._mae:
            low = max(0.0, price - self._mae)
            high = price + self._mae
        return {"price": price, "low": low, "high": high}

    def get_holdout_examples(self, n: int = 6) -> list[dict[str, Any]]:
        if not self.is_ready:
            self.load()

        params = load_params()
        df = pd.read_csv(CLEAN_TRAIN_PATH)
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        _, X_holdout, _, y_holdout = train_test_split(
            X,
            y,
            test_size=params["training"]["test_size"],
            random_state=params["training"]["random_state"],
        )

        holdout = X_holdout.copy()
        holdout[TARGET_COLUMN] = y_holdout.values
        holdout = holdout.reset_index(drop=True)

        indices = self._select_diverse_indices(holdout[TARGET_COLUMN], n)
        examples: list[dict[str, Any]] = []

        for idx, row_idx in enumerate(indices):
            row = holdout.iloc[row_idx]
            actual = float(row[TARGET_COLUMN])
            feature_dict = row.drop(labels=[TARGET_COLUMN]).to_dict()
            predicted = self.predict_row(feature_dict)
            error = predicted - actual
            examples.append(
                {
                    "id": idx + 1,
                    "row_index": int(row_idx),
                    "neighborhood": str(row.get("Neighborhood", "—")),
                    "gr_liv_area": float(row.get("GrLivArea", 0)),
                    "overall_qual": int(row.get("OverallQual", 0)),
                    "year_built": int(row.get("YearBuilt", 0)),
                    "house_style": str(row.get("HouseStyle", "—")),
                    "actual_price": actual,
                    "predicted_price": predicted,
                    "error": error,
                    "error_pct": (error / actual) * 100 if actual else None,
                }
            )
        return examples

    @staticmethod
    def _select_diverse_indices(prices: pd.Series, n: int) -> list[int]:
        sorted_idx = prices.sort_values().index.tolist()
        if len(sorted_idx) <= n:
            return sorted_idx
        step = max(1, len(sorted_idx) // n)
        picks = [sorted_idx[i] for i in range(0, len(sorted_idx), step)][: n - 1]
        picks.append(sorted_idx[len(sorted_idx) // 2])
        return sorted(set(picks))[:n]

    def get_scatter_sample(self, limit: int = 80) -> list[dict[str, float]]:
        if not self.is_ready:
            self.load()

        params = load_params()
        df = pd.read_csv(CLEAN_TRAIN_PATH)
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        _, X_holdout, _, y_holdout = train_test_split(
            X,
            y,
            test_size=params["training"]["test_size"],
            random_state=params["training"]["random_state"],
        )

        points: list[dict[str, float]] = []
        for i in range(min(limit, len(X_holdout))):
            actual = float(y_holdout.iloc[i])
            predicted = self.predict_row(X_holdout.iloc[i].to_dict())
            points.append({"actual": actual, "predicted": predicted})
        return points

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "model_version": self._model_version,
            "log_target": self._log_target,
            "mae": self._mae,
        }


predictor_service = PredictorService()
