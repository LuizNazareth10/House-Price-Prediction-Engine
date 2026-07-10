"""Tests for feature engineering pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_preprocessor, build_feature_datasets
from src.features.transformers import DerivedFeaturesTransformer


def _mini_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GrLivArea": [1500, 1800, 1200],
            "OverallQual": [7, 8, 5],
            "YearBuilt": [2000, 1990, 2010],
            "YrSold": [2010, 2010, 2010],
            "YearRemodAdd": [2000, 1995, 2010],
            "GarageYrBlt": [2000, 0, 2010],
            "TotalBsmtSF": [800, 600, 500],
            "1stFlrSF": [900, 1000, 700],
            "2ndFlrSF": [600, 800, 500],
            "FullBath": [2, 2, 1],
            "HalfBath": [1, 0, 1],
            "BsmtFullBath": [1, 1, 0],
            "BsmtHalfBath": [0, 0, 0],
            "WoodDeckSF": [100, 0, 50],
            "OpenPorchSF": [20, 30, 10],
            "EnclosedPorch": [0, 0, 0],
            "3SsnPorch": [0, 0, 0],
            "ScreenPorch": [0, 10, 0],
            "Neighborhood": ["NAmes", "CollgCr", "Edwards"],
            "SalePrice": [210000.0, 250000.0, 150000.0],
        }
    )


def test_derived_features_transformer() -> None:
    df = _mini_train().drop(columns=["SalePrice"])
    transformed = DerivedFeaturesTransformer().fit_transform(df)

    assert "HouseAge" in transformed.columns
    assert "TotalSF" in transformed.columns
    assert "QualLivArea" in transformed.columns
    assert transformed.loc[1, "GarageAge"] is pd.NA or np.isnan(transformed.loc[1, "GarageAge"])


def test_build_preprocessor_shapes() -> None:
    df = DerivedFeaturesTransformer().fit_transform(_mini_train().drop(columns=["SalePrice"]))
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = build_preprocessor(numeric, categorical, use_polynomial=False, polynomial_degree=2)
    matrix = preprocessor.fit_transform(df)

    assert matrix.shape[0] == 3
    assert matrix.shape[1] > len(numeric)


def test_build_feature_datasets_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    interim.mkdir()
    processed.mkdir()

    train = _mini_train()
    inference = train.drop(columns=["SalePrice"]).head(2)

    monkeypatch.setattr("src.features.build_features.INTERIM_DIR", interim)
    monkeypatch.setattr("src.features.build_features.PROCESSED_DIR", processed)
    monkeypatch.setattr("src.features.build_features.CLEAN_TRAIN_PATH", interim / "clean.csv")
    monkeypatch.setattr("src.features.build_features.CLEAN_TEST_PATH", interim / "clean_test.csv")

    train.to_csv(interim / "clean.csv", index=False)
    inference.to_csv(interim / "clean_test.csv", index=False)

    params = {
        "training": {"test_size": 0.34, "random_state": 42},
        "features": {
            "use_polynomial": False,
            "polynomial_degree": 2,
            "log_target": True,
        },
    }

    metadata = build_feature_datasets(params)

    assert (processed / "X_train.csv").exists()
    assert (processed / "X_test.csv").exists()
    assert (processed / "y_train.csv").exists()
    assert (processed / "y_test.csv").exists()
    assert (processed / "X_inference.csv").exists()
    assert (processed / "preprocessor.pkl").exists()
    assert metadata["n_features"] > 0
    assert metadata["log_target"] is True

    saved_metadata = json.loads((processed / "features_metadata.json").read_text(encoding="utf-8"))
    assert saved_metadata["n_train_rows"] + saved_metadata["n_test_rows"] == 3
