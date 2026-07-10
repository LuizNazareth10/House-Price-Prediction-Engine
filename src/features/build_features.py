"""Build processed feature matrices from interim clean datasets."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from src.features.transformers import DerivedFeaturesTransformer
from src.utils.config import load_params

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLEAN_TRAIN_PATH = INTERIM_DIR / "clean.csv"
CLEAN_TEST_PATH = INTERIM_DIR / "clean_test.csv"

TARGET_COLUMN = "SalePrice"


def _split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_df = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    numeric_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    use_polynomial: bool,
    polynomial_degree: int,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if use_polynomial:
        numeric_steps.append(("poly", PolynomialFeatures(degree=polynomial_degree, include_bias=False)))
    numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)


def _transform_target(y: pd.Series, log_target: bool) -> pd.Series:
    if log_target:
        return pd.Series(np.log1p(y), index=y.index)
    return y


def build_feature_datasets(params: dict | None = None) -> dict:
    params = params or load_params()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(CLEAN_TRAIN_PATH)
    inference_df = pd.read_csv(CLEAN_TEST_PATH)

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {CLEAN_TRAIN_PATH}")

    X = train_df.drop(columns=[TARGET_COLUMN])
    y = train_df[TARGET_COLUMN]

    training_params = params["training"]
    feature_params = params["features"]

    X_train, X_holdout, y_train_raw, y_holdout_raw = train_test_split(
        X,
        y,
        test_size=training_params["test_size"],
        random_state=training_params["random_state"],
    )

    derived_sample = DerivedFeaturesTransformer().fit_transform(X_train.head(1))
    numeric_features, categorical_features = _split_feature_types(derived_sample)

    preprocessor = Pipeline(
        [
            ("derived", DerivedFeaturesTransformer()),
            (
                "preprocess",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                    use_polynomial=feature_params["use_polynomial"],
                    polynomial_degree=feature_params.get("polynomial_degree", 2),
                ),
            ),
        ]
    )

    X_train_matrix = preprocessor.fit_transform(X_train)
    X_holdout_matrix = preprocessor.transform(X_holdout)
    X_inference_matrix = preprocessor.transform(inference_df)

    feature_names = preprocessor.named_steps["preprocess"].get_feature_names_out().tolist()

    log_target = feature_params["log_target"]
    y_train = _transform_target(y_train_raw, log_target)
    y_holdout = _transform_target(y_holdout_raw, log_target)

    X_train_out = pd.DataFrame(X_train_matrix, columns=feature_names)
    X_holdout_out = pd.DataFrame(X_holdout_matrix, columns=feature_names)
    X_inference_out = pd.DataFrame(X_inference_matrix, columns=feature_names)
    y_train_out = pd.DataFrame({TARGET_COLUMN: y_train.reset_index(drop=True)})
    y_holdout_out = pd.DataFrame({TARGET_COLUMN: y_holdout.reset_index(drop=True)})

    paths = {
        "X_train": PROCESSED_DIR / "X_train.csv",
        "X_test": PROCESSED_DIR / "X_test.csv",
        "y_train": PROCESSED_DIR / "y_train.csv",
        "y_test": PROCESSED_DIR / "y_test.csv",
        "X_inference": PROCESSED_DIR / "X_inference.csv",
        "preprocessor": PROCESSED_DIR / "preprocessor.pkl",
        "metadata": PROCESSED_DIR / "features_metadata.json",
    }

    X_train_out.to_csv(paths["X_train"], index=False)
    X_holdout_out.to_csv(paths["X_test"], index=False)
    y_train_out.to_csv(paths["y_train"], index=False)
    y_holdout_out.to_csv(paths["y_test"], index=False)
    X_inference_out.to_csv(paths["X_inference"], index=False)
    joblib.dump(preprocessor, paths["preprocessor"])

    metadata = {
        "n_features": len(feature_names),
        "n_train_rows": int(len(X_train_out)),
        "n_test_rows": int(len(X_holdout_out)),
        "n_inference_rows": int(len(X_inference_out)),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "derived_features": [
            "HouseAge",
            "YearsSinceRemod",
            "GarageAge",
            "TotalSF",
            "LivAreaRatio",
            "TotalBath",
            "TotalPorchSF",
            "QualLivArea",
        ],
        "log_target": log_target,
        "use_polynomial": feature_params["use_polynomial"],
        "polynomial_degree": feature_params.get("polynomial_degree", 2),
        "test_size": training_params["test_size"],
        "random_state": training_params["random_state"],
        "feature_names_sample": feature_names[:20],
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for name, path in paths.items():
        if path.suffix in {".csv", ".pkl"}:
            print(f"Saved {path}")

    print(f"Saved {paths['metadata']}")
    return metadata


def main() -> None:
    build_feature_datasets()


if __name__ == "__main__":
    main()
