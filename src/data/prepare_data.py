"""Prepare raw Ames Housing data for downstream feature engineering."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.load_data import load_test, load_train
from src.data.validate_data import validate_test, validate_train
from src.utils.config import load_params

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

CLEAN_TRAIN_PATH = INTERIM_DIR / "clean.csv"
CLEAN_TEST_PATH = INTERIM_DIR / "clean_test.csv"
PROFILE_PATH = INTERIM_DIR / "data_profile.json"


def _drop_high_missing_columns(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    missing_ratio = df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    cols_to_drop = [col for col in cols_to_drop if col != "SalePrice"]
    return df.drop(columns=cols_to_drop), cols_to_drop


def _prepare_split(df: pd.DataFrame, params: dict, *, is_train: bool) -> tuple[pd.DataFrame, dict]:
    data_params = params["data"]
    working = df.copy()

    id_column = data_params.get("id_column")
    if id_column and id_column in working.columns:
        working = working.drop(columns=[id_column])

    working, dropped_cols = _drop_high_missing_columns(
        working,
        threshold=data_params["missing_threshold"],
    )

    profile = {
        "rows": int(len(working)),
        "columns": int(working.shape[1]),
        "dropped_high_missing_columns": dropped_cols,
        "missing_columns": int(working.isna().any().sum()),
        "is_train": is_train,
    }

    if is_train and "SalePrice" in working.columns:
        profile["target_skewness"] = float(working["SalePrice"].skew())

    return working, profile


def prepare_datasets(params: dict | None = None) -> dict:
    params = params or load_params()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    train = load_train()
    test = load_test()

    train_validation = validate_train(train, min_rows=params["data"]["min_train_rows"])
    test_validation = validate_test(test, min_rows=params["data"]["min_test_rows"])
    train_validation.raise_if_invalid()
    test_validation.raise_if_invalid()

    clean_train, train_profile = _prepare_split(train, params, is_train=True)
    clean_test, test_profile = _prepare_split(test, params, is_train=False)

    clean_train.to_csv(CLEAN_TRAIN_PATH, index=False)
    clean_test.to_csv(CLEAN_TEST_PATH, index=False)

    profile = {
        "train": train_profile,
        "test": test_profile,
        "validation_warnings": train_validation.warnings + test_validation.warnings,
    }
    PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print(f"Saved {CLEAN_TRAIN_PATH} ({clean_train.shape})")
    print(f"Saved {CLEAN_TEST_PATH} ({clean_test.shape})")
    print(f"Saved {PROFILE_PATH}")

    return profile


def main() -> None:
    prepare_datasets()


if __name__ == "__main__":
    main()
