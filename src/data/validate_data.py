"""Schema and quality checks for Ames Housing raw data."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_TRAIN_COLUMNS = (
    "SalePrice",
    "GrLivArea",
    "OverallQual",
    "YearBuilt",
    "Neighborhood",
)
OPTIONAL_ID_COLUMN = "Id"


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            message = "Data validation failed:\n" + "\n".join(f"- {err}" for err in self.errors)
            raise ValueError(message)


def validate_train(df: pd.DataFrame, min_rows: int = 1000) -> ValidationResult:
    result = ValidationResult(is_valid=True)

    if len(df) < min_rows:
        result.errors.append(f"Expected at least {min_rows} rows, got {len(df)}")
        result.is_valid = False

    missing_required = [col for col in REQUIRED_TRAIN_COLUMNS if col not in df.columns]
    if missing_required:
        result.errors.append(f"Missing required columns: {missing_required}")
        result.is_valid = False

    if "SalePrice" in df.columns:
        if df["SalePrice"].isna().any():
            result.errors.append("SalePrice contains missing values")
            result.is_valid = False
        if (df["SalePrice"] <= 0).any():
            result.errors.append("SalePrice must be positive")
            result.is_valid = False

    if OPTIONAL_ID_COLUMN in df.columns and df[OPTIONAL_ID_COLUMN].duplicated().any():
        result.errors.append("Duplicate values found in Id column")
        result.is_valid = False

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 30:
        result.warnings.append(
            f"Only {len(numeric_cols)} numeric columns detected; expected ~35+ for Ames Housing"
        )

    return result


def validate_test(df: pd.DataFrame, min_rows: int = 100) -> ValidationResult:
    result = ValidationResult(is_valid=True)

    if len(df) < min_rows:
        result.errors.append(f"Expected at least {min_rows} rows in test set, got {len(df)}")
        result.is_valid = False

    if "SalePrice" in df.columns:
        result.warnings.append("Test set contains SalePrice; expected features only")

    feature_cols = [col for col in REQUIRED_TRAIN_COLUMNS if col != "SalePrice"]
    missing_required = [col for col in feature_cols if col not in df.columns]
    if missing_required:
        result.errors.append(f"Missing required feature columns in test set: {missing_required}")
        result.is_valid = False

    return result
