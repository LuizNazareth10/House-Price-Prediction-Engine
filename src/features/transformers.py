"""Custom sklearn-compatible transformers for Ames Housing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DerivedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Create domain-specific features from raw Ames Housing columns."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DerivedFeaturesTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        if {"YrSold", "YearBuilt"}.issubset(df.columns):
            df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

        if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
            df["YearsSinceRemod"] = df["YrSold"] - df["YearRemodAdd"]

        if {"YrSold", "GarageYrBlt"}.issubset(df.columns):
            garage_age = df["YrSold"] - df["GarageYrBlt"]
            df["GarageAge"] = garage_age.where(df["GarageYrBlt"] > 0)

        area_cols = ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]
        if all(col in df.columns for col in area_cols):
            df["TotalSF"] = df[area_cols].sum(axis=1)

        if {"GrLivArea", "TotalBsmtSF"}.issubset(df.columns):
            df["LivAreaRatio"] = df["GrLivArea"] / df["TotalBsmtSF"].replace(0, np.nan)

        bath_cols = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
        if all(col in df.columns for col in bath_cols):
            df["TotalBath"] = (
                df["FullBath"]
                + 0.5 * df["HalfBath"]
                + df["BsmtFullBath"]
                + 0.5 * df["BsmtHalfBath"]
            )

        porch_cols = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
        existing_porch = [col for col in porch_cols if col in df.columns]
        if existing_porch:
            df["TotalPorchSF"] = df[existing_porch].sum(axis=1)

        if {"OverallQual", "GrLivArea"}.issubset(df.columns):
            df["QualLivArea"] = df["OverallQual"] * df["GrLivArea"]

        return df
