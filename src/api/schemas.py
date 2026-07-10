"""Pydantic schemas for the prediction API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    model_name: str | None = None
    model_version: str | None = None


class PredictionRequest(BaseModel):
    """Key Ames Housing features — remaining columns use training defaults."""

    model_config = ConfigDict(extra="forbid")

    GrLivArea: float = Field(..., ge=300, le=8000, description="Above grade living area (sq ft)")
    OverallQual: int = Field(..., ge=1, le=10, description="Overall material and finish quality")
    OverallCond: int = Field(default=5, ge=1, le=10, description="Overall condition")
    YearBuilt: int = Field(..., ge=1872, le=2026, description="Original construction date")
    YearRemodAdd: int | None = Field(default=None, ge=1872, le=2026)
    Neighborhood: str = Field(default="CollgCr", description="Neighborhood within Ames city limits")
    BldgType: str = Field(default="1Fam", description="Building type")
    HouseStyle: str = Field(default="1Story", description="House style")
    LotArea: float = Field(default=9600, ge=500, le=250000)
    LotFrontage: float | None = Field(default=70.0, ge=0)
    TotalBsmtSF: float = Field(default=900, ge=0)
    first_flr_sf: float = Field(default=1100, ge=0, alias="1stFlrSF")
    second_flr_sf: float = Field(default=0, ge=0, alias="2ndFlrSF")
    FullBath: int = Field(default=2, ge=0, le=5)
    HalfBath: int = Field(default=0, ge=0, le=3)
    BedroomAbvGr: int = Field(default=3, ge=0, le=8)
    KitchenQual: str = Field(default="TA", description="Kitchen quality")
    GarageCars: int = Field(default=2, ge=0, le=5)
    GarageArea: float = Field(default=400, ge=0)
    GarageYrBlt: float | None = Field(default=None)
    CentralAir: str = Field(default="Y", pattern="^[YN]$")
    ExterQual: str = Field(default="TA")
    Foundation: str = Field(default="PConc")
    MSZoning: str = Field(default="RL")
    SaleCondition: str = Field(default="Normal")


class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    confidence_interval_low: float | None = None
    confidence_interval_high: float | None = None
    model_name: str
    log_scale: bool


class BatchPredictionItem(BaseModel):
    id: int
    neighborhood: str
    gr_liv_area: float
    overall_qual: int
    year_built: int
    actual_price: float | None
    predicted_price: float
    error: float | None
    error_pct: float | None


class DashboardOverview(BaseModel):
    project_name: str
    tagline: str
    best_model: dict[str, Any]
    registry: dict[str, Any] | None
    baseline: dict[str, Any]
    pipeline_stages: list[dict[str, Any]]
    stats: dict[str, Any]


class AlgorithmComparison(BaseModel):
    algorithms: list[dict[str, Any]]
    significance_tests: list[dict[str, Any]]


class PipelineStage(BaseModel):
    id: str
    name: str
    description: str
    outputs: list[str]
    status: str


class ExampleComparison(BaseModel):
    examples: list[dict[str, Any]]
    scatter: list[dict[str, float]]
