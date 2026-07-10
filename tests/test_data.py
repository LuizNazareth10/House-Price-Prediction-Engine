"""Tests for data loading, validation and preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.prepare_data import _drop_high_missing_columns, _prepare_split
from src.data.validate_data import validate_test, validate_train


def _sample_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "SalePrice": [200000.0, 150000.0, 300000.0],
            "GrLivArea": [1500, 1200, 2200],
            "OverallQual": [7, 6, 8],
            "YearBuilt": [2000, 1995, 2010],
            "Neighborhood": ["NAmes", "CollgCr", "NAmes"],
            "PoolQC": [None, None, None],
        }
    )


def _sample_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": [4, 5],
            "GrLivArea": [1100, 1800],
            "OverallQual": [5, 7],
            "YearBuilt": [1980, 2005],
            "Neighborhood": ["Edwards", "Gilbert"],
        }
    )


def test_validate_train_success() -> None:
    result = validate_train(_sample_train(), min_rows=3)
    assert result.is_valid
    assert not result.errors


def test_validate_train_missing_target() -> None:
    df = _sample_train().drop(columns=["SalePrice"])
    result = validate_train(df, min_rows=3)
    assert not result.is_valid
    assert any("SalePrice" in err for err in result.errors)


def test_validate_test_success() -> None:
    result = validate_test(_sample_test(), min_rows=2)
    assert result.is_valid


def test_drop_high_missing_columns() -> None:
    df = _sample_train()
    cleaned, dropped = _drop_high_missing_columns(df, threshold=0.5)
    assert "PoolQC" in dropped
    assert "SalePrice" in cleaned.columns


def test_prepare_split_removes_id() -> None:
    params = {
        "data": {
            "id_column": "Id",
            "missing_threshold": 0.5,
        }
    }
    cleaned, profile = _prepare_split(_sample_train(), params, is_train=True)
    assert "Id" not in cleaned.columns
    assert profile["rows"] == 3


def test_data_profile_written_after_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()

    monkeypatch.setattr("src.data.prepare_data.INTERIM_DIR", interim_dir)
    monkeypatch.setattr("src.data.prepare_data.CLEAN_TRAIN_PATH", interim_dir / "clean.csv")
    monkeypatch.setattr("src.data.prepare_data.CLEAN_TEST_PATH", interim_dir / "clean_test.csv")
    monkeypatch.setattr("src.data.prepare_data.PROFILE_PATH", interim_dir / "data_profile.json")
    monkeypatch.setattr("src.data.prepare_data.load_train", lambda: _sample_train())
    monkeypatch.setattr("src.data.prepare_data.load_test", lambda: _sample_test())

    params = {
        "data": {
            "id_column": "Id",
            "missing_threshold": 0.5,
            "min_train_rows": 3,
            "min_test_rows": 2,
        }
    }

    from src.data.prepare_data import prepare_datasets

    profile = prepare_datasets(params)
    assert profile["train"]["rows"] == 3
    assert (interim_dir / "data_profile.json").exists()

    saved_profile = json.loads((interim_dir / "data_profile.json").read_text(encoding="utf-8"))
    assert "train" in saved_profile
    assert "test" in saved_profile
