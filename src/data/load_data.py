"""Helpers to load Ames Housing raw files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
DESCRIPTION_PATH = RAW_DATA_DIR / "data_description.txt"


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}. "
            "Run `python scripts/download_data.py` first."
        )


def load_train() -> pd.DataFrame:
    _ensure_exists(TRAIN_PATH)
    return pd.read_csv(TRAIN_PATH)


def load_test() -> pd.DataFrame:
    _ensure_exists(TEST_PATH)
    return pd.read_csv(TEST_PATH)


def load_data_description() -> str:
    _ensure_exists(DESCRIPTION_PATH)
    return DESCRIPTION_PATH.read_text(encoding="utf-8")
