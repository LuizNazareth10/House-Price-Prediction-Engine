"""Download Ames Housing dataset into data/raw/."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

COMPETITION = "house-prices-advanced-regression-techniques"
REQUIRED_FILES = ("train.csv", "test.csv", "data_description.txt")
OPTIONAL_FILES = ("sample_submission.csv",)

OPENML_DATA_ID = 42165
DESCRIPTION_URL = (
    "https://raw.githubusercontent.com/jeffheaton/"
    "t81_558_deep_learning/master/keras/datasets/ames/data_description.txt"
)


def _files_ready() -> bool:
    return all((RAW_DATA_DIR / filename).exists() for filename in REQUIRED_FILES)


def download_from_kaggle(force: bool = False) -> Path:
    import kagglehub

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not force and _files_ready():
        print(f"Dataset already present in {RAW_DATA_DIR}")
        return RAW_DATA_DIR

    print(f"Downloading competition from Kaggle: {COMPETITION}")
    cache_path = Path(kagglehub.competition_download(COMPETITION))
    print(f"Kaggle cache path: {cache_path}")

    for filename in REQUIRED_FILES + OPTIONAL_FILES:
        source = cache_path / filename
        if not source.exists():
            if filename in OPTIONAL_FILES:
                print(f"Skipping optional file not found: {filename}")
                continue
            raise FileNotFoundError(f"Expected file not found in cache: {source}")

        destination = RAW_DATA_DIR / filename
        shutil.copy2(source, destination)
        print(f"Copied {filename} -> {destination}")

    return RAW_DATA_DIR


def download_from_openml() -> Path:
    from urllib.request import urlopen

    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading training data from OpenML (id={OPENML_DATA_ID})...")
    features, target = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, return_X_y=True)

    train = features.copy()
    train["SalePrice"] = target.astype(float)

    train_path = RAW_DATA_DIR / "train.csv"
    train.to_csv(train_path, index=False)
    print(f"Saved {train_path} ({train.shape[0]} rows, {train.shape[1]} columns)")

    # OpenML não fornece o test.csv oficial do Kaggle; criamos split local só para EDA inicial.
    train_split, test_split = train_test_split(
        train.drop(columns=["SalePrice"]),
        test_size=0.2,
        random_state=42,
    )
    test_path = RAW_DATA_DIR / "test.csv"
    test_split.to_csv(test_path, index=False)
    print(f"Saved {test_path} (split local para EDA; não é o test oficial do Kaggle)")

    description_path = RAW_DATA_DIR / "data_description.txt"
    try:
        with urlopen(DESCRIPTION_URL, timeout=30) as response:
            description_path.write_bytes(response.read())
        print(f"Saved {description_path}")
    except Exception as exc:
        description_path.write_text(
            "Ames Housing dataset description unavailable offline.\n"
            "See Kaggle competition data_description.txt for full column docs.\n",
            encoding="utf-8",
        )
        print(f"Saved fallback description file ({exc})")

    return RAW_DATA_DIR


def download_raw_data(force: bool = False, source: str = "auto") -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not force and _files_ready():
        print(f"Dataset already present in {RAW_DATA_DIR}")
        return RAW_DATA_DIR

    if source == "openml":
        download_from_openml()
    elif source == "kaggle":
        download_from_kaggle(force=force)
    else:
        try:
            download_from_kaggle(force=force)
        except Exception as exc:
            print(f"Kaggle download failed: {exc}")
            print("Trying OpenML fallback...")
            download_from_openml()

    print(f"Done. Raw files available in {RAW_DATA_DIR}")
    return RAW_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Ames Housing dataset")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite files even if they already exist",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "kaggle", "openml"),
        default="auto",
        help="Data source (default: auto = Kaggle with OpenML fallback)",
    )
    args = parser.parse_args()
    download_raw_data(force=args.force, source=args.source)


if __name__ == "__main__":
    main()
