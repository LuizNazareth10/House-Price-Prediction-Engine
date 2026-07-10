"""Load project configuration from params.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params(path: Path | None = None) -> dict[str, Any]:
    params_file = path or PARAMS_PATH
    if not params_file.exists():
        raise FileNotFoundError(f"Params file not found: {params_file}")

    with params_file.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
