"""Lightweight file-based model registry.

Layout:
    models/<version>/model.joblib
    models/<version>/model_card.json
    models/<version>/reference.parquet   (training features, used as the drift baseline)
    models/current.json                  ({"version": "<version>"}, the promoted model)

Experiments are tracked in MLflow; this registry only decides what the API serves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class NoModelError(RuntimeError):
    """Raised when no promoted model exists yet."""


def promote(registry_dir: str | Path, version: str) -> None:
    root = Path(registry_dir)
    if not (root / version / "model.joblib").exists():
        raise FileNotFoundError(f"Cannot promote unknown version: {version}")
    (root / "current.json").write_text(json.dumps({"version": version}))


def current_version(registry_dir: str | Path) -> str:
    pointer = Path(registry_dir) / "current.json"
    if not pointer.exists():
        raise NoModelError(f"No promoted model in {Path(registry_dir).resolve()}. Run `train` first.")
    return json.loads(pointer.read_text())["version"]


def load_current(registry_dir: str | Path) -> tuple[Any, dict[str, Any], str]:
    """Return (model, model_card, version) for the promoted model."""
    version = current_version(registry_dir)
    folder = Path(registry_dir) / version
    model = joblib.load(folder / "model.joblib")
    card = json.loads((folder / "model_card.json").read_text())
    return model, card, version


def load_reference(registry_dir: str | Path) -> pd.DataFrame:
    version = current_version(registry_dir)
    return pd.read_parquet(Path(registry_dir) / version / "reference.parquet")
