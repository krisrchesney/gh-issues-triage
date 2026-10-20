"""Config loading. One YAML file drives every stage so projects only change config, not code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config. Raises FileNotFoundError with a clear message if it is missing."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")
    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh)
    for section in ("lake", "data", "model", "mlflow", "drift"):
        if section not in cfg:
            raise ValueError(f"Config is missing required section: '{section}'")
    return cfg
