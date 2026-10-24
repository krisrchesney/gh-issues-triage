"""Evidently HTML drift report (optional extra: pip install -e ".[monitor]").

The PSI gate in drift.py decides whether to retrain. This report is the human-readable view that
goes in the README and the demo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MISSING_MSG = "Evidently is not installed. Run: pip install -e '.[monitor]'"


def evidently_html(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str], out_path: str | Path
) -> Path:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        # Report the real cause: an installed but incompatible Evidently also raises ImportError.
        raise RuntimeError(f"{MISSING_MSG} (import failed: {exc})") from exc
    snapshot = Report([DataDriftPreset()]).run(current[features], reference[features])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out))
    return out
