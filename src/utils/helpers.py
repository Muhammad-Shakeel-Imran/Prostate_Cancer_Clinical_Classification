"""General-purpose helpers for paths, serialization, and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    """Write a JSON document to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def save_dataframe(frame: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Write a dataframe to CSV and create parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=index)
