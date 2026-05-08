"""General-purpose helpers for paths, serialization, and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and return it as a Path object."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    """Write a JSON document to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON document from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_dataframe(frame: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Write a dataframe to CSV and create parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=index)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def flatten_metrics(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Convert nested metric dictionaries into a tabular dataframe."""
    return pd.DataFrame.from_dict(results, orient="index").reset_index(names="model")
