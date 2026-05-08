"""Validation helpers for the prostate cancer dataset."""

from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

from .feature_engineering import (
    ID_COLUMN,
    INFERENCE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)


def _missing_columns(columns: Iterable[str], expected_columns: Iterable[str]) -> list[str]:
    return sorted(set(expected_columns) - set(columns))


def validate_training_frame(dataframe: pd.DataFrame) -> None:
    """Validate the full dataset used for model training."""
    missing_columns = _missing_columns(dataframe.columns, REQUIRED_COLUMNS)
    if missing_columns:
        raise ValueError(f"Training data is missing required columns: {missing_columns}")

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    if ID_COLUMN in dataframe.columns and dataframe[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate patient ids detected in the dataset.")


def validate_inference_frame(dataframe: pd.DataFrame) -> None:
    """Validate new patient data before inference."""
    missing_columns = _missing_columns(dataframe.columns, INFERENCE_COLUMNS)
    if missing_columns:
        raise ValueError(f"Inference data is missing required columns: {missing_columns}")


def summarize_dataset(dataframe: pd.DataFrame) -> Dict[str, object]:
    """Return a compact dataset summary for reports."""
    summary = {
        "n_rows": int(dataframe.shape[0]),
        "n_columns": int(dataframe.shape[1]),
        "class_distribution": {
            str(label): int(count)
            for label, count in dataframe[TARGET_COLUMN].value_counts().to_dict().items()
        },
        "missing_values": {
            str(column): int(count)
            for column, count in dataframe.isna().sum().to_dict().items()
        },
    }
    return summary
