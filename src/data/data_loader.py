"""Data loading and preparation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .feature_engineering import (
    TARGET_COLUMN,
    build_feature_frame,
    encode_target,
    normalize_columns,
)
from .validation import summarize_dataset, validate_inference_frame, validate_training_frame

logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    """Container for the raw data, engineered features, and target vector."""

    raw_frame: pd.DataFrame
    feature_frame: pd.DataFrame
    target: pd.Series
    summary: Dict[str, Any]


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_data(data_path: str | Path, sheet_name: Optional[str | int] = 0) -> pd.DataFrame:
    """Load tabular data from CSV or Excel."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    elif suffix == ".csv":
        dataframe = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported data format: {suffix}")

    logger.info("Loaded data from %s with shape %s", path, dataframe.shape)
    return dataframe


def prepare_training_dataset(dataframe: pd.DataFrame) -> DatasetBundle:
    """Validate the dataset and construct the model feature matrix."""
    raw_frame = normalize_columns(dataframe)
    validate_training_frame(raw_frame)
    feature_frame = build_feature_frame(raw_frame)
    target = encode_target(raw_frame[TARGET_COLUMN])
    summary = summarize_dataset(raw_frame)

    logger.info("Prepared feature matrix with shape %s", feature_frame.shape)
    return DatasetBundle(
        raw_frame=raw_frame,
        feature_frame=feature_frame,
        target=target.rename("target"),
        summary=summary,
    )


def load_and_prepare_training_data(
    data_path: str | Path,
    sheet_name: Optional[str | int] = 0,
) -> DatasetBundle:
    """Convenience wrapper to load and transform the training dataset."""
    dataframe = load_data(data_path=data_path, sheet_name=sheet_name)
    return prepare_training_dataset(dataframe)


def prepare_inference_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build the same feature set for new patient records."""
    normalized = normalize_columns(dataframe)
    validate_inference_frame(normalized)
    feature_frame = build_feature_frame(normalized)
    logger.info("Prepared inference feature matrix with shape %s", feature_frame.shape)
    return feature_frame


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Create a stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info("Train set: %s, Test set: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test
