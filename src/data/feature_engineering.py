"""Feature engineering helpers for the prostate cancer tabular dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "diagnosis_result"
ID_COLUMN = "id"

RAW_NUMERIC_COLUMNS = [
    "age",
    "BMI",
    "PSA_level",
    "free_PSA_ratio",
    "testosterone_level",
    "alkaline_phosphatase",
    "family_history",
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "symmetry",
    "fractal_dimension",
]
RAW_CATEGORICAL_COLUMNS = ["smoking_status"]
REQUIRED_COLUMNS = [ID_COLUMN, *RAW_NUMERIC_COLUMNS, *RAW_CATEGORICAL_COLUMNS, TARGET_COLUMN]
INFERENCE_COLUMNS = [ID_COLUMN, *RAW_NUMERIC_COLUMNS, *RAW_CATEGORICAL_COLUMNS]
TARGET_MAPPING = {"B": 0, "M": 1}


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardize column labels to avoid whitespace-related bugs."""
    normalized = dataframe.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    return normalized


def encode_target(target: pd.Series) -> pd.Series:
    """Map the diagnosis label to an integer target."""
    encoded = target.astype(str).str.strip().str.upper()
    unknown_labels = sorted(set(encoded) - set(TARGET_MAPPING))
    if unknown_labels:
        raise ValueError(
            f"Unsupported target labels: {unknown_labels}. "
            f"Expected one of {sorted(TARGET_MAPPING)}."
        )
    return encoded.map(TARGET_MAPPING).astype(int)


def coerce_numeric_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert expected numeric columns while preserving categorical variables."""
    coerced = dataframe.copy()
    for column in RAW_NUMERIC_COLUMNS:
        if column in coerced.columns:
            coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced


def build_feature_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create a model-ready feature frame with domain-inspired interaction terms.

    The additional features are intentionally lightweight and interpretable so the
    project remains strong in an interview setting rather than looking overfit.
    """
    features = normalize_columns(dataframe)
    features = coerce_numeric_types(features)
    features = features.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")

    epsilon = 1e-6

    features["log_psa"] = np.log1p(features["PSA_level"].clip(lower=0))
    features["log_alkaline_phosphatase"] = np.log1p(
        features["alkaline_phosphatase"].clip(lower=0)
    )
    features["psa_age_ratio"] = features["PSA_level"] / features["age"].clip(lower=epsilon)
    features["psa_bmi_interaction"] = features["PSA_level"] * features["BMI"]
    features["psa_to_testosterone_ratio"] = features["PSA_level"] / features[
        "testosterone_level"
    ].clip(lower=epsilon)
    features["free_to_total_psa_index"] = features["free_PSA_ratio"] / features[
        "PSA_level"
    ].clip(lower=epsilon)
    features["alkaline_phosphatase_to_psa_ratio"] = features[
        "alkaline_phosphatase"
    ] / features["PSA_level"].clip(lower=epsilon)
    features["area_perimeter_ratio"] = features["area"] / features["perimeter"].clip(
        lower=epsilon
    )
    features["texture_radius_ratio"] = features["texture"] / features["radius"].clip(
        lower=epsilon
    )
    features["compactness_symmetry_interaction"] = (
        features["compactness"] * features["symmetry"]
    )
    features["shape_risk_index"] = (
        features["radius"] * features["perimeter"] * features["compactness"]
    )

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    return features
