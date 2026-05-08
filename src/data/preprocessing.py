"""Preprocessing utilities for mixed clinical tabular data."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_feature_types(feature_frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split features into numeric and categorical columns."""
    numeric_columns = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        column for column in feature_frame.columns if column not in numeric_columns
    ]
    return numeric_columns, categorical_columns


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    """Build the shared preprocessing block used by every model pipeline."""
    numeric_columns, categorical_columns = infer_feature_types(feature_frame)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = [("numeric", numeric_pipeline, numeric_columns)]
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
