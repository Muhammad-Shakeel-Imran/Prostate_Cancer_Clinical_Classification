"""Statistical summaries that complement the model-driven features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_numeric_features(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for the numeric columns."""
    numeric = feature_frame.select_dtypes(include=[np.number])
    summary = numeric.describe().T
    summary["missing"] = numeric.isna().sum()
    return summary


def standardized_mean_differences(feature_frame: pd.DataFrame, target) -> pd.DataFrame:
    """Measure class separation feature-by-feature via standardized mean difference."""
    frame = feature_frame.copy()
    frame["target"] = target.values if hasattr(target, "values") else target
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.drop("target")

    rows = []
    for column in numeric_columns:
        class_zero = frame.loc[frame["target"] == 0, column]
        class_one = frame.loc[frame["target"] == 1, column]
        pooled_std = np.sqrt((class_zero.var() + class_one.var()) / 2.0)
        smd = 0.0 if pooled_std == 0 or np.isnan(pooled_std) else (class_one.mean() - class_zero.mean()) / pooled_std
        rows.append(
            {
                "feature": column,
                "mean_class_0": class_zero.mean(),
                "mean_class_1": class_one.mean(),
                "standardized_mean_difference": smd,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "standardized_mean_difference",
        key=lambda series: series.abs(),
        ascending=False,
    )


def correlation_matrix(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute a Pearson correlation matrix for the numeric variables."""
    numeric = feature_frame.select_dtypes(include=[np.number])
    return numeric.corr()
