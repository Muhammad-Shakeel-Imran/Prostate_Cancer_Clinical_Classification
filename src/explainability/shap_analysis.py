"""Optional SHAP-based explainability helpers with safe fallbacks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.explainability.feature_importance import (
    _extract_classifier_and_features,
    extract_native_feature_importance,
)

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None


def compute_shap_summary(model, X: pd.DataFrame, sample_size: int = 200) -> pd.DataFrame:
    """Compute mean absolute SHAP values, or fall back gracefully."""
    sample = X.sample(min(len(X), sample_size), random_state=42)

    if shap is None:
        fallback = extract_native_feature_importance(model, sample).rename(
            columns={"importance": "mean_abs_shap"}
        )
        fallback["source"] = "native_importance_fallback"
        return fallback

    classifier, transformed, feature_names = _extract_classifier_and_features(model, sample)

    if hasattr(classifier, "feature_importances_"):
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)
    else:
        explainer = shap.Explainer(classifier, transformed)
        shap_values = explainer(transformed).values

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    if getattr(shap_values, "ndim", 0) == 3:
        shap_values = shap_values[:, :, -1]

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(np.asarray(shap_values)).mean(axis=0),
                "source": "shap",
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
