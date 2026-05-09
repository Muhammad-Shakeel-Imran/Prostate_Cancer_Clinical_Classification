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


def _native_or_placeholder_summary(model, X: pd.DataFrame, source: str) -> pd.DataFrame:
    """Return native importance output when available, otherwise a placeholder row."""
    try:
        fallback = extract_native_feature_importance(model, X).rename(
            columns={"importance": "mean_abs_shap"}
        )
        fallback["source"] = source
        return fallback
    except ValueError:
        return pd.DataFrame(
            [
                {
                    "feature": "SHAP unavailable for this model",
                    "mean_abs_shap": np.nan,
                    "source": source,
                }
            ]
        )


def compute_shap_summary(model, X: pd.DataFrame, sample_size: int = 200) -> pd.DataFrame:
    """Compute mean absolute SHAP values, or fall back gracefully."""
    sample = X.sample(min(len(X), sample_size), random_state=42)

    if shap is None:
        return _native_or_placeholder_summary(model, sample, "shap_not_installed")

    classifier, transformed, feature_names = _extract_classifier_and_features(model, sample)
    transformed = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)

    try:
        if hasattr(classifier, "feature_importances_"):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed)
        else:
            prediction_fn = (
                classifier.predict_proba if hasattr(classifier, "predict_proba") else classifier.predict
            )
            explainer = shap.Explainer(prediction_fn, transformed)
            shap_values = explainer(transformed).values
    except Exception:
        return _native_or_placeholder_summary(model, sample, "native_importance_fallback")

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
