"""Optional local explanation helpers inspired by LIME."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.explainability.feature_importance import _extract_classifier_and_features

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:  # pragma: no cover - optional dependency
    LimeTabularExplainer = None


def explain_instance(model, X: pd.DataFrame, row_index: int = 0, num_features: int = 10) -> pd.DataFrame:
    """Explain one instance locally, with a fallback when LIME is unavailable."""
    if row_index >= len(X):
        raise IndexError("row_index is out of bounds for the provided dataframe.")

    classifier, transformed, feature_names = _extract_classifier_and_features(model, X)
    instance = np.asarray(transformed[row_index]).ravel()

    if LimeTabularExplainer is None:
        if hasattr(classifier, "coef_"):
            contributions = np.abs(np.asarray(classifier.coef_)).ravel() * np.abs(instance)
            return (
                pd.DataFrame(
                    {"feature": feature_names, "local_contribution": contributions, "source": "linear_fallback"}
                )
                .sort_values("local_contribution", ascending=False)
                .head(num_features)
                .reset_index(drop=True)
            )

        return pd.DataFrame(
            [{"feature": "LIME unavailable", "local_contribution": np.nan, "source": "not_installed"}]
        )

    dense_matrix = np.asarray(transformed)
    explainer = LimeTabularExplainer(
        training_data=dense_matrix,
        feature_names=feature_names,
        class_names=["benign", "malignant"],
        mode="classification",
    )
    explanation = explainer.explain_instance(instance, classifier.predict_proba, num_features=num_features)
    return pd.DataFrame(
        [{"feature": feature, "local_contribution": weight, "source": "lime"} for feature, weight in explanation.as_list()]
    )
