"""Global feature-importance utilities for fitted model pipelines."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def _extract_classifier_and_features(model, X: pd.DataFrame) -> Tuple[object, np.ndarray, list[str]]:
    """Return the classifier, transformed feature matrix, and feature names."""
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        transformed = preprocessor.transform(X)
        feature_names = preprocessor.get_feature_names_out().tolist()
        return classifier, transformed, feature_names

    feature_names = (
        X.columns.tolist()
        if isinstance(X, pd.DataFrame)
        else [f"feature_{i}" for i in range(X.shape[1])]
    )
    return model, X, feature_names


def extract_native_feature_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    """Extract native feature importance or coefficient magnitudes."""
    classifier, _, feature_names = _extract_classifier_and_features(model, X)

    if hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_)
    elif hasattr(classifier, "coef_"):
        importances = np.abs(np.asarray(classifier.coef_)).ravel()
    else:
        raise ValueError("Model does not expose native feature importances.")

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def compute_permutation_importance_summary(
    model,
    X: pd.DataFrame,
    y,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance on the original feature frame."""
    scores = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": scores.importances_mean,
                "importance_std": scores.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
