"""Uncertainty estimation helpers for probabilistic ensemble predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def predictive_entropy(probabilities) -> np.ndarray:
    """Compute binary predictive entropy for positive-class probabilities."""
    probabilities = np.clip(np.asarray(probabilities), 1e-8, 1 - 1e-8)
    return -(probabilities * np.log(probabilities) + (1 - probabilities) * np.log(1 - probabilities))


def confidence_margin(probabilities) -> np.ndarray:
    """Distance from the decision boundary, where smaller means more uncertain."""
    probabilities = np.asarray(probabilities)
    return np.abs(probabilities - 0.5)


def ensemble_disagreement(probability_map: dict[str, np.ndarray]) -> np.ndarray:
    """Per-sample standard deviation across model probabilities."""
    stacked = np.column_stack(list(probability_map.values()))
    return stacked.std(axis=1)


def build_uncertainty_frame(probability_map: dict[str, np.ndarray]) -> pd.DataFrame:
    """Assemble several uncertainty signals into one dataframe."""
    if not probability_map:
        return pd.DataFrame()

    mean_probability = np.column_stack(list(probability_map.values())).mean(axis=1)
    return pd.DataFrame(
        {
            "mean_probability": mean_probability,
            "predictive_entropy": predictive_entropy(mean_probability),
            "confidence_margin": confidence_margin(mean_probability),
            "ensemble_disagreement": ensemble_disagreement(probability_map),
        }
    )
