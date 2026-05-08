"""Soft voting ensemble utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import ModelEvaluator

logger = logging.getLogger(__name__)


class VotingEnsemble:
    """Soft voting over already-fitted base models."""

    def __init__(self, models: Dict, weights: Dict[str, float] | None = None):
        self.models = models
        self.model_names = list(models.keys())
        self.weights = self._normalize_weights(weights)

    def _normalize_weights(self, weights: Dict[str, float] | None) -> Dict[str, float]:
        if not self.models:
            return {}
        if weights is None:
            equal_weight = 1.0 / len(self.models)
            return {name: equal_weight for name in self.models}

        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("Voting weights must sum to a positive value.")
        return {name: value / total_weight for name, value in weights.items()}

    @staticmethod
    def derive_weights(
        cv_results: pd.DataFrame,
        metric: str = "roc_auc",
        power: float = 1.0,
    ) -> Dict[str, float]:
        """Turn cross-validated model quality into normalized voting weights."""
        if cv_results.empty:
            return {}

        metric_column = f"{metric}_mean" if f"{metric}_mean" in cv_results.columns else metric
        raw_weights = {}
        for model_name, row in cv_results.iterrows():
            score = float(row[metric_column])
            baseline = score - 0.5 if "auc" in metric else score
            raw_weights[model_name] = max(baseline, 1e-3) ** power

        total = sum(raw_weights.values())
        return {name: value / total for name, value in raw_weights.items()}

    def predict_proba(self, X) -> np.ndarray:
        """Average per-model class probabilities using the configured weights."""
        ensemble_proba = np.zeros((X.shape[0], 2))
        for model_name, model in self.models.items():
            ensemble_proba += self.weights[model_name] * model.predict_proba(X)
        return ensemble_proba

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)

    def evaluate(self, X, y, threshold: float = 0.5) -> Dict[str, float]:
        y_proba = self.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        metrics = ModelEvaluator.evaluate_model(y, y_pred, y_proba)
        logger.info("Voting ensemble metrics: %s", metrics)
        return metrics

    def save(self, save_path: str | Path) -> None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        logger.info("Saved voting ensemble to %s", output_path)

    @staticmethod
    def load(save_path: str | Path) -> "VotingEnsemble":
        return joblib.load(save_path)
