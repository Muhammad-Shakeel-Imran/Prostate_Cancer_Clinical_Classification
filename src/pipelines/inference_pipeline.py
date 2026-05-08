"""Inference pipeline for ensemble and single-model predictions."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from src.data.data_loader import prepare_inference_data
from src.ensemble.stacking_ensemble import StackingEnsemble
from src.ensemble.voting_ensemble import VotingEnsemble

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Load persisted models and score new patient records."""

    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.base_models = {}
        self.voting_ensemble = None
        self.stacking_ensemble = None

    def load_models(self) -> None:
        """Load base models plus the saved ensemble artifacts."""
        trained_models_dir = self.model_dir / "trained_models"
        ensemble_models_dir = self.model_dir / "ensemble_models"

        for model_path in sorted(trained_models_dir.glob("*_model.pkl")):
            model_name = model_path.stem.replace("_model", "")
            self.base_models[model_name] = joblib.load(model_path)
            logger.info("Loaded base model: %s", model_name)

        voting_path = ensemble_models_dir / "soft_voting_ensemble.pkl"
        if voting_path.exists():
            self.voting_ensemble = VotingEnsemble.load(voting_path)
            logger.info("Loaded soft voting ensemble")

        stacking_path = ensemble_models_dir / "stacking_ensemble.pkl"
        if stacking_path.exists():
            self.stacking_ensemble = StackingEnsemble.load(stacking_path)
            logger.info("Loaded stacking ensemble")

    def preprocess_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature engineering used during training."""
        return prepare_inference_data(X)

    def predict(self, X: pd.DataFrame, method: str = "stacking", threshold: float = 0.5):
        """
        Score new examples using a selected model family.

        Supported values for ``method`` are ``stacking``, ``voting``, or any saved
        base-model name such as ``logistic_regression`` or ``xgboost``.
        """
        logger.info("Generating predictions with %s", method)
        features = self.preprocess_input(X)

        if method == "stacking":
            if self.stacking_ensemble is None:
                raise ValueError("Stacking ensemble is not loaded.")
            probabilities = self.stacking_ensemble.predict_proba(features)[:, 1]
        elif method == "voting":
            if self.voting_ensemble is None:
                raise ValueError("Voting ensemble is not loaded.")
            probabilities = self.voting_ensemble.predict_proba(features)[:, 1]
        else:
            if method not in self.base_models:
                raise ValueError(f"Unknown model '{method}'.")
            probabilities = self.base_models[method].predict_proba(features)[:, 1]

        predictions = (probabilities >= threshold).astype(int)
        return predictions, probabilities
