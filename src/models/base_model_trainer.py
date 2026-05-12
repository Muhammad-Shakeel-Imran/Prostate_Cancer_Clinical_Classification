"""Training utilities for the interview-ready ensemble base learners."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data.preprocessing import build_preprocessor

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - depends on optional runtime package
    XGBClassifier = None

logger = logging.getLogger(__name__)


class BaseModelTrainer:
    """Train, evaluate, and persist the base models used in the ensemble."""

    def __init__(self, model_config: Dict, random_state: int = 42):
        self.model_config = model_config
        self.random_state = random_state
        self.trained_models: Dict[str, Pipeline] = {}

    def _enabled_model_configs(self) -> Dict[str, Dict]:
        return {
            name: params
            for name, params in self.model_config["models"].items()
            if params.get("enabled", True)
        }

    def _build_classifier(self, model_name: str):
        params = dict(self.model_config["models"][model_name])
        params.pop("enabled", None)

        if model_name == "logistic_regression":
            params.setdefault("random_state", self.random_state)
            return LogisticRegression(**params)

        if model_name == "random_forest":
            params.setdefault("random_state", self.random_state)
            return RandomForestClassifier(**params)

        if model_name == "xgboost":
            if XGBClassifier is None:
                raise ImportError("xgboost is not installed. Install it to enable this model.")
            params.setdefault("random_state", self.random_state)
            return XGBClassifier(**params)

        if model_name == "svm":
            params.setdefault("random_state", self.random_state)
            return SVC(**params)

        if model_name == "knn":
            return KNeighborsClassifier(**params)

        raise ValueError(f"Unsupported model: {model_name}")

    def build_estimators(self, feature_frame: pd.DataFrame) -> Dict[str, Pipeline]:
        """Build unfitted preprocessing + model pipelines for every enabled model."""
        estimators: Dict[str, Pipeline] = {}
        for model_name in self._enabled_model_configs():
            estimator = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(feature_frame)),
                    ("classifier", self._build_classifier(model_name)),
                ]
            )
            estimators[model_name] = estimator
        return estimators

    def cross_validate_models(self, X: pd.DataFrame, y, cv_folds: int = 5) -> pd.DataFrame:
        """Benchmark the base models with stratified cross-validation."""
        scoring = {
            "roc_auc": "roc_auc",
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        }
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        rows = {}
        for model_name, estimator in self.build_estimators(X).items():
            logger.info("Cross-validating %s...", model_name)
            scores = cross_validate(
                estimator,
                X,
                y,
                cv=splitter,
                scoring=scoring,
                return_train_score=False,
            )
            rows[model_name] = {
                metric.replace("test_", "") + "_mean": float(values.mean())
                for metric, values in scores.items()
                if metric.startswith("test_")
            }
            rows[model_name].update(
                {
                    metric.replace("test_", "") + "_std": float(values.std())
                    for metric, values in scores.items()
                    if metric.startswith("test_")
                }
            )

        results = pd.DataFrame.from_dict(rows, orient="index")
        if not results.empty and "roc_auc_mean" in results.columns:
            results = results.sort_values(by="roc_auc_mean", ascending=False)
        return results

    def train_all_models(self, X_train: pd.DataFrame, y_train) -> Dict[str, Pipeline]:
        """Fit every enabled base learner on the full training split."""
        self.trained_models = {}
        for model_name, estimator in self.build_estimators(X_train).items():
            logger.info("Training %s...", model_name)
            estimator.fit(X_train, y_train)
            self.trained_models[model_name] = estimator

        logger.info("Trained %s base models", len(self.trained_models))
        return self.trained_models

    def save_models(self, save_dir: str | Path) -> None:
        """Persist fitted base models individually for later inspection or reuse."""
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model in self.trained_models.items():
            model_path = output_dir / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info("Saved %s model to %s", model_name, model_path)
