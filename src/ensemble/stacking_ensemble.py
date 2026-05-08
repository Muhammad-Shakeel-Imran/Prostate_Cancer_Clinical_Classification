"""Stacking ensemble utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.data.preprocessing import build_preprocessor
from src.evaluation.metrics import ModelEvaluator

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - depends on optional runtime package
    XGBClassifier = None

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """Custom stacking ensemble with safe passthrough preprocessing."""

    def __init__(
        self,
        base_estimators: Dict,
        meta_learner=None,
        meta_learner_name: str = "logistic_regression",
        meta_params: Dict | None = None,
        cv_folds: int = 5,
        passthrough: bool = True,
        random_state: int = 42,
    ):
        self.base_estimators = base_estimators
        self.model_names = list(base_estimators.keys())
        self.cv_folds = cv_folds
        self.passthrough = passthrough
        self.random_state = random_state
        self.meta_learner = meta_learner or self._build_meta_learner(
            meta_learner_name,
            meta_params or {},
        )
        self.fitted_base_estimators_ = {}
        self.meta_learner_ = None
        self.passthrough_preprocessor_ = None
        self.is_fitted_ = False

    def _build_meta_learner(self, meta_learner_name: str, meta_params: Dict):
        if meta_learner_name == "logistic_regression":
            params = dict(meta_params)
            params.setdefault("random_state", self.random_state)
            return LogisticRegression(**params)

        if meta_learner_name == "xgboost":
            if XGBClassifier is None:
                raise ImportError("xgboost is not installed. Install it to use it as meta-learner.")
            params = dict(meta_params)
            params.setdefault("random_state", self.random_state)
            params.setdefault("objective", "binary:logistic")
            params.setdefault("eval_metric", "logloss")
            return XGBClassifier(**params)

        raise ValueError(f"Unsupported stacking meta-learner: {meta_learner_name}")

    @staticmethod
    def _slice_rows(X, indices):
        """Slice pandas objects or numpy arrays by row index."""
        if hasattr(X, "iloc"):
            return X.iloc[indices]
        return X[indices]

    def _fit_passthrough_preprocessor(self, X_train):
        """Fit the auxiliary preprocessor used by the meta-model passthrough path."""
        if not self.passthrough:
            self.passthrough_preprocessor_ = None
            return None

        self.passthrough_preprocessor_ = build_preprocessor(X_train)
        return self.passthrough_preprocessor_.fit_transform(X_train)

    def _build_meta_features(self, X, use_passthrough: bool = True):
        """Create the meta-model design matrix from base-model predictions."""
        if not self.fitted_base_estimators_:
            raise ValueError("Base estimators are not fitted. Call fit() first.")

        probability_columns = [
            self.fitted_base_estimators_[model_name].predict_proba(X)[:, 1].reshape(-1, 1)
            for model_name in self.model_names
        ]
        meta_matrix = np.hstack(probability_columns)

        if use_passthrough and self.passthrough and self.passthrough_preprocessor_ is not None:
            passthrough_matrix = self.passthrough_preprocessor_.transform(X)
            meta_matrix = np.hstack([meta_matrix, np.asarray(passthrough_matrix)])

        return meta_matrix

    def fit(self, X_train, y_train) -> "StackingEnsemble":
        logger.info("Training stacking ensemble with %s base estimators...", len(self.model_names))
        splitter = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        y_array = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)
        oof_probability_columns = []
        self.fitted_base_estimators_ = {}

        for model_name in self.model_names:
            estimator = self.base_estimators[model_name]
            oof_predictions = np.zeros(len(y_array), dtype=float)

            for train_idx, valid_idx in splitter.split(X_train, y_array):
                estimator_fold = clone(estimator)
                X_fold_train = self._slice_rows(X_train, train_idx)
                X_fold_valid = self._slice_rows(X_train, valid_idx)
                y_fold_train = self._slice_rows(y_train, train_idx)
                estimator_fold.fit(X_fold_train, y_fold_train)
                oof_predictions[valid_idx] = estimator_fold.predict_proba(X_fold_valid)[:, 1]

            full_estimator = clone(estimator)
            full_estimator.fit(X_train, y_train)
            self.fitted_base_estimators_[model_name] = full_estimator
            oof_probability_columns.append(oof_predictions.reshape(-1, 1))

        meta_matrix = np.hstack(oof_probability_columns)
        passthrough_matrix = self._fit_passthrough_preprocessor(X_train)
        if passthrough_matrix is not None:
            meta_matrix = np.hstack([meta_matrix, np.asarray(passthrough_matrix)])

        self.meta_learner_ = clone(self.meta_learner)
        self.meta_learner_.fit(meta_matrix, y_train)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted_ or self.meta_learner_ is None:
            raise ValueError("Stacking ensemble is not fitted. Call fit() first.")
        meta_matrix = self._build_meta_features(X)
        return self.meta_learner_.predict_proba(meta_matrix)

    def predict(self, X, threshold: float = 0.5):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)

    def evaluate(self, X, y, threshold: float = 0.5):
        y_proba = self.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        metrics = ModelEvaluator.evaluate_model(y, y_pred, y_proba)
        logger.info("Stacking ensemble metrics: %s", metrics)
        return metrics

    def save(self, save_path: str | Path) -> None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        logger.info("Saved stacking ensemble to %s", output_path)

    @staticmethod
    def load(save_path: str | Path) -> "StackingEnsemble":
        return joblib.load(save_path)
