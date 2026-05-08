"""Stacking ensemble utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import ModelEvaluator

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - depends on optional runtime package
    XGBClassifier = None

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """Stacking wrapper around scikit-learn's StackingClassifier."""

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
        self.cv_folds = cv_folds
        self.passthrough = passthrough
        self.random_state = random_state
        self.meta_learner = meta_learner or self._build_meta_learner(
            meta_learner_name,
            meta_params or {},
        )
        self.model = StackingClassifier(
            estimators=list(self.base_estimators.items()),
            final_estimator=self.meta_learner,
            stack_method="predict_proba",
            passthrough=self.passthrough,
            cv=StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state,
            ),
            n_jobs=None,
        )

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

    def fit(self, X_train, y_train) -> "StackingEnsemble":
        logger.info("Training stacking ensemble...")
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

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
