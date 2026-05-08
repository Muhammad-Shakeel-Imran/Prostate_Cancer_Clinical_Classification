"""Hyperparameter tuning utilities for the explicit model family."""

from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.models.base_model_trainer import BaseModelTrainer


DEFAULT_SEARCH_SPACES: Dict[str, Dict[str, Iterable]] = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1.0, 3.0, 10.0],
    },
    "random_forest": {
        "classifier__n_estimators": [200, 400, 600],
        "classifier__max_depth": [None, 6, 10, 14],
        "classifier__min_samples_split": [2, 4, 6],
    },
    "xgboost": {
        "classifier__n_estimators": [200, 300, 500],
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__subsample": [0.75, 0.85, 1.0],
    },
    "lightgbm": {
        "classifier__n_estimators": [200, 300, 500],
        "classifier__num_leaves": [15, 31, 63],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
    },
    "svm": {
        "classifier__C": [0.5, 1.0, 2.0, 5.0],
        "classifier__gamma": ["scale", 0.01, 0.05, 0.1],
    },
    "knn": {
        "classifier__n_neighbors": [5, 7, 9, 11, 15],
        "classifier__weights": ["uniform", "distance"],
    },
}


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y,
    model_config: Dict,
    param_distributions: Dict[str, Iterable] | None = None,
    n_iter: int = 10,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> RandomizedSearchCV:
    """Tune one model family using randomized search on the project pipeline."""
    trainer = BaseModelTrainer(model_config=model_config, random_state=random_state)
    estimator = trainer.build_estimators(X)[model_name]
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions or DEFAULT_SEARCH_SPACES[model_name],
        n_iter=n_iter,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    return search
