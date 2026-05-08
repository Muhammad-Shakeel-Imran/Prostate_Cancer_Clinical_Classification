"""Feature selection helpers for tabular prostate cancer modelling."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer


def _prepare_dense_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Create a dense numeric design matrix for filter-based feature ranking."""
    encoded = pd.get_dummies(X.copy(), drop_first=False)
    imputer = SimpleImputer(strategy="median")
    transformed = imputer.fit_transform(encoded)
    return pd.DataFrame(transformed, columns=encoded.columns, index=encoded.index)


def rank_features_by_mutual_info(X: pd.DataFrame, y) -> pd.DataFrame:
    """Rank features using mutual information against the binary diagnosis label."""
    matrix = _prepare_dense_matrix(X)
    scores = mutual_info_classif(matrix, y, random_state=42)
    ranking = pd.DataFrame(
        {"feature": matrix.columns, "mutual_information": scores}
    ).sort_values("mutual_information", ascending=False)
    return ranking.reset_index(drop=True)


def rank_features_by_f_score(X: pd.DataFrame, y) -> pd.DataFrame:
    """Rank features using ANOVA F-statistics."""
    matrix = _prepare_dense_matrix(X)
    scores, p_values = f_classif(matrix, y)
    ranking = pd.DataFrame(
        {"feature": matrix.columns, "f_score": scores, "p_value": p_values}
    ).sort_values("f_score", ascending=False)
    return ranking.reset_index(drop=True)


def select_top_features(
    X: pd.DataFrame,
    y,
    k: int = 10,
    strategy: str = "mutual_info",
) -> pd.DataFrame:
    """Return a reduced feature matrix with the top-k ranked variables."""
    matrix = _prepare_dense_matrix(X)
    scorer = mutual_info_classif if strategy == "mutual_info" else f_classif
    selector = SelectKBest(score_func=scorer, k=min(k, matrix.shape[1]))
    reduced = selector.fit_transform(matrix, y)
    selected_columns = matrix.columns[selector.get_support()].tolist()
    return pd.DataFrame(reduced, columns=selected_columns, index=X.index)
