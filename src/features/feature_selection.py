"""Feature selection helpers for tabular prostate cancer modelling."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
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
