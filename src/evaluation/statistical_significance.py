"""Bootstrap-based significance checks for model comparison."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


METRIC_FUNCTIONS = {
    "roc_auc": roc_auc_score,
    "average_precision": average_precision_score,
}


def bootstrap_metric_difference(
    y_true,
    y_proba_a,
    y_proba_b,
    metric: str = "roc_auc",
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict[str, float]:
    """Estimate the distribution of the performance gap between two models."""
    rng = np.random.default_rng(random_state)
    metric_fn = METRIC_FUNCTIONS[metric]
    y_true = np.asarray(y_true)
    y_proba_a = np.asarray(y_proba_a)
    y_proba_b = np.asarray(y_proba_b)

    differences = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        sample_y = y_true[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        score_a = metric_fn(sample_y, y_proba_a[indices])
        score_b = metric_fn(sample_y, y_proba_b[indices])
        differences.append(score_a - score_b)

    if not differences:
        return {"mean_difference": float("nan"), "lower": float("nan"), "upper": float("nan")}

    return {
        "mean_difference": float(np.mean(differences)),
        "lower": float(np.percentile(differences, 2.5)),
        "upper": float(np.percentile(differences, 97.5)),
    }


def pairwise_model_significance(
    y_true,
    probability_map: dict[str, np.ndarray],
    metric: str = "roc_auc",
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Generate pairwise bootstrap comparisons for all supplied models."""
    rows = []
    for model_a, model_b in combinations(probability_map.keys(), 2):
        comparison = bootstrap_metric_difference(
            y_true,
            probability_map[model_a],
            probability_map[model_b],
            metric=metric,
            n_bootstrap=n_bootstrap,
        )
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                **comparison,
            }
        )
    return pd.DataFrame(rows)
