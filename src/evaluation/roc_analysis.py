"""ROC analysis helpers for base learners and ensembles."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def bootstrap_auc_confidence_interval(
    y_true,
    y_proba,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict[str, float]:
    """Estimate a simple bootstrap confidence interval for ROC-AUC."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    auc_scores = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        sample_y = y_true[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        auc_scores.append(roc_auc_score(sample_y, y_proba[indices]))

    if not auc_scores:
        return {"auc": float("nan"), "lower": float("nan"), "upper": float("nan")}

    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "lower": float(np.percentile(auc_scores, 2.5)),
        "upper": float(np.percentile(auc_scores, 97.5)),
    }


def plot_multi_model_roc_curves(
    y_true,
    probability_map: dict[str, np.ndarray],
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Plot ROC curves for several models and return a summary table."""
    plt.figure(figsize=(8, 6))
    rows = []

    for model_name, y_proba in probability_map.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_summary = bootstrap_auc_confidence_interval(y_true, y_proba)
        rows.append({"model": model_name, **auc_summary})
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_summary['auc']:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Prostate Cancer Classification")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return pd.DataFrame(rows).sort_values("auc", ascending=False)
