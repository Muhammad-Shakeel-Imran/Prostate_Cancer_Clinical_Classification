"""Evaluation helpers for model comparison and reporting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Compute evaluation metrics that are useful for clinical classifiers."""

    @staticmethod
    def evaluate_model(y_true, y_pred, y_proba=None) -> Dict[str, float]:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(specificity),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        if y_proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["average_precision"] = float(average_precision_score(y_true, y_proba))
            metrics["brier_score"] = float(brier_score_loss(y_true, y_proba))

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
        print(f"\n{model_name} Performance Metrics")
        print("-" * 50)
        for key, value in metrics.items():
            print(f"{key:20s}: {value:.4f}")

    @staticmethod
    def compare_models(
        results_dict: Dict[str, Dict[str, float]],
        save_path: str | Path | None = None,
        sort_by: str = "roc_auc",
    ) -> pd.DataFrame:
        """Build a comparison table and optionally export a score plot."""
        comparison = pd.DataFrame.from_dict(results_dict, orient="index")
        if sort_by in comparison.columns:
            comparison = comparison.sort_values(by=sort_by, ascending=False)

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            plot_frame = comparison[[col for col in ["roc_auc", "f1", "accuracy"] if col in comparison.columns]]
            ax = plot_frame.plot(kind="bar", figsize=(12, 6))
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score")
            ax.set_title("Model Comparison on Prostate Cancer Classification")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Saved comparison plot to %s", output_path)

        return comparison
