"""Evaluation helpers for model comparison and reporting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
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
    def plot_confusion_matrix(
        y_true,
        y_pred,
        model_name: str,
        save_path: str | Path,
    ) -> None:
        """Save a confusion matrix heatmap for one model."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(cm, cmap="Blues")
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1], labels=["Benign", "Malignant"])
        ax.set_yticks([0, 1], labels=["Benign", "Malignant"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix: {model_name}")

        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, f"{cm[row, col]}", ha="center", va="center", color="black")

        fig.tight_layout()
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_roc_curves(
        y_true,
        probability_map: Dict[str, np.ndarray],
        save_path: str | Path,
    ) -> None:
        """Save a multi-model ROC curve figure."""
        fig, ax = plt.subplots(figsize=(8, 6))
        for model_name, y_proba in probability_map.items():
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_value = roc_auc_score(y_true, y_proba)
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.3f})")

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend(loc="lower right")
        fig.tight_layout()
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_precision_recall_curves(
        y_true,
        probability_map: Dict[str, np.ndarray],
        save_path: str | Path,
    ) -> None:
        """Save a multi-model precision-recall figure."""
        fig, ax = plt.subplots(figsize=(8, 6))
        for model_name, y_proba in probability_map.items():
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            ap_value = average_precision_score(y_true, y_proba)
            ax.plot(recall, precision, label=f"{model_name} (AP={ap_value:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend(loc="lower left")
        fig.tight_layout()
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_calibration_curves(
        y_true,
        probability_map: Dict[str, np.ndarray],
        save_path: str | Path,
        n_bins: int = 10,
    ) -> None:
        """Save a multi-model calibration figure."""
        fig, ax = plt.subplots(figsize=(8, 6))
        for model_name, y_proba in probability_map.items():
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true,
                y_proba,
                n_bins=n_bins,
                strategy="quantile",
            )
            ax.plot(
                mean_predicted_value,
                fraction_of_positives,
                marker="o",
                label=model_name,
            )

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Positive Rate")
        ax.set_title("Calibration Curves")
        ax.legend(loc="upper left")
        fig.tight_layout()
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def save_prediction_frame(
        prediction_frame: pd.DataFrame,
        save_path: str | Path,
    ) -> None:
        """Persist model predictions for later inspection."""
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_csv(output_path, index=False)

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
