"""Plotting helpers used by analysis notebooks and reporting pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_feature_importance_bar(
    importance_frame: pd.DataFrame,
    score_column: str,
    top_n: int = 15,
    save_path: str | Path | None = None,
) -> None:
    """Plot a horizontal bar chart for a feature-importance dataframe."""
    ranking = importance_frame.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(ranking["feature"], ranking[score_column])
    plt.xlabel(score_column.replace("_", " ").title())
    plt.title("Top Feature Importances")
    plt.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_probability_histogram(
    probabilities,
    title: str = "Predicted Malignancy Probabilities",
    save_path: str | Path | None = None,
) -> None:
    """Plot a histogram of predicted positive-class probabilities."""
    plt.figure(figsize=(8, 5))
    plt.hist(np.asarray(probabilities), bins=20, edgecolor="black", alpha=0.8)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_numeric_feature_distributions(
    feature_frame: pd.DataFrame,
    columns: Iterable[str],
    save_dir: str | Path,
) -> None:
    """Save histograms for selected numeric features."""
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for column in columns:
        if column not in feature_frame.columns:
            continue
        plt.figure(figsize=(7, 4))
        plt.hist(feature_frame[column].dropna(), bins=20, edgecolor="black", alpha=0.8)
        plt.title(f"Distribution: {column}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{column}_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()
