"""Cross-validation benchmarking helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def save_cv_results(results: pd.DataFrame, save_path: str | Path) -> None:
    """Persist the cross-validation table to disk."""
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path)


def plot_cv_results(
    results: pd.DataFrame,
    metric: str = "roc_auc_mean",
    save_path: str | Path | None = None,
):
    """Visualize model ranking by a chosen cross-validation metric."""
    if metric not in results.columns:
        raise ValueError(f"Metric '{metric}' not found in results.")

    ranking = results.sort_values(metric, ascending=False)
    ax = ranking[metric].plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Cross-Validated Model Ranking")
    plt.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
