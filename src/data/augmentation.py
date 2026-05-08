"""Optional tabular augmentation helpers for experimentation."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def augment_minority_class(
    X: pd.DataFrame,
    y: pd.Series,
    noise_scale: float = 0.02,
    copies: int = 1,
    random_state: int = 42,
    numeric_columns: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create light Gaussian-noise variants of the minority-class rows.

    This is deliberately conservative and meant for experiments only. The default
    training pipeline does not apply augmentation automatically.
    """
    rng = np.random.default_rng(random_state)
    X_augmented = X.copy()
    y_augmented = y.copy()

    numeric_columns = list(numeric_columns or X.select_dtypes(include=[np.number]).columns)
    if not numeric_columns:
        return X_augmented, y_augmented

    minority_label = y.value_counts().idxmin()
    minority_rows = X.loc[y == minority_label]
    if minority_rows.empty:
        return X_augmented, y_augmented

    column_scales = minority_rows[numeric_columns].std().replace(0, 1.0) * noise_scale

    generated_frames = []
    for _ in range(max(copies, 0)):
        sampled = minority_rows.sample(
            n=len(minority_rows),
            replace=True,
            random_state=int(rng.integers(0, 1_000_000)),
        ).copy()
        noise = rng.normal(
            loc=0.0,
            scale=column_scales.to_numpy(),
            size=(len(sampled), len(numeric_columns)),
        )
        sampled.loc[:, numeric_columns] = sampled[numeric_columns].to_numpy() + noise
        generated_frames.append(sampled)

    if not generated_frames:
        return X_augmented, y_augmented

    synthetic_frame = pd.concat(generated_frames, ignore_index=True)
    synthetic_target = pd.Series([minority_label] * len(synthetic_frame), name=y.name)
    X_augmented = pd.concat([X_augmented, synthetic_frame], ignore_index=True)
    y_augmented = pd.concat([y_augmented.reset_index(drop=True), synthetic_target], ignore_index=True)
    return X_augmented, y_augmented
