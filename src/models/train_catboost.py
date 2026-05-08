"""Legacy CatBoost entry point kept for clarity after the model-stack refactor."""

from __future__ import annotations


def main():
    raise RuntimeError(
        "CatBoost is not part of the active project model stack. "
        "Use Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, or KNN instead."
    )


if __name__ == "__main__":
    main()
