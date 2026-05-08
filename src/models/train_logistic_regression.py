"""Train Logistic Regression as the interpretable linear baseline."""

from __future__ import annotations

from src.models.training_helpers import run_training_entrypoint


def main():
    return run_training_entrypoint("logistic_regression")


if __name__ == "__main__":
    main()
