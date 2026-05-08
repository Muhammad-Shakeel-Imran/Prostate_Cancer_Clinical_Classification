"""Train the KNN base learner used to diversify the ensemble."""

from __future__ import annotations

from src.models.training_helpers import run_training_entrypoint


def main():
    return run_training_entrypoint("knn")


if __name__ == "__main__":
    main()
