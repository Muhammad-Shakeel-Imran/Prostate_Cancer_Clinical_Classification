"""Train the SVM base learner with the configured RBF kernel."""

from __future__ import annotations

from src.models.training_helpers import run_training_entrypoint


def main():
    return run_training_entrypoint("svm")


if __name__ == "__main__":
    main()
