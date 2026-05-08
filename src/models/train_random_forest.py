"""Train the Random Forest base learner."""

from __future__ import annotations

from src.models.training_helpers import run_training_entrypoint


def main():
    return run_training_entrypoint("random_forest")


if __name__ == "__main__":
    main()
