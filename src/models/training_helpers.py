"""Shared helpers for explicit single-model training entry points."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from src.data.data_loader import load_and_prepare_training_data, load_config, split_data
from src.evaluation.metrics import ModelEvaluator
from src.models.base_model_trainer import BaseModelTrainer


def _build_single_model_config(model_config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Clone the model configuration and enable only one base learner."""
    single_model_config = deepcopy(model_config)
    if model_name not in single_model_config["models"]:
        raise ValueError(f"Unknown model '{model_name}'.")

    for candidate_name, params in single_model_config["models"].items():
        params["enabled"] = candidate_name == model_name
    return single_model_config


def train_single_model(
    model_name: str,
    config_path: str | Path = "configs/config.yaml",
    model_config_path: str | Path = "configs/model_config.yaml",
    paths_config_path: str | Path = "configs/paths.yaml",
) -> Dict[str, Any]:
    """Train, evaluate, and persist a single base learner."""
    config = load_config(config_path)
    model_config = load_config(model_config_path)
    paths_config = load_config(paths_config_path)

    dataset_bundle = load_and_prepare_training_data(paths_config["data"]["raw_data"])
    X_train, X_test, y_train, y_test = split_data(
        dataset_bundle.feature_frame,
        dataset_bundle.target,
        test_size=config["test_size"],
        random_state=config["random_seed"],
    )

    single_model_config = _build_single_model_config(model_config, model_name)
    trainer = BaseModelTrainer(single_model_config, random_state=config["random_seed"])

    cv_results = trainer.cross_validate_models(
        X_train,
        y_train,
        cv_folds=model_config["training"]["cv_folds"],
    )
    trained_models = trainer.train_all_models(X_train, y_train)
    trainer.save_models(paths_config["models"]["base_models"])

    model = trained_models[model_name]
    threshold = model_config["training"].get("decision_threshold", 0.5)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    metrics = ModelEvaluator.evaluate_model(y_test, y_pred, y_proba)

    return {
        "model_name": model_name,
        "model": model,
        "metrics": metrics,
        "cv_results": cv_results,
        "X_test": X_test,
        "y_test": y_test,
    }


def print_single_model_report(report: Dict[str, Any]) -> None:
    """Print a compact report for a single trained model."""
    ModelEvaluator.print_metrics(report["metrics"], report["model_name"])
    cv_results = report.get("cv_results")
    if cv_results is not None and not cv_results.empty:
        print("\nCross-Validation Summary")
        print("-" * 50)
        print(cv_results.round(4).to_string())


def run_training_entrypoint(model_name: str) -> Dict[str, Any]:
    """CLI-friendly wrapper for the single-model training scripts."""
    report = train_single_model(model_name=model_name)
    print_single_model_report(report)
    return report
