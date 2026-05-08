"""Full training pipeline for the prostate cancer ensemble project."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_and_prepare_training_data, load_config, split_data
from src.data.feature_engineering import RAW_CATEGORICAL_COLUMNS, RAW_NUMERIC_COLUMNS
from src.ensemble.stacking_ensemble import StackingEnsemble
from src.ensemble.voting_ensemble import VotingEnsemble
from src.evaluation.metrics import ModelEvaluator
from src.models.base_model_trainer import BaseModelTrainer


def _ensure_output_directories(paths_config):
    directories = [
        paths_config["models"]["base_models"],
        paths_config["models"]["ensemble_models"],
        paths_config["models"]["checkpoints"],
        paths_config["reports"]["figures"],
        paths_config["reports"]["tables"],
        Path(paths_config["data"]["processed_data"]).parent,
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def _save_processed_snapshot(feature_frame, target, output_path: str | Path) -> None:
    snapshot = feature_frame.copy()
    snapshot["target"] = target.values
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(output_path, index=False)


def _save_training_summary(summary, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)


def main():
    """Execute the end-to-end training workflow."""
    logger.info("=" * 80)
    logger.info("PROSTATE CANCER CLINICAL CLASSIFICATION - ENSEMBLE PIPELINE")
    logger.info("=" * 80)

    config = load_config("configs/config.yaml")
    model_config = load_config("configs/model_config.yaml")
    paths_config = load_config("configs/paths.yaml")
    _ensure_output_directories(paths_config)

    logger.info("Loading and preparing the prostate cancer dataset...")
    dataset_bundle = load_and_prepare_training_data(paths_config["data"]["raw_data"])
    _save_processed_snapshot(
        dataset_bundle.feature_frame,
        dataset_bundle.target,
        paths_config["data"]["processed_data"],
    )

    X_train, X_test, y_train, y_test = split_data(
        dataset_bundle.feature_frame,
        dataset_bundle.target,
        test_size=config["test_size"],
        random_state=config["random_seed"],
    )

    trainer = BaseModelTrainer(model_config=model_config, random_state=config["random_seed"])

    logger.info("Running cross-validation on the base model pool...")
    cv_results = trainer.cross_validate_models(
        X_train,
        y_train,
        cv_folds=model_config["training"]["cv_folds"],
    )

    logger.info("Training the final base models on the full training split...")
    base_models = trainer.train_all_models(X_train, y_train)
    trainer.save_models(paths_config["models"]["base_models"])

    evaluator = ModelEvaluator()
    decision_threshold = model_config["training"].get("decision_threshold", 0.5)
    test_results = {}

    logger.info("Evaluating individual base learners on the held-out test split...")
    for model_name, model in base_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= decision_threshold).astype(int)
        metrics = evaluator.evaluate_model(y_test, y_pred, y_proba)
        test_results[model_name] = metrics
        evaluator.print_metrics(metrics, model_name)

    logger.info("Building performance-weighted soft voting ensemble...")
    voting_weights = VotingEnsemble.derive_weights(
        cv_results=cv_results,
        metric=model_config["ensemble"]["voting"]["weight_metric"],
        power=model_config["ensemble"]["voting"]["weight_power"],
    )
    voting_ensemble = VotingEnsemble(base_models, weights=voting_weights or None)
    voting_metrics = voting_ensemble.evaluate(X_test, y_test, threshold=decision_threshold)
    test_results["soft_voting"] = voting_metrics
    voting_ensemble.save(paths_config["artifacts"]["voting_model"])
    evaluator.print_metrics(voting_metrics, "Soft Voting Ensemble")

    logger.info("Building stacking ensemble...")
    stacking_config = model_config["ensemble"]["stacking"]
    stacking_ensemble = StackingEnsemble(
        base_estimators=trainer.build_estimators(X_train),
        meta_learner_name=stacking_config.get("meta_learner", "logistic_regression"),
        meta_params=stacking_config.get("meta_params", {}),
        cv_folds=stacking_config.get("cv_folds", 5),
        passthrough=stacking_config.get("passthrough", True),
        random_state=config["random_seed"],
    )
    stacking_ensemble.fit(X_train, y_train)
    stacking_metrics = stacking_ensemble.evaluate(X_test, y_test, threshold=decision_threshold)
    test_results["stacking"] = stacking_metrics
    stacking_ensemble.save(paths_config["artifacts"]["stacking_model"])
    evaluator.print_metrics(stacking_metrics, "Stacking Ensemble")

    logger.info("Compiling final comparison report...")
    comparison_df = evaluator.compare_models(
        test_results,
        save_path=Path(paths_config["reports"]["figures"]) / "model_comparison.png",
    )
    export_df = comparison_df.join(cv_results, how="left")
    results_path = Path(paths_config["reports"]["results"]) / "model_results.csv"
    export_df.to_csv(results_path)
    logger.info("Saved model comparison table to %s", results_path)

    raw_feature_names = RAW_NUMERIC_COLUMNS + RAW_CATEGORICAL_COLUMNS
    engineered_features = [
        column
        for column in dataset_bundle.feature_frame.columns
        if column not in raw_feature_names
    ]
    summary = {
        "project_name": config["project_name"],
        "dataset_summary": dataset_bundle.summary,
        "feature_count": int(dataset_bundle.feature_frame.shape[1]),
        "raw_feature_count": len(raw_feature_names),
        "engineered_features": engineered_features,
        "base_models": list(base_models.keys()),
        "voting_weights": voting_ensemble.weights,
        "cv_results": cv_results.round(4).to_dict(orient="index"),
        "test_results": {
            model_name: {metric: round(float(value), 4) for metric, value in metrics.items()}
            for model_name, metrics in test_results.items()
        },
    }
    _save_training_summary(summary, paths_config["artifacts"]["training_summary"])

    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    return {
        "base_models": base_models,
        "voting_ensemble": voting_ensemble,
        "stacking_ensemble": stacking_ensemble,
        "results": export_df,
        "cv_results": cv_results,
        "X_test": X_test,
        "y_test": y_test,
        "dataset_bundle": dataset_bundle,
    }


if __name__ == "__main__":
    main()
