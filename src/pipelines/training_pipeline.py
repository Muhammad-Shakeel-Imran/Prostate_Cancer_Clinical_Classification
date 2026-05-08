"""Full training pipeline for the prostate cancer ensemble project."""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\..*",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_and_prepare_training_data, load_config, split_data
from src.data.feature_engineering import RAW_CATEGORICAL_COLUMNS, RAW_NUMERIC_COLUMNS
from src.ensemble.stacking_ensemble import StackingEnsemble
from src.ensemble.voting_ensemble import VotingEnsemble
from src.evaluation.cross_validation import plot_cv_results, save_cv_results
from src.evaluation.metrics import ModelEvaluator
from src.models.base_model_trainer import BaseModelTrainer


QUIET_LOGGERS = [
    "src.data.data_loader",
    "src.models.base_model_trainer",
    "src.ensemble.voting_ensemble",
    "src.ensemble.stacking_ensemble",
    "src.evaluation.metrics",
    "matplotlib",
]


def _ensure_output_directories(paths_config):
    directories = [
        paths_config["models"]["base_models"],
        paths_config["models"]["ensemble_models"],
        paths_config["models"]["checkpoints"],
        paths_config["reports"]["figures"],
        paths_config["reports"]["tables"],
        paths_config["reports"]["predictions"],
        Path(paths_config["data"]["processed_data"]).parent,
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def _configure_terminal_logging() -> None:
    """Keep the terminal focused on the top-level pipeline messages."""
    for logger_name in QUIET_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _save_processed_snapshot(feature_frame, target, output_path: str | Path) -> None:
    snapshot = feature_frame.copy()
    snapshot["target"] = target.values
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(output_path, index=False)


def _save_training_summary(summary, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)


def _log_section(title: str) -> None:
    """Print a clean, readable section header."""
    border = "=" * len(title)
    logger.info("\n%s\n%s\n%s", border, title, border)


def _format_metric_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render a compact metric table for terminal output."""
    visible_columns = [column for column in columns if column in frame.columns]
    display_frame = frame[visible_columns].copy()
    return display_frame.round(4).to_string()


def _save_evaluation_outputs(
    evaluator: ModelEvaluator,
    X_test,
    y_test,
    prediction_map,
    probability_map,
    paths_config,
) -> None:
    """Save per-model predictions and evaluation figures under the results directory."""
    figures_dir = Path(paths_config["reports"]["figures"])
    tables_dir = Path(paths_config["reports"]["tables"])
    predictions_dir = Path(paths_config["reports"]["predictions"])

    prediction_frame = pd.DataFrame({"y_true": y_test.values if hasattr(y_test, "values") else y_test})
    for model_name, y_pred in prediction_map.items():
        prediction_frame[f"{model_name}_pred"] = y_pred
        prediction_frame[f"{model_name}_proba"] = probability_map[model_name]
        evaluator.plot_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            model_name=model_name,
            save_path=figures_dir / f"{model_name}_confusion_matrix.png",
        )

    evaluator.save_prediction_frame(
        prediction_frame,
        predictions_dir / "test_set_predictions.csv",
    )
    evaluator.plot_roc_curves(
        y_true=y_test,
        probability_map=probability_map,
        save_path=figures_dir / "roc_curves.png",
    )
    evaluator.plot_precision_recall_curves(
        y_true=y_test,
        probability_map=probability_map,
        save_path=figures_dir / "precision_recall_curves.png",
    )
    evaluator.plot_calibration_curves(
        y_true=y_test,
        probability_map=probability_map,
        save_path=figures_dir / "calibration_curves.png",
    )

    pd.DataFrame(
        {
            "sample_index": list(X_test.index),
            "y_true": y_test.values if hasattr(y_test, "values") else y_test,
        }
    ).to_csv(tables_dir / "test_index_reference.csv", index=False)


def main():
    """Execute the end-to-end training workflow."""
    _configure_terminal_logging()
    _log_section("PROSTATE CANCER CLINICAL CLASSIFICATION - ENSEMBLE PIPELINE")

    config = load_config("configs/config.yaml")
    model_config = load_config("configs/model_config.yaml")
    paths_config = load_config("configs/paths.yaml")
    _ensure_output_directories(paths_config)

    logger.info("Loading and preparing dataset...")
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
    logger.info(
        "Dataset ready: %s rows, %s engineered features, train=%s, test=%s",
        dataset_bundle.summary["n_rows"],
        dataset_bundle.feature_frame.shape[1],
        X_train.shape[0],
        X_test.shape[0],
    )

    trainer = BaseModelTrainer(model_config=model_config, random_state=config["random_seed"])

    _log_section("Cross-Validation")
    logger.info("Benchmarking base learners with %s-fold CV...", model_config["training"]["cv_folds"])
    cv_results = trainer.cross_validate_models(
        X_train,
        y_train,
        cv_folds=model_config["training"]["cv_folds"],
    )
    save_cv_results(cv_results, Path(paths_config["reports"]["tables"]) / "cross_validation_results.csv")
    plot_cv_results(
        cv_results,
        metric="roc_auc_mean",
        save_path=Path(paths_config["reports"]["figures"]) / "cross_validation_ranking.png",
    )
    logger.info(
        "\n%s",
        _format_metric_table(
            cv_results,
            ["roc_auc_mean", "f1_mean", "accuracy_mean", "precision_mean", "recall_mean"],
        ),
    )

    _log_section("Final Model Training")
    logger.info("Fitting enabled base models on the full training split...")
    base_models = trainer.train_all_models(X_train, y_train)
    trainer.save_models(paths_config["models"]["base_models"])

    evaluator = ModelEvaluator()
    decision_threshold = model_config["training"].get("decision_threshold", 0.5)
    test_results = {}
    prediction_map = {}
    probability_map = {}

    _log_section("Held-Out Test Evaluation")
    logger.info("Scoring base learners and ensembles on the test split...")
    for model_name, model in base_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= decision_threshold).astype(int)
        metrics = evaluator.evaluate_model(y_test, y_pred, y_proba)
        test_results[model_name] = metrics
        prediction_map[model_name] = y_pred
        probability_map[model_name] = y_proba

    logger.info("Building soft voting ensemble...")
    voting_weights = VotingEnsemble.derive_weights(
        cv_results=cv_results,
        metric=model_config["ensemble"]["voting"]["weight_metric"],
        power=model_config["ensemble"]["voting"]["weight_power"],
    )
    voting_ensemble = VotingEnsemble(base_models, weights=voting_weights or None)
    voting_metrics = voting_ensemble.evaluate(X_test, y_test, threshold=decision_threshold)
    test_results["soft_voting"] = voting_metrics
    voting_proba = voting_ensemble.predict_proba(X_test)[:, 1]
    voting_pred = (voting_proba >= decision_threshold).astype(int)
    prediction_map["soft_voting"] = voting_pred
    probability_map["soft_voting"] = voting_proba
    voting_ensemble.save(paths_config["artifacts"]["voting_model"])

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
    stacking_proba = stacking_ensemble.predict_proba(X_test)[:, 1]
    stacking_pred = (stacking_proba >= decision_threshold).astype(int)
    prediction_map["stacking"] = stacking_pred
    probability_map["stacking"] = stacking_proba
    stacking_ensemble.save(paths_config["artifacts"]["stacking_model"])

    comparison_df_for_terminal = (
        pd.DataFrame.from_dict(test_results, orient="index")
        .sort_values(by="roc_auc", ascending=False)
    )
    logger.info(
        "\n%s",
        _format_metric_table(
            comparison_df_for_terminal,
            ["roc_auc", "f1", "accuracy", "balanced_accuracy", "precision", "recall", "specificity"],
        ),
    )

    _log_section("Saving Results")
    comparison_df = evaluator.compare_models(
        test_results,
        save_path=Path(paths_config["reports"]["figures"]) / "model_comparison.png",
    )
    export_df = comparison_df.join(cv_results, how="left")
    results_path = Path(paths_config["reports"]["results"]) / "model_results.csv"
    export_df.to_csv(results_path)
    _save_evaluation_outputs(
        evaluator=evaluator,
        X_test=X_test,
        y_test=y_test,
        prediction_map=prediction_map,
        probability_map=probability_map,
        paths_config=paths_config,
    )

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

    logger.info("Saved metrics table: %s", results_path)
    logger.info("Saved figures: %s", paths_config["reports"]["figures"])
    logger.info("Saved tables: %s", paths_config["reports"]["tables"])
    logger.info("Saved predictions: %s", paths_config["reports"]["predictions"])
    _log_section("TRAINING PIPELINE COMPLETED SUCCESSFULLY")

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
