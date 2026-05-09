"""Extended pipeline that trains models and emits richer analysis artifacts."""

from __future__ import annotations

from pathlib import Path

from src.data.data_loader import load_and_prepare_training_data, load_config, split_data
from src.evaluation.roc_analysis import plot_multi_model_roc_curves
from src.evaluation.statistical_significance import pairwise_model_significance
from src.explainability.clinical_interpretation import interpret_prediction
from src.explainability.feature_importance import (
    compute_permutation_importance_summary,
    extract_native_feature_importance,
)
from src.explainability.shap_analysis import compute_shap_summary
from src.explainability.uncertainty_estimation import build_uncertainty_frame
from src.features.feature_selection import rank_features_by_mutual_info
from src.features.statistical_features import standardized_mean_differences, summarize_numeric_features
from src.pipelines.training_pipeline import main as training_main
from src.utils.helpers import save_dataframe, save_json


def main():
    """Run training and then generate analysis tables and figures."""
    artifacts = training_main()
    config = load_config("configs/config.yaml")
    paths_config = load_config("configs/paths.yaml")

    dataset_bundle = artifacts.get("dataset_bundle") or load_and_prepare_training_data(
        paths_config["data"]["raw_data"]
    )
    X_test = artifacts.get("X_test")
    y_test = artifacts.get("y_test")
    if X_test is None or y_test is None:
        _, X_test, _, y_test = split_data(
            dataset_bundle.feature_frame,
            dataset_bundle.target,
            test_size=config["test_size"],
            random_state=config["random_seed"],
        )

    tables_dir = Path(paths_config["reports"]["tables"])
    figures_dir = Path(paths_config["reports"]["figures"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_dataframe(
        rank_features_by_mutual_info(dataset_bundle.feature_frame, dataset_bundle.target),
        tables_dir / "mutual_information_feature_ranking.csv",
    )
    save_dataframe(
        summarize_numeric_features(dataset_bundle.feature_frame),
        tables_dir / "numeric_feature_summary.csv",
        index=True,
    )
    save_dataframe(
        standardized_mean_differences(dataset_bundle.feature_frame, dataset_bundle.target),
        tables_dir / "standardized_mean_differences.csv",
    )

    probability_map = {}
    for model_name, model in artifacts["base_models"].items():
        probability_map[model_name] = model.predict_proba(X_test)[:, 1]
        try:
            save_dataframe(
                extract_native_feature_importance(model, X_test),
                tables_dir / f"{model_name}_native_feature_importance.csv",
            )
        except Exception:
            pass
        try:
            save_dataframe(
                compute_permutation_importance_summary(model, X_test, y_test),
                tables_dir / f"{model_name}_permutation_importance.csv",
            )
        except Exception:
            pass
        try:
            save_dataframe(
                compute_shap_summary(model, X_test),
                tables_dir / f"{model_name}_shap_summary.csv",
            )
        except Exception:
            pass

    probability_map["soft_voting"] = artifacts["voting_ensemble"].predict_proba(X_test)[:, 1]
    probability_map["stacking"] = artifacts["stacking_ensemble"].predict_proba(X_test)[:, 1]

    save_dataframe(
        plot_multi_model_roc_curves(
            y_test,
            probability_map,
            save_path=figures_dir / "roc_curves.png",
        ),
        tables_dir / "roc_auc_summary.csv",
    )
    save_dataframe(
        pairwise_model_significance(y_test, probability_map),
        tables_dir / "pairwise_model_significance.csv",
    )

    uncertainty_frame = build_uncertainty_frame(probability_map)
    save_dataframe(uncertainty_frame, tables_dir / "prediction_uncertainty.csv")

    highest_risk_index = uncertainty_frame["mean_probability"].idxmax()
    patient_narrative = interpret_prediction(
        dataset_bundle.raw_frame.loc[X_test.index[int(highest_risk_index)]],
        float(uncertainty_frame.loc[highest_risk_index, "mean_probability"]),
    )
    save_json(patient_narrative, tables_dir / "example_clinical_interpretation.json")

    return artifacts


if __name__ == "__main__":
    main()
