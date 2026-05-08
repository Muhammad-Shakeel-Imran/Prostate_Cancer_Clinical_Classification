# Architecture

## Design Goal

The repository is structured like a small production-style machine learning system so that data preparation, model training, evaluation, explainability, and inference are separated into reusable modules.

## Module Layout

- `src/data/`: dataset loading, schema validation, preprocessing, and feature engineering
- `src/models/`: base learner construction and training utilities
- `src/ensemble/`: soft voting and stacking implementations
- `src/evaluation/`: metrics, cross-validation summaries, ROC analysis, and significance testing
- `src/explainability/`: feature importance, SHAP helpers, local explanations, uncertainty scoring, and clinical interpretation
- `src/features/`: statistical summaries and feature ranking utilities
- `src/pipelines/`: end-to-end training, extended analysis, and inference entry points
- `src/utils/`: general-purpose helpers for logging, plotting, persistence, and reproducibility

## End-to-End Flow

1. Raw data is loaded from `data/raw/Prostate_Cancer.xlsx`.
2. Validation checks the required schema and target format.
3. Feature engineering creates clinically motivated derived variables.
4. Preprocessing pipelines prepare inputs consistently for each estimator.
5. Base models are cross-validated, then fitted on the training split.
6. Soft voting and stacking ensembles are trained from the fitted base learners.
7. Evaluation artifacts, predictions, plots, and summary tables are saved under `results/`.
8. Extended analysis produces feature ranking, significance, uncertainty, and interpretation outputs.

## Why This Structure Helps

This layout improves:
- reproducibility, because the same pipeline is reused across training and inference
- maintainability, because model logic and analysis logic are isolated
- scalability, because new models, explainers, or deployment layers can be added without rewriting the full workflow

## Extension Points

The current architecture is well suited for future additions such as:
- Optuna-based Bayesian tuning modules under `src/models/`
- imbalance-aware resampling inside cross-validation workflows
- a Streamlit application layered on top of `src/pipelines/inference_pipeline.py`
