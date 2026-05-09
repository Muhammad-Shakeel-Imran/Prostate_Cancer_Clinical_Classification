# Methodology

## Overview

This project implements a research-oriented prostate cancer classification workflow using structured biomedical data. The goal is to evaluate how well diverse machine learning models and ensemble strategies can predict malignant versus benign outcomes while preserving interpretability and reproducibility.

## Dataset Framing

The repository uses a Kaggle-sourced prostate cancer dataset stored in `data/raw/Prostate_Cancer.xlsx`. The data includes:
- clinical variables such as age, BMI, family history, and smoking status
- biomarker measurements including PSA-related signals, testosterone, and alkaline phosphatase
- morphology-inspired descriptors such as radius, texture, perimeter, area, compactness, symmetry, and fractal dimension

The raw dataset contains 18 original columns. After feature engineering, the training matrix expands to 27 model features.

## Preprocessing and Feature Engineering

The preprocessing pipeline standardizes numeric inputs, encodes categorical variables, and preserves a consistent feature transformation path for training and inference. On top of the original variables, the project derives biomedical risk features such as:
- `log_psa`
- `psa_age_ratio`
- `psa_bmi_interaction`
- `psa_to_testosterone_ratio`
- `free_to_total_psa_index`
- `alkaline_phosphatase_to_psa_ratio`
- `shape_risk_index`

These engineered features are intentionally lightweight and clinically interpretable rather than overly abstract.

## Models

The benchmark includes several complementary learning strategies:
- Logistic Regression for interpretability and a strong linear baseline
- Random Forest for nonlinear pattern learning and variance reduction
- XGBoost for complex feature interactions
- SVM with an RBF kernel for an alternative margin-based decision boundary
- KNN for local neighborhood-based diversity
- optional LightGBM support through configuration

## Ensemble Strategy

Two ensemble methods are implemented:
- Soft voting, where model probabilities are combined using performance-derived weights
- Stacked generalization, where out-of-fold predictions from base learners become meta-features for a Logistic Regression meta-model

This design allows the project to compare whether ensemble integration improves robustness beyond the best standalone learner.

## Evaluation Workflow

The current pipeline uses:
- stratified cross-validation for model benchmarking
- a held-out test split for final reporting
- ROC-AUC, F1, accuracy, balanced accuracy, precision, recall, and specificity
- confusion matrices, ROC curves, precision-recall curves, and calibration plots

Several base learners also use `class_weight="balanced"` to improve robustness under moderate imbalance. The decision threshold is configurable in `configs/model_config.yaml`.

## Explainability and Error Analysis

The repository extends beyond raw prediction accuracy by providing:
- native and permutation feature importance tables
- optional SHAP summaries with safe fallbacks when SHAP is not installed
- uncertainty signals based on predictive entropy, confidence margin, and ensemble disagreement
- clinical-style interpretation narratives for high-risk predictions

This makes the project more suitable for biomedical AI discussion than a simple benchmark notebook.

## Current Scope and Next Extensions

The implemented codebase already supports the core ensemble training, evaluation, interpretability, and reporting workflow. The architecture is also ready for future additions commonly expected in stronger medical ML systems, such as:
- Optuna-based Bayesian hyperparameter optimization
- SMOTE-based imbalance handling within training folds
- a Streamlit front end for interactive clinical risk scoring

Those extensions are best described as next-step enhancements unless and until they are added directly to the default pipeline.
