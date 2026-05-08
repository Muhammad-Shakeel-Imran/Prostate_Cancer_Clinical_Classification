# Deployment Guide

## Current Status

The repository already includes a reusable inference interface in `src/pipelines/inference_pipeline.py`, which means the core prediction logic is separated from the training workflow and can be wrapped by an application layer.

At the moment, the project is deployment-ready in architecture, but a full Streamlit UI is not yet included in the default source tree.

## Recommended Streamlit Wrapper

A practical deployment path is to build a Streamlit app that:
- collects patient-level clinical, biomarker, and morphology inputs
- loads the saved trained models from `models/`
- generates malignancy probabilities using the stacking or soft-voting ensemble
- shows class prediction, confidence, and risk band
- displays feature importance or SHAP-style explanation outputs when available
- surfaces uncertainty indicators for borderline cases

## Suggested User Workflow

1. User enters patient values such as age, BMI, PSA, free PSA ratio, testosterone, and morphology features.
2. The app converts inputs into the same schema expected by the preprocessing pipeline.
3. The inference pipeline returns predicted class labels and probabilities.
4. The UI presents risk level, confidence, and interpretation text in a clinician-friendly format.

## Deployment Checklist

- train and save the base and ensemble models locally
- confirm the expected input schema matches the app form
- install optional explainability packages if explanation panels are required
- validate prediction outputs on known examples before release
- document the clinical disclaimer clearly in the UI

## Important Note

Any deployed version of this project should be treated as a research support tool, not as a validated clinical diagnostic product.
