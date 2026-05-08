# Quick Start Guide

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Run Training Pipeline
```bash
python main.py
```

## 3. Expected Output
- Trained base models in `models/trained_models/`
- Soft voting and stacking artifacts in `models/ensemble_models/`
- Results saved to `reports/model_results.csv`
- Performance plots in `reports/figures/`
- Training summary in `reports/training_summary.json`

## 4. Use Trained Models

```python
from src.pipelines.inference_pipeline import InferencePipeline
import pandas as pd

pipeline = InferencePipeline('models')
pipeline.load_models()

X_new = pd.read_excel('data/your_data.xlsx')
predictions, probabilities = pipeline.predict(X_new, method='stacking')
```

## Key Ensemble Methods Implemented

### Voting Ensemble
- Soft voting with configurable weights
- Averages predictions from all base models
- Fast and simple

### Stacking Ensemble
- Two-level approach with meta-learner
- Uses cross-validation for robust meta-feature generation
- Generally better performance

## Project Features

✅ Diverse tabular base learners including Logistic Regression, RF, XGBoost, SVM, and KNN
✅ Weighted Voting & Stacking Ensembles
✅ Comprehensive Evaluation Metrics
✅ Modular & Extensible Code
✅ Configuration-Driven Setup
✅ Real Excel-Based Clinical Classification Focus

See README.md for complete documentation.
