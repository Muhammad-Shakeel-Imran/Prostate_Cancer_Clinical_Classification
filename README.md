# Prostate Cancer Clinical Classification - Ensemble Model

A machine learning ensemble-based system for clinical classification and risk assessment of prostate cancer using multiple base learners and ensemble methods.

## Project Overview

This project demonstrates:
- **Multiple Base Learners**: Logistic Regression, Random Forest, XGBoost, SVM, KNN, and optional LightGBM
- **Ensemble Methods**: Performance-weighted soft voting and stacking
- **Clinical Classification**: Binary classification for prostate cancer risk (low-risk vs high-risk)
- **Clinical Tabular Pipeline**: Excel ingestion, validation, preprocessing, and interpretable feature engineering
- **Comprehensive Evaluation**: ROC-AUC, Balanced Accuracy, Precision, Recall, Specificity, F1, and calibration-aware scoring

## Project Structure

```
├── configs/                          # Configuration files
│   ├── config.yaml                  # General project settings
│   ├── model_config.yaml            # Model hyperparameters
│   ├── training_config.yaml         # Training settings
│   └── paths.yaml                   # Data and output paths
│
├── data/                            # Data directory
│   ├── raw/                         # Raw input data
│   ├── processed/                   # Processed data
│   └── interim/                     # Intermediate data
│
├── src/                             # Source code
│   ├── data/
│   │   ├── data_loader.py          # Data loading and preprocessing
│   │   └── __init__.py
│   │
│   ├── models/
│   │   ├── base_model_trainer.py   # Training for individual models
│   │   └── __init__.py
│   │
│   ├── ensemble/
│   │   ├── voting_ensemble.py      # Voting ensemble implementation
│   │   ├── stacking_ensemble.py    # Stacking ensemble implementation
│   │   └── __init__.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # Evaluation metrics and visualization
│   │   └── __init__.py
│   │
│   ├── pipelines/
│   │   ├── training_pipeline.py    # Full training pipeline
│   │   ├── inference_pipeline.py   # Inference on new data
│   │   └── __init__.py
│   │
│   └── utils/
│       ├── logger.py               # Logging utilities
│       └── __init__.py
│
├── models/                          # Trained model storage
│   ├── trained_models/             # Base model files
│   └── ensemble_models/            # Ensemble meta-learners
│
├── reports/                         # Reports and visualizations
│   ├── figures/                    # Generated plots
│   ├── tables/                     # Results tables
│   └── model_results.csv           # Results
│
├── notebooks/                       # Jupyter notebooks
├── tests/                          # Test files
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
└── README.md                       # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd prostate-cancer-ensemble-platform
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the Full Training Pipeline

Execute the complete training workflow (data loading → model training → ensemble building → evaluation):

```bash
python src/pipelines/training_pipeline.py
```

This will:
- Load the prostate cancer Excel dataset from `data/raw/Prostate_Cancer.xlsx`
- Validate schema and engineer clinically meaningful tabular features
- Train diverse base learners such as Logistic Regression, Random Forest, XGBoost, SVM, and KNN
- Build a performance-weighted soft voting ensemble
- Build a stacking ensemble with a Logistic Regression meta-learner
- Evaluate all models
- Save trained models, ensemble artifacts, feature snapshots, and comparison reports

### 2. Make Predictions on New Data

Use the trained models for inference:

```python
from src.pipelines.inference_pipeline import InferencePipeline
import pandas as pd

# Initialize pipeline
pipeline = InferencePipeline('models')
pipeline.load_models()

# Load your new data with the same clinical columns as the training file
X_new = pd.read_excel('data/new_patients.xlsx')

# Make predictions using voting ensemble
voting_pred, voting_proba = pipeline.predict(X_new, method='voting')

# Make predictions using stacking ensemble
stacking_pred, stacking_proba = pipeline.predict(X_new, method='stacking')

print(f"Voting predictions: {voting_pred}")
print(f"Stacking predictions: {stacking_pred}")
```

### 3. Custom Data

To use your own clinical data:

1. Place your Excel or CSV file in `data/raw/`
2. Update `configs/paths.yaml` if needed
3. Run the training pipeline

Expected training columns:
```
id,age,BMI,PSA_level,free_PSA_ratio,testosterone_level,alkaline_phosphatase,family_history,smoking_status,radius,texture,perimeter,area,smoothness,compactness,symmetry,fractal_dimension,diagnosis_result
```

## Configuration

### Model Hyperparameters (`configs/model_config.yaml`)

Customize model parameters:
```yaml
models:
  logistic_regression:
    enabled: true

  xgboost:
    enabled: true
    n_estimators: 300
    learning_rate: 0.05

ensemble:
  voting:
    weight_metric: "roc_auc"
    weight_power: 4.0
```

### Ensemble Configuration

**Voting Ensemble**:
- Uses soft voting on predicted probabilities
- Learns weights from cross-validated base-model ROC-AUC

**Stacking Ensemble**:
- Uses 5-fold cross-validation to build robust meta-features
- Logistic Regression meta-learner with optional passthrough of original features

## Ensemble Methods

### Voting Ensemble
Weighted average of predicted probabilities from all base models:
```
P(class=1) = w₁P₁ + w₂P₂ + ... + wₙPₙ
```

### Stacking Ensemble
Two-level approach:
1. Base models generate predictions (meta-features)
2. Meta-learner (Logistic Regression) learns optimal combination

## Results

Models are evaluated using:
- **Accuracy**: Overall correctness
- **Balanced Accuracy**: Robustness under class imbalance
- **Precision**: Positive prediction accuracy
- **Recall**: True positive rate
- **Specificity**: True negative rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve

Results are saved to `reports/model_results.csv` and visualized in `reports/figures/`

## Model Files

- **Base Models**: `models/trained_models/{model_name}_model.pkl`
- **Soft Voting Ensemble**: `models/ensemble_models/soft_voting_ensemble.pkl`
- **Stacking Ensemble**: `models/ensemble_models/stacking_ensemble.pkl`

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Adding New Models

To add a new base learner:
1. Add training method to `src/models/base_model_trainer.py`
2. Call it from `train_all_models()`
3. Update config with hyperparameters

## Key Features

✅ Multiple diverse base learners for tabular medical data
✅ Two ensemble methods (performance-weighted Voting + Stacking)
✅ Comprehensive evaluation metrics
✅ Model persistence and loading
✅ Configuration-driven setup
✅ Real Excel-based clinical pipeline
✅ Reproducible results (seed management)
✅ Modular and extensible architecture

## Performance Notes

- Training time: ~5-10 minutes (depends on data size)
- Designed around the included Kaggle-style prostate cancer Excel dataset
- Use external validation before any real clinical deployment
- Stacking ensemble typically performs better than voting

## Clinical Relevance

This model framework is designed for:
- **Risk Stratification**: Identifying high-risk prostate cancer patients
- **Treatment Planning**: Supporting clinical decision-making
- **Research**: Ensemble method comparison in medical ML

**⚠️ Medical Disclaimer**: This is a demonstration model. Any clinical application requires validation, regulatory approval, and integration with clinical workflows.

## References

- XGBoost: Chen & Guestrin (2016)
- LightGBM: Ke et al. (2017)
- Ensemble Methods: Zhou (2012)

## License

See LICENSE file for details.

## Contact

For questions or contributions, please contact the development team.

## Future Enhancements

- [ ] Deep learning model integration
- [ ] Feature importance analysis (SHAP/LIME)
- [ ] Hyperparameter optimization
- [ ] Cross-validation strategies
- [ ] Real clinical data integration
- [ ] Model explainability
- [ ] Production API deployment

---

**Last Updated**: May 2024
**Version**: 1.0
