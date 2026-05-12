# Prostate Cancer Clinical Classification Model

This repository is a research-oriented and clinically focused machine learning framework for prostate cancer risk prediction using structured biomedical data. The pipeline combines clinical variables, biomarker measurements, and morphology-inspired features in a modular ensemble workflow that is designed to feel closer to a biomedical AI prototype than a one-off notebook experiment.

The project is especially useful for research presentations, interview discussions, and academic review because it emphasizes:
- clinically meaningful tabular data from kaggle preprocessing instead of toy data
- interpretable and non-linear models side by side
- ensemble learning through soft voting and stacked generalization
- reproducible evaluation with saved artifacts, figures, and summary reports
- explainability, uncertainty analysis, and clinical-style interpretation outputs

## Project Overview

This project built as a clinical decision-support style prototype for accurate and interpretable prostate cancer prediction. It uses a clincal tabular dataset to benchmark diverse learning strategies, compare single-model and ensemble performance, and surface clinically meaningful insights from model outputs.

Implemented strengths of the current codebase include:
- stratified cross-validation for stable model comparison
- class-weighted learners for more robust handling of class imbalance
- domain-inspired biomedical feature engineering
- soft voting and stacked generalization for predictive robustness
- SHAP-based explainability
- uncertainty analysis and clinical narrative generation for high-risk cases

## Clinical Phenotypes 

The task is binary clinical classification:
- `B`: benign / lower-risk pattern
- `M`: malignant / higher-risk pattern

Rather than treating this as a generic Kaggle exercise, the pipeline is structured as a clinical decision-support style benchmark:
- Logistic Regression serves as an interpretable linear baseline
- Random Forest captures non-linear interactions and reduces variance
- XGBoost models more complex feature interactions
- SVM provides an alternative margin-based decision boundary
- KNN adds diversity through local neighborhood structure
- Soft Voting and Stacking test whether combining models improves robustness or not 

## Dataset Context

The training data used in this project comes from a Kaggle prostate cancer dataset stored locally as `data/raw/Prostate_Cancer.xlsx`. The dataset contains patient-level biomedical information spanning:
- demographic and clinical variables such as age, BMI, family history, and smoking status
- biomarker measurements such as PSA, free PSA ratio, testosterone, and alkaline phosphatase
- morphology-inspired descriptors such as radius, texture, perimeter, area, smoothness, compactness, symmetry, and fractal dimension

The current dataset snapshot contains:
- `2000` patient records
- `18` original columns
- `27` final model features after feature engineering

Within this repository, the dataset is reframed as a structured biomedical benchmark for malignancy prediction rather than a simple leaderboard exercise.

## Modeling Approach

### Base Models
- Logistic Regression
- Random Forest
- XGBoost
- SVM with RBF kernel
- KNN

### Ensemble Models
- **Soft Voting**: weighted probability averaging using cross-validated model quality
- **Stacking**: out-of-fold base-model probabilities used as meta-features for a Logistic Regression meta-learner, with optional passthrough features

### Feature Engineering

In addition to the original clinical variables, the pipeline constructs derived features such as:
- `log_psa`
- `log_alkaline_phosphatase`
- `psa_age_ratio`
- `psa_bmi_interaction`
- `psa_to_testosterone_ratio`
- `free_to_total_psa_index`
- `alkaline_phosphatase_to_psa_ratio`
- `area_perimeter_ratio`
- `texture_radius_ratio`
- `compactness_symmetry_interaction`
- `shape_risk_index`

These are intentionally lightweight and interpretable so the model remains explainable in a clinical research setting.

### Analysis and Explainability

Beyond model training, the repository includes:
- stratified 5-fold cross-validation for comparative benchmarking
- held-out evaluation with ROC, precision-recall, calibration, and confusion-matrix outputs
- native and permutation feature importance summaries
- optional SHAP-based global explanations with safe fallbacks
- uncertainty estimation through predictive entropy, confidence margin, and ensemble disagreement
- clinical-style narratives highlighting intuitive risk flags such as elevated PSA or older age

## Latest Model Performance

The following held-out test results were produced by the current pipeline on the included dataset:

| Model | ROC-AUC | F1 | Accuracy | Balanced Accuracy | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7408 | 0.7218 | 0.7225 | 0.7226 | 0.7310 | 0.7129 | 0.7323 |
| Stacking Ensemble | 0.7394 | 0.7200 | 0.7200 | 0.7201 | 0.7273 | 0.7129 | 0.7273 |
| Random Forest | 0.7380 | 0.7103 | 0.7125 | 0.7126 | 0.7231 | 0.6980 | 0.7273 |
| Soft Voting Ensemble | 0.7261 | 0.7168 | 0.7175 | 0.7176 | 0.7259 | 0.7079 | 0.7273 |
| SVM | 0.7171 | 0.7186 | 0.7200 | 0.7201 | 0.7296 | 0.7079 | 0.7323 |
| XGBoost | 0.7150 | 0.6751 | 0.6800 | 0.6802 | 0.6927 | 0.6584 | 0.7020 |
| KNN | 0.6831 | 0.6888 | 0.6725 | 0.6720 | 0.6621 | 0.7178 | 0.6263 |

### Interpretation of Results

- Logistic Regression is currently the strongest single model on held-out ROC-AUC.
- Stacking performs almost identically to Logistic Regression and provides a strong ensemble baseline.
- Random Forest remains competitive and is a useful non-linear complement.
- Soft Voting underperforms the top single model because weaker but correlated models dilute the strongest probability signal.

This is a useful research finding in itself: for this dataset, a strong linear baseline remains hard to beat, and not every ensemble strategy improves ranking performance.

## Cross-Validation Summary

5-fold cross-validation on the training split produced the following mean metrics:

| Model | ROC-AUC Mean | F1 Mean | Accuracy Mean | Precision Mean | Recall Mean |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6888 | 0.6902 | 0.6912 | 0.6980 | 0.6827 |
| Random Forest | 0.6851 | 0.6878 | 0.6856 | 0.6883 | 0.6876 |
| KNN | 0.6836 | 0.6744 | 0.6656 | 0.6627 | 0.6865 |
| SVM | 0.6828 | 0.6936 | 0.6944 | 0.7011 | 0.6864 |
| XGBoost | 0.6630 | 0.6536 | 0.6506 | 0.6535 | 0.6542 |

## Repository Structure

```text
├── configs/                    # Model, path, and training configuration
├── data/
│   ├── raw/                    # Input data (Excel dataset)
│   └── processed/              # Saved processed feature snapshot
├── models/
│   ├── trained_models/         # Saved base learners
│   └── ensemble_models/        # Saved voting and stacking ensembles
├── results/
│   ├── figures/                # ROC, PR, calibration, confusion matrices, comparisons
│   ├── tables/                 # CV results and analysis tables
│   ├── predictions/            # Saved test-set predictions
│   ├── model_results.csv       # Final metrics table
│   └── training_summary.json   # Run summary and metadata
├── src/
│   ├── data/                   # Loading, validation, preprocessing, feature engineering
│   ├── models/                 # Base-model training and tuning utilities
│   ├── ensemble/               # Voting and stacking implementations
│   ├── evaluation/             # Metrics, ROC analysis, significance testing
│   ├── explainability/         # Feature importance and uncertainty helpers
│   ├── features/               # Feature selection and statistical summaries
│   ├── pipelines/              # Training, inference, and full analysis pipelines
│   └── utils/                  # Helpers for logging, saving, plotting, etc.
├── main.py
├── requirements.txt
├── environment.yml
└── README.md
```

## Installation

### Requirements
- Python 3.11+
- `pip` or `conda`

### Install with pip

```bash
python -m pip install -r requirements.txt
```

### Optional Explainability Dependencies

To enable SHAP-based explanation workflows:

```bash
python -m pip install shap
```

### Install with conda

```bash
conda env create -f environment.yml
conda activate prostate-cancer
```

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for the full text.

## Supporting Documentation

Additional project notes are available in:
- [Methodology](docs/methodology.md)
- [Clinical Relevance](docs/clinical_relevance.md)
- [Architecture](docs/architecture.md)
- [Deployment Guide](docs/deployment_guide.md)
- [References](docs/references.md)

## Usage

### Run the Main Training Pipeline

```bash
python src/pipelines/training_pipeline.py
```

This will:
- load the Excel dataset from `data/raw/Prostate_Cancer.xlsx`
- validate the schema
- build engineered clinical features
- run 5-fold cross-validation
- train all enabled base models
- train soft voting and stacking ensembles
- evaluate on a held-out test split
- save models and results to `models/` and `results/`

### Run the Extended Analysis Pipeline

```bash
python -m src.pipelines.full_pipeline
```

This additionally generates:
- ROC analysis summaries
- feature importance tables
- uncertainty analysis
- pairwise model significance tables
- example clinical interpretation output

### Run a Single Base Model

```bash
python -m src.models.train_logistic_regression
python -m src.models.train_random_forest
python -m src.models.train_xgboost
python -m src.models.train_svm
python -m src.models.train_knn
```

### Inference on New Patient Data

```python
from src.pipelines.inference_pipeline import InferencePipeline
import pandas as pd

pipeline = InferencePipeline("models")
pipeline.load_models()

X_new = pd.read_excel("data/new_patients.xlsx")
predictions, probabilities = pipeline.predict(X_new, method="stacking")
```

## Expected Input Columns

Training data should contain:

```text
id,age,BMI,PSA_level,free_PSA_ratio,testosterone_level,alkaline_phosphatase,family_history,smoking_status,radius,texture,perimeter,area,smoothness,compactness,symmetry,fractal_dimension,diagnosis_result
```

## Saved Outputs

After training, the project stores:

- base models in `models/trained_models/`
- ensemble models in `models/ensemble_models/`
- metrics in `results/model_results.csv`
- evaluation figures in `results/figures/`
- cross-validation tables in `results/tables/`
- test-set predictions in `results/predictions/`
- run metadata in `results/training_summary.json`

## Clinical Disclaimer

This repository is for machine learning research and educational demonstration. It is **not** a validated clinical diagnostic system. Any real-world clinical use would require:
- external validation
- prospective study design
- regulatory review
- clinician oversight

## References

Selected references supporting the modeling, evaluation, and explainability choices used in this repository are listed below. A fuller bibliography is available in [Extended References](docs/references.md).

- Cox, D. R. (1958). *The Regression Analysis of Binary Sequences*. Journal of the Royal Statistical Society: Series B (Methodological), 20(2), 215-232. https://doi.org/10.1111/j.2517-6161.1958.tb00292.x
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324
- Cortes, C., and Vapnik, V. (1995). *Support-vector networks*. Machine Learning, 20, 273-297. https://doi.org/10.1007/BF00994018
- Cover, T. M., and Hart, P. E. (1967). *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory, 13(1), 21-27. https://doi.org/10.1109/TIT.1967.1053964
- Chen, T., and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. https://doi.org/10.1145/2939672.2939785
- Wolpert, D. H. (1992). *Stacked generalization*. Neural Networks, 5(2), 241-259. https://doi.org/10.1016/S0893-6080(05)80023-1
- Zhou, Z.-H. (2012). *Ensemble Methods: Foundations and Algorithms*. CRC Press. https://doi.org/10.1201/b12207
- Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection*. In *Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI)*, 1137-1143. https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf
- Lundberg, S. M., and Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. In *Advances in Neural Information Processing Systems 30*, 4765-4774. https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
- Fisher, A., Rudin, C., and Dominici, F. (2019). *All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously*. Journal of Machine Learning Research, 20(177), 1-81. https://jmlr.csail.mit.edu/papers/v20/18-760.html
- Saito, T., and Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
