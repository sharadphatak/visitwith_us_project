# Visit With Us: Tourism ML Ops Project

A complete machine learning operations pipeline for predicting customer purchase propensity in a wellness tourism business. This project demonstrates end-to-end ML workflow automation using GitHub Actions CI/CD, including data validation, feature engineering, model training, hyperparameter tuning, and automated model deployment.

## Overview

**Problem:** Predict which customers are most likely to purchase a tourism package before proactive sales contact, enabling targeted campaign prioritization.

**Solution:** Automated MLOps pipeline that:
- Validates and registers raw data assets
- Performs stratified train/test split
- Tunes a scikit-learn RandomForest classifier with GridSearchCV
- Evaluates on ROC-AUC and enforces quality gates
- Logs models and artifacts to MLflow
- Deploys predictions via a Streamlit web interface

## Architecture

```
tourism_mlops_project/
  data/                 Raw CSV input (tourism.csv)
  model_building/       Training scripts
    ├── data_register.py    Dataset validation & integrity checks
    ├── prep.py             Train/test split & feature engineering
    └── train.py            Model tuning & evaluation
  deployment/           Production inference
    ├── app.py              Streamlit UI for predictions
    ├── predictor.py        Model loading & inference logic
    └── requirements.txt    Deployment-only dependencies
  artifacts/            Training outputs (committed by CI)
    ├── metrics.json        Test set performance
    ├── parameter_grid.json Grid search configuration
    └── cv_results.csv      Cross-validation folds
  prepared_data/        Train/test splits (ephemeral)
  requirements.txt      Model training dependencies
```

**Data Flow:**
1. Raw data → **data_register.py** validates schema, columns, duplicates
2. Clean data → **prep.py** applies feature engineering, creates train/test split
3. Train data → **train.py** trains RandomForest, logs to MLflow, saves joblib model
4. Model → **app.py** loads and serves predictions via Streamlit

## Setup

### Local Development

```bash
# Clone repository
git clone https://github.com/sharadphatak/visitwith_us_project.git
cd visitwith_us_project

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r tourism_mlops_project/requirements.txt
```

### Running Locally

**Step 1: Register & validate data**
```bash
python tourism_mlops_project/model_building/data_register.py
```

**Step 2: Prepare train/test split**
```bash
python tourism_mlops_project/model_building/prep.py
```

**Step 3: Train model**
```bash
python tourism_mlops_project/model_building/train.py
```

**Step 4: Deploy Streamlit app**
```bash
pip install -r tourism_mlops_project/deployment/requirements.txt
streamlit run tourism_mlops_project/deployment/app.py
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/pipeline.yml`) runs on every push to `main` that touches files in `tourism_mlops_project/`:

### Jobs

1. **data-registration** — Validates dataset integrity
2. **data-preparation** — Creates train/test splits, uploads as artifact
3. **model-training** — Trains RandomForest, evaluates, commits model artifacts

### Quality Gates

- **Minimum ROC-AUC:** 0.80 on test set (raises error if violated)
- **Model format:** Cloudpickle serialization ensures compatibility across NumPy/scikit-learn versions

### Outputs

The workflow commits the following files back to `main`:
- `tourism_mlops_project/deployment/tourism_model.joblib` — Trained model artifact
- `tourism_mlops_project/artifacts/metrics.json` — Test set performance
- `tourism_mlops_project/artifacts/parameter_grid.json` — Hyperparameter config
- `tourism_mlops_project/artifacts/cv_results.csv` — Cross-validation results

## Model Details

### Features

18 customer attributes: Age, TypeofContact, CityTier, DurationOfPitch, Occupation, Gender, NumberOfPersonVisiting, NumberOfFollowups, ProductPitched, PreferredPropertyStar, MaritalStatus, NumberOfTrips, Passport, PitchSatisfactionScore, OwnCar, NumberOfChildrenVisiting, Designation, MonthlyIncome

### Target

Binary classification: `ProdTaken` (0 = no purchase, 1 = purchase)

### Algorithm

**RandomForestClassifier** with:
- Hyperparameter tuning via 3-fold stratified GridSearchCV
- `n_estimators`: [200, 400]
- `max_depth`: [None, 12]
- `min_samples_leaf`: [1, 2]
- Class balancing via `class_weight="balanced_subsample"`

### Performance

Evaluation metrics on held-out test set:
- Accuracy
- Precision (zero_division=0)
- Recall (zero_division=0)
- F1-score
- ROC-AUC (primary metric)
- Precision-Recall AUC

## Deployment

### Streamlit Web App

Start the app:
```bash
streamlit run tourism_mlops_project/deployment/app.py
```

Features:
- Interactive form for 18 customer attributes
- Real-time purchase propensity prediction (0–100%)
- Adjustable decision threshold (10%–90%)
- Binary lead prioritization recommendation

### Requirements

The deployment uses a separate, lighter `requirements.txt` (no MLflow):
```
streamlit==1.64.0
pandas==2.2.3
numpy==2.2.6
scipy==1.15.3
scikit-learn==1.6.1
joblib==1.5.1
```

## Troubleshooting

### GitHub Actions Workflow Fails

**Issue:** `UntrustedTypesFoundException` during model logging  
**Solution:** Ensured via `train.py` — uses `serialization_format="cloudpickle"` instead of skops type allowlisting

**Issue:** Version incompatibility errors  
**Solution:** `requirements.txt` pinned to stable versions: numpy 2.2.6, scikit-learn 1.6.1, mlflow 3.6.0

### Streamlit App: "Model artifact not found"

**Solution:** Run the GitHub Actions workflow to train and commit the model, or run `train.py` locally

## Future Enhancements

- Add cross-validation metrics to MLflow UI
- Implement model registry and versioning
- Add explainability via SHAP values
- Containerize deployment with Docker
- Add data drift monitoring
- Implement A/B testing framework

## License

MIT

## Author

Sharad Phatak
