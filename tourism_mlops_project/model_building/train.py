
from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
PREPARED_DIR = ROOT / "prepared_data"
DEPLOY_DIR = ROOT / "deployment"
ARTIFACT_DIR = ROOT / "artifacts"

DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Rubric: load train/test data downloaded from workflow artifact.
X_train = pd.read_csv(PREPARED_DIR / "X_train.csv")
X_test = pd.read_csv(PREPARED_DIR / "X_test.csv")
y_train = pd.read_csv(PREPARED_DIR / "y_train.csv")["ProdTaken"].astype(int)
y_test = pd.read_csv(PREPARED_DIR / "y_test.csv")["ProdTaken"].astype(int)

categorical = X_train.select_dtypes(include="object").columns.tolist()
numeric = [c for c in X_train.columns if c not in categorical]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )),
])

param_grid = {
    "classifier__n_estimators": [200, 400],
    "classifier__max_depth": [None, 12],
    "classifier__min_samples_leaf": [1, 2],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
)

mlflow.set_experiment("visit-with-us-tourism-ci")

with mlflow.start_run():
    search.fit(X_train, y_train)
    model = search.best_estimator_

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
        "pr_auc": average_precision_score(y_test, prob),
    }

    # Log every selected tuned parameter.
    mlflow.log_params({
        key.replace("classifier__", ""): value
        for key, value in search.best_params_.items()
    })
    mlflow.log_metric("best_cv_roc_auc", float(search.best_score_))
    for name, value in metrics.items():
        mlflow.log_metric(f"test_{name}", float(value))

    # Log complete search space and CV table.
    grid_path = ARTIFACT_DIR / "parameter_grid.json"
    cv_path = ARTIFACT_DIR / "cv_results.csv"
    metrics_path = ARTIFACT_DIR / "metrics.json"

    grid_path.write_text(json.dumps(param_grid, indent=2), encoding="utf-8")
    pd.DataFrame(search.cv_results_).to_csv(cv_path, index=False)
    metrics_path.write_text(json.dumps({
        "best_cv_roc_auc": float(search.best_score_),
        "best_params": search.best_params_,
        "test_metrics": {k: float(v) for k, v in metrics.items()},
    }, indent=2), encoding="utf-8")

    model_path = DEPLOY_DIR / "tourism_model.joblib"
    joblib.dump(model, model_path)

    mlflow.log_artifact(str(grid_path))
    mlflow.log_artifact(str(cv_path))
    mlflow.log_artifact(str(metrics_path))

    example = X_train.head(5)
    signature = infer_signature(example, model.predict(example))
    
    # Comprehensive list of trusted types for sklearn RandomForest pipeline with all dependencies
    # This includes all numpy and sklearn types that may be serialized with the model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="tourism_random_forest",
        signature=signature,
        input_example=example,
        skops_trusted_types=[
            "sklearn.pipeline.Pipeline",
            "sklearn.compose._column_transformer.ColumnTransformer",
            "sklearn.ensemble._forest.RandomForestClassifier",
            "sklearn.impute._simple.SimpleImputer",
            "sklearn.preprocessing._encoders.OneHotEncoder",
            "sklearn.preprocessing._scaler.StandardScaler",
            "sklearn.preprocessing._label.LabelEncoder",
            "sklearn.tree._tree.Tree",
            "sklearn.tree._classes.DecisionTreeClassifier",
            "numpy.ndarray",
            "numpy.dtype",
            "numpy.generic",
            "numpy.int64",
            "numpy.float64",
            "numpy.bool_",
            "numpy.random.RandomState",
        ],
    )

    # Simple CI quality gate.
    MIN_ROC_AUC = 0.80
    if metrics["roc_auc"] < MIN_ROC_AUC:
        raise RuntimeError(
            f"Quality gate failed: ROC-AUC {metrics['roc_auc']:.4f} < {MIN_ROC_AUC}"
        )

    print("MODEL TRAINING PASSED")
    print("Best parameters:", search.best_params_)
    print("Metrics:", metrics)
    print("Saved model:", model_path)
