"""
train.py - Indian Medical Insurance Cost Predictor: Multi-Model Training Pipeline
==================================================================================
Trains and compares 3 regression models (RandomForest, GradientBoosting,
LinearRegression), computes comprehensive evaluation metrics, saves
the best pipeline, predictions, and all metrics for the Streamlit dashboard.

Usage:
    python train.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    max_error,
    median_absolute_error,
    mean_absolute_percentage_error,
)
import joblib

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "insurance.csv")
MODEL_PATH = os.path.join(BASE_DIR, "insurance_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "test_predictions.csv")
LEARNING_CURVES_PATH = os.path.join(BASE_DIR, "learning_curves.json")

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
TARGET = "charges"

print("=" * 60)
print("  Indian Medical Insurance Cost Predictor - Training Pipeline")
print("=" * 60)

# ============================================================
# 1. Load and inspect dataset
# ============================================================
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{DATA_PATH}'.\n"
        "Run 'python generate_dataset.py' to download and Indianize "
        "the Kaggle Medical Cost dataset."
    )

df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"     Target column: '{TARGET}'")
print(f"     Numeric features: {NUMERIC_FEATURES}")
print(f"     Categorical features: {CATEGORICAL_FEATURES}\n")

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

# ============================================================
# 2. Build preprocessing pipeline
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ]
)

# ============================================================
# 3. Define candidate models
# ============================================================
MODELS = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_split=5, random_state=42,
    ),
    "Linear Regression": LinearRegression(),
}

# ============================================================
# 4. Train-test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"[OK] Split -> Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}\n")

# ============================================================
# 5. Train, evaluate, and compare all models
# ============================================================
all_metrics = {}
best_r2 = -np.inf
best_model_name = None
best_pipeline = None
best_predictions = None


def compute_metrics(y_true, y_pred):
    """Compute a comprehensive set of regression metrics."""
    residuals = y_true - y_pred
    n = len(y_true)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # Number of features after preprocessing
    p = X_train.shape[1]
    adj_r2 = 1 - ((1 - r2_score(y_true, y_pred)) * (n - 1) / (n - p - 1))

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "adjusted_r2": float(adj_r2),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
    }


for model_name, model in MODELS.items():
    print(f"--- Training: {model_name} ---")

    # Build pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model),
    ])

    # Fit
    pipeline.fit(X_train, y_train)

    # Predict on train and test
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Compute metrics on both sets
    train_metrics = compute_metrics(y_train.values, y_train_pred)
    test_metrics = compute_metrics(y_test.values, y_test_pred)

    # Cross-validation (5-fold)
    cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
    cv_neg_mae = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    cv_neg_rmse = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")

    model_data = {
        "train": train_metrics,
        "test": test_metrics,
        "cv_r2_scores": cv_r2.tolist(),
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "cv_mae_scores": (-cv_neg_mae).tolist(),
        "cv_mae_mean": float(-cv_neg_mae.mean()),
        "cv_mae_std": float(cv_neg_mae.std()),
        "cv_rmse_scores": (-cv_neg_rmse).tolist(),
        "cv_rmse_mean": float(-cv_neg_rmse.mean()),
        "cv_rmse_std": float(cv_neg_rmse.std()),
    }

    all_metrics[model_name] = model_data

    # Print summary
    print(f"  Train R2: {train_metrics['r2']:.4f}  |  Test R2: {test_metrics['r2']:.4f}")
    print(f"  Test MAE: INR {test_metrics['mae']:,.2f}  |  Test RMSE: INR {test_metrics['rmse']:,.2f}")
    print(f"  CV R2: {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")
    print()

    # Track best model
    if test_metrics["r2"] > best_r2:
        best_r2 = test_metrics["r2"]
        best_model_name = model_name
        best_pipeline = pipeline
        best_predictions = y_test_pred


print(f"[BEST] {best_model_name} with Test R2 = {best_r2:.4f}\n")

# ============================================================
# 6. Extract feature importances from the best model
# ============================================================
ohe = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
all_feature_names = NUMERIC_FEATURES + cat_feature_names

feature_importances = {}
regressor = best_pipeline.named_steps["regressor"]
if hasattr(regressor, "feature_importances_"):
    importances = regressor.feature_importances_
    for name, imp in zip(all_feature_names, importances):
        feature_importances[name] = float(imp)
elif hasattr(regressor, "coef_"):
    # For linear regression, use absolute coefficients as proxy
    coefs = np.abs(regressor.coef_)
    coefs_norm = coefs / coefs.sum() if coefs.sum() > 0 else coefs
    for name, imp in zip(all_feature_names, coefs_norm):
        feature_importances[name] = float(imp)

# Also get importances from ALL tree-based models
all_importances = {}
for model_name in ["Random Forest", "Gradient Boosting"]:
    pipeline_temp = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", MODELS[model_name]),
    ])
    pipeline_temp.fit(X_train, y_train)
    imp = pipeline_temp.named_steps["regressor"].feature_importances_
    all_importances[model_name] = {
        name: float(v) for name, v in zip(all_feature_names, imp)
    }

print("-- Feature Importances (Best Model) --")
sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
for rank, (name, imp) in enumerate(sorted_features, 1):
    bar = "#" * int(imp * 60)
    print(f"  {rank:>2}. {name:<25s} {imp:.4f}  {bar}")
print()

# ============================================================
# 7. Compute learning curves for the best model type
# ============================================================
print("[OK] Computing learning curves ...")
train_sizes_abs, train_lc_scores, val_lc_scores = learning_curve(
    best_pipeline, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring="r2", n_jobs=-1,
)

lc_data = {
    "train_sizes": train_sizes_abs.tolist(),
    "train_scores_mean": train_lc_scores.mean(axis=1).tolist(),
    "train_scores_std": train_lc_scores.std(axis=1).tolist(),
    "val_scores_mean": val_lc_scores.mean(axis=1).tolist(),
    "val_scores_std": val_lc_scores.std(axis=1).tolist(),
}

# ============================================================
# 8. Save everything
# ============================================================
# 8a. Save best model pipeline
joblib.dump(best_pipeline, MODEL_PATH)
print(f"[OK] Best model pipeline saved -> {MODEL_PATH}")

# 8b. Save test predictions (for residual plots in the app)
pred_df = pd.DataFrame({
    "actual": y_test.values,
    "predicted": best_predictions,
    "residual": y_test.values - best_predictions,
})
# Also include features for deeper analysis
pred_df = pd.concat([pred_df, X_test.reset_index(drop=True)], axis=1)
pred_df.to_csv(PREDICTIONS_PATH, index=False)
print(f"[OK] Test predictions saved -> {PREDICTIONS_PATH}")

# 8c. Save all metrics as JSON
output = {
    "best_model": best_model_name,
    "models": all_metrics,
    "feature_importances": feature_importances,
    "all_importances": all_importances,
    "feature_names": all_feature_names,
    "dataset_info": {
        "total_rows": int(df.shape[0]),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "num_features_raw": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
        "num_features_encoded": len(all_feature_names),
    },
    "learning_curves": lc_data,
}

with open(METRICS_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"[OK] All metrics saved -> {METRICS_PATH}")

print(f"\n{'=' * 60}")
print(f"  DONE! Launch the dashboard:  streamlit run app.py")
print(f"{'=' * 60}")
