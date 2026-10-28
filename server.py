"""
server.py - Production Flask Backend for Indian Medical Insurance Cost Predictor
================================================================================
Exposes a modern REST API for real-time ML inference, comprehensive
evaluation metric scorecards, model leaderboards, residual diagnostics,
dynamic sensitivity sweeps, and dataset profiling.

Usage:
    python server.py
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
import joblib
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "insurance_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "test_predictions.csv")
DATA_PATH = os.path.join(BASE_DIR, "insurance.csv")

app = Flask(__name__, static_folder="static")

# Load persistent assets
model = None
metrics_data = None
predictions_df = None
dataset_df = None

def init_assets():
    global model, metrics_data, predictions_df, dataset_df
    if os.path.isfile(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.isfile(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
    if os.path.isfile(PREDICTIONS_PATH):
        predictions_df = pd.read_csv(PREDICTIONS_PATH)
    if os.path.isfile(DATA_PATH):
        dataset_df = pd.read_csv(DATA_PATH)

init_assets()

# ────────────────────────────────────────────────────────────
# Static route
# ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)

# ────────────────────────────────────────────────────────────
# API: Health & Status
# ────────────────────────────────────────────────────────────
@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "metrics_loaded": metrics_data is not None,
        "best_model": metrics_data.get("best_model") if metrics_data else None,
    })

# ────────────────────────────────────────────────────────────
# API: Real-time Prediction
# ────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 500

    data = request.get_json(force=True)
    try:
        age = float(data.get("age", 30))
        sex = str(data.get("sex", "male")).lower()
        bmi = float(data.get("bmi", 25.0))
        children = int(data.get("children", 0))
        smoker = str(data.get("smoker", "no")).lower()
        region = str(data.get("region", "north")).lower()

        input_df = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
        }])

        pred_charge = float(model.predict(input_df)[0])

        # Calculate risk score and classification
        risk_score = 0
        risk_factors = []
        if smoker == "yes":
            risk_score += 55
            risk_factors.append("Smoker Status (+55 pts)")
        if bmi >= 30:
            bmi_pts = min(30, int((bmi - 25) * 2))
            risk_score += bmi_pts
            risk_factors.append(f"BMI {bmi:.1f} ({bmi_pts} pts)")
        elif bmi < 18.5:
            risk_score += 5
            risk_factors.append("Underweight (+5 pts)")
        if age >= 50:
            age_pts = int((age - 40) * 0.8)
            risk_score += age_pts
            risk_factors.append(f"Age {int(age)} ({age_pts} pts)")
        if children >= 3:
            risk_score += 8
            risk_factors.append(f"{children} Children (+8 pts)")

        if risk_score < 25:
            tier = "Low Risk"
            tier_color = "#38ef7d"
        elif risk_score < 50:
            tier = "Moderate Risk"
            tier_color = "#f1c40f"
        elif risk_score < 75:
            tier = "High Risk"
            tier_color = "#e67e22"
        else:
            tier = "Severe Risk"
            tier_color = "#e74c3c"

        # Baseline comparison: non-smoker, BMI 22, age 30, 0 kids
        base_df = pd.DataFrame([{
            "age": 30, "sex": sex, "bmi": 22.0, "children": 0, "smoker": "no", "region": "south"
        }])
        baseline_cost = float(model.predict(base_df)[0])
        cost_multiplier = pred_charge / max(baseline_cost, 1.0)

        # Confidence Interval approximation from test RMSE
        best_model_name = metrics_data.get("best_model", "Gradient Boosting") if metrics_data else "Gradient Boosting"
        rmse = metrics_data["models"][best_model_name]["test"]["rmse"] if metrics_data else 2000.0
        ci_lower = max(1000.0, pred_charge - 1.96 * (rmse / 2.0))
        ci_upper = pred_charge + 1.96 * (rmse / 2.0)

        return jsonify({
            "predicted_charge": round(pred_charge, 2),
            "formatted_charge": f"\u20b9{pred_charge:,.2f}",
            "ci_lower": round(ci_lower, 2),
            "ci_upper": round(ci_upper, 2),
            "risk_score": min(100, risk_score),
            "risk_tier": tier,
            "risk_tier_color": tier_color,
            "risk_factors": risk_factors,
            "baseline_comparison": {
                "baseline_cost": round(baseline_cost, 2),
                "delta": round(pred_charge - baseline_cost, 2),
                "multiplier": round(cost_multiplier, 2),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ────────────────────────────────────────────────────────────
# API: Full Metrics & Leaderboard
# ────────────────────────────────────────────────────────────
@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    if metrics_data is None:
        return jsonify({"error": "Metrics not found. Run train.py first."}), 500
    return jsonify(metrics_data)

# ────────────────────────────────────────────────────────────
# API: Residuals & Statistical Diagnostics
# ────────────────────────────────────────────────────────────
@app.route("/api/residuals", methods=["GET"])
def get_residuals():
    if predictions_df is None:
        return jsonify({"error": "Predictions file not found."}), 500

    actual = predictions_df["actual"].values
    predicted = predictions_df["predicted"].values
    residual = predictions_df["residual"].values

    # Statistical moments and normality test
    mean_res = float(np.mean(residual))
    std_res = float(np.std(residual))
    skew_res = float(stats.skew(residual))
    kurt_res = float(stats.kurtosis(residual))

    # Sample for Shapiro-Wilk if > 5000
    sample_res = residual if len(residual) <= 5000 else np.random.choice(residual, 5000, replace=False)
    shapiro_w, shapiro_p = stats.shapiro(sample_res)

    # Q-Q plot theoretical points
    sorted_res = np.sort(residual).tolist()
    n = len(sorted_res)
    theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n).tolist()

    return jsonify({
        "actual": actual.tolist(),
        "predicted": predicted.tolist(),
        "residual": residual.tolist(),
        "statistics": {
            "mean": round(mean_res, 4),
            "std": round(std_res, 4),
            "skewness": round(skew_res, 4),
            "kurtosis": round(kurt_res, 4),
            "shapiro_w": round(float(shapiro_w), 4),
            "shapiro_p": float(shapiro_p),
            "is_normal": bool(shapiro_p > 0.05),
        },
        "qq_plot": {
            "theoretical": theoretical_quantiles,
            "sample": sorted_res,
        }
    })

# ────────────────────────────────────────────────────────────
# API: Dynamic Sensitivity Sweep
# ────────────────────────────────────────────────────────────
@app.route("/api/sensitivity", methods=["POST"])
def get_sensitivity():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json(force=True)
    base = {
        "age": float(data.get("age", 30)),
        "sex": str(data.get("sex", "male")),
        "bmi": float(data.get("bmi", 25.0)),
        "children": int(data.get("children", 0)),
        "smoker": str(data.get("smoker", "no")),
        "region": str(data.get("region", "north")),
    }

    variable = data.get("variable", "all")
    results = {}

    if variable in ["all", "age"]:
        ages = list(range(18, 65))
        dfs = [dict(base, age=a) for a in ages]
        preds = model.predict(pd.DataFrame(dfs)).tolist()
        results["age"] = {"x": ages, "y": [round(p, 2) for p in preds]}

    if variable in ["all", "bmi"]:
        bmis = [round(b, 1) for b in np.arange(15.0, 50.5, 0.5)]
        dfs = [dict(base, bmi=b) for b in bmis]
        preds = model.predict(pd.DataFrame(dfs)).tolist()
        results["bmi"] = {"x": bmis, "y": [round(p, 2) for p in preds]}

    if variable in ["all", "children"]:
        kids = list(range(6))
        dfs = [dict(base, children=k) for k in kids]
        preds = model.predict(pd.DataFrame(dfs)).tolist()
        results["children"] = {"x": kids, "y": [round(p, 2) for p in preds]}

    if variable in ["all", "smoker"]:
        statuses = ["no", "yes"]
        dfs = [dict(base, smoker=s) for s in statuses]
        preds = model.predict(pd.DataFrame(dfs)).tolist()
        results["smoker"] = {
            "labels": ["Non-Smoker", "Smoker"],
            "y": [round(p, 2) for p in preds],
            "penalty": round(preds[1] - preds[0], 2),
            "percentage_increase": round((preds[1] - preds[0]) / max(preds[0], 1) * 100, 1),
        }

    if variable in ["all", "region"]:
        regions = ["north", "south", "east", "west", "central", "northeast"]
        dfs = [dict(base, region=r) for r in regions]
        preds = model.predict(pd.DataFrame(dfs)).tolist()
        results["region"] = {"labels": regions, "y": [round(p, 2) for p in preds]}

    return jsonify(results)

# ────────────────────────────────────────────────────────────
# API: Dataset Profiler & EDA
# ────────────────────────────────────────────────────────────
@app.route("/api/eda", methods=["GET"])
def get_eda():
    if dataset_df is None:
        return jsonify({"error": "Dataset not found."}), 500

    # Summary statistics
    num_cols = ["age", "bmi", "children", "charges"]
    desc = dataset_df[num_cols].describe().round(2).to_dict()

    # Correlation matrix
    corr = dataset_df[num_cols].corr().round(3).to_dict()

    # Category counts
    cat_counts = {
        "sex": dataset_df["sex"].value_counts().to_dict(),
        "smoker": dataset_df["smoker"].value_counts().to_dict(),
        "region": dataset_df["region"].value_counts().to_dict(),
    }

    # Charges by smoker
    charges_by_smoker = {
        "non_smoker": dataset_df[dataset_df["smoker"] == "no"]["charges"].tolist(),
        "smoker": dataset_df[dataset_df["smoker"] == "yes"]["charges"].tolist(),
    }

    # Sample rows (first 30)
    sample_records = dataset_df.head(30).to_dict(orient="records")

    return jsonify({
        "summary": desc,
        "correlation": corr,
        "categorical_counts": cat_counts,
        "sample": sample_records,
        "total_records": len(dataset_df),
        "charges_by_smoker": charges_by_smoker,
    })

if __name__ == "__main__":
    print("=" * 60)
    print("  Indian Medical Insurance Cost Predictor - Flask Server")
    print("  Web Dashboard: http://localhost:5000")
    print("  API Docs/Status: http://localhost:5000/api/status")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
