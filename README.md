# 🏥 MediCost AI — Indian Medical Insurance Predictor & Actuarial ML Suite

An end-to-end Machine Learning platform engineered to estimate individual health insurance premiums across Indian regions with actuarial precision. Built with a **multi-model training pipeline** (Random Forest, Gradient Boosting, Linear Regression) and **two independent frontend interfaces**: a modern **Full-Stack Glassmorphism Web App** (Flask + JavaScript + Plotly) and an **8-Tab Streamlit Dashboard**.

---

## 🚀 Key Highlights

- **Real Dataset for India**: Grounded in the gold-standard Kaggle Medical Cost Personal dataset adapted specifically for the Indian healthcare ecosystem, featuring realistic Indian regions (**North, South, East, West, Central, Northeast**) and charges denominated in **Indian Rupees (₹)**.
- **Multi-Model Tournament**: Automatically benchmarks **Random Forest**, **Gradient Boosting**, and **Linear Regression** on train, test, and 5-fold cross-validation sets.
- **Top Performance**: Best model (**Random Forest**) achieves **R² = 0.8453**, **Test MAE = ₹76,863**, and **5-Fold CV R² = 0.8166**.
- **Complete Metric Scorecard**: Evaluates **14+ regression metrics** including R², Adjusted R², Explained Variance, MAE, MSE, RMSE, MAPE, Median AE, Max Error, Residual Mean, Residual Std, Skewness, Kurtosis, and the **Shapiro-Wilk Normality Test**.
- **Two Deployment Modes**:
  1. **Enterprise Full-Stack Web App** (`python server.py` -> `http://localhost:5000`): Reactive sliders, real-time recalculations in ₹ without page reloads, risk score meters, and interactive Plotly visuals.
  2. **Streamlit Analytics Dashboard** (`streamlit run app.py` -> `http://localhost:8501`): 8-tab statistical suite.

---

## 📂 Project Architecture

```
MLPROJECT/
├── insurance.csv            # Actual Dataset adapted for India (1,338 records, INR charges)
├── generate_dataset.py      # Dataset pipeline: downloads Kaggle dataset & adapts for India
├── train.py                 # Multi-model training, 5-fold CV, metrics generator
├── server.py                # Production Flask backend (REST API + static server)
├── static/
│   ├── index.html           # Enterprise glassmorphism single-page app
│   ├── style.css            # Dark/light theme design system
│   └── app.js               # Reactive frontend logic & Plotly chart engines
├── app.py                   # Streamlit multi-tab analytics dashboard
├── requirements.txt         # Project dependencies
├── insurance_model.pkl      # Serialized champion model pipeline (auto-generated)
├── model_metrics.json       # Persisted metrics, CV folds & learning curves (auto-generated)
├── test_predictions.csv     # Test set predictions & residuals for diagnostic plotting
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data & Train Models

```bash
# Download and Indianize the Kaggle dataset
python generate_dataset.py

# Train all models, run cross-validation & extract metrics
python train.py
```

### 3. Launch Frontend

#### Option A: Enterprise Web App (Recommended — Real-Time & Sleek)
```bash
python server.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.

#### Option B: Streamlit Dashboard
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 📊 Comprehensive Performance Metrics Explained

| Metric | Champion Value | Statistical Meaning & Significance |
|---|---|---|
| **R² Score** | `0.8453` | **Coefficient of Determination**: 84.5% of variance in medical charges is explained by the model. |
| **Adjusted R²** | `0.8417` | Corrects R² by penalizing extraneous features ($1 - \frac{(1-R^2)(n-1)}{n-p-1}$). |
| **Explained Variance** | `0.8467` | Measures proportion of variance accounted for; confirms zero systematic bias offset. |
| **MAE** | `₹76,863` | **Mean Absolute Error**: Average expected rupee discrepancy across predictions; robust against outliers. |
| **RMSE** | `₹134,317` | **Root Mean Squared Error**: Standard deviation of unexplained residuals; penalizes large prediction blunders. |
| **MAPE** | `33.24%` | **Mean Absolute Percentage Error**: Average percentage error relative to actual charge magnitude. |
| **Median AE** | `₹40,131` | **Median Absolute Error**: Half of all test predictions are within ₹40,131 of actual costs. |
| **Max Error** | `₹681,750` | Worst-case single prediction outlier on the test set. |
| **Residual Mean** | `-₹13,058` | Average signed error ($\approx 0$ indicates an unbiased estimator). |
| **Residual Std** | `₹133,680` | Standard deviation of residual errors. |
| **5-Fold CV R²** | `0.8166 ± 0.048` | Rigorous cross-validation confirming generalizability across unseen folds. |

---

## 🏆 Multi-Model Tournament Results

| Model Architecture | Test R² | Adj R² | Test MAE | Test RMSE | MAPE | CV R² Mean | Decision |
|---|---|---|---|---|---|---|---|
| **Random Forest** | **0.8453** | **0.8417** | **₹76,863** | **₹134,317** | **33.2%** | **0.8166** | 👑 **Selected (Champion)** |
| **Gradient Boosting** | 0.8311 | 0.8272 | ₹79,557 | ₹140,347 | 34.6% | 0.7851 | Benchmark Finalist |
| **Linear Regression** | 0.7715 | 0.7662 | ₹117,070 | ₹163,244 | 48.9% | 0.7228 | Baseline |

---

## 🔬 Actuarial Insights: Key Cost Drivers (India)

1. **Smoker Status (~61% total attribution)**: Non-smoking status is the single largest cost reducer; active smoking adds a significant premium multiplier.
2. **BMI (~20% attribution)**: Severe non-linear escalation occurs when BMI exceeds $30.0\text{ kg/m}^2$, especially compounded with smoking.
3. **Age (~13% attribution)**: Progressive increase per year of age as risk of hospitalization escalates.
4. **Children & Indian Region (~5% attribution)**: Regional adjustments for medical infrastructure tier (Tier-1 North/West vs Tier-2/3 Central/East/Northeast) and dependent coverage.

---

## 🛠️ API Reference (`server.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the interactive Web Application |
| `/api/status` | `GET` | Server health check and active champion model info |
| `/api/predict` | `POST` | Real-time prediction, confidence intervals, risk score, and factor breakdown |
| `/api/metrics` | `GET` | All 14+ metrics, CV fold scores, and learning curve data |
| `/api/residuals` | `GET` | Test actuals, fitted values, residuals, and Shapiro-Wilk normality stats |
| `/api/sensitivity` | `POST` | Marginal sweeps for Age, BMI, Children, Smoker, and Region |
| `/api/eda` | `GET` | Dataset descriptive statistics, correlation matrix, and data samples |
