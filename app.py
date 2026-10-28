"""
app.py - Indian Medical Insurance Cost Predictor: Full Analytics Dashboard
==========================================================================
A comprehensive Streamlit dashboard featuring:
  - Interactive cost prediction
  - Complete model performance metrics (11+ metrics)
  - Multi-model comparison
  - Actual vs Predicted plots
  - Residual analysis (scatter, distribution, QQ-style)
  - Feature importance comparison across models
  - Correlation heatmap
  - Learning curves
  - Sensitivity analysis
  - Data explorer

Usage:
    streamlit run app.py
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Indian Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 40%, #2c5364 100%);
        padding: 2.2rem 2rem 1.8rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(56,239,125,0.06) 0%, transparent 50%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }
    .hero h1 { color: #fff; font-size: 2.3rem; font-weight: 800; margin: 0 0 0.3rem; position: relative; }
    .hero p { color: #94c5d6; font-size: 1rem; margin: 0; position: relative; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafb 100%);
        border: 1px solid #e8eef1;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.1); }
    .metric-card .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #6b8a99;
        margin-bottom: 0.3rem;
    }
    .metric-card .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #1a3a4a;
    }
    .metric-card .metric-sub {
        font-size: 0.72rem;
        color: #8fa8b5;
        margin-top: 0.15rem;
    }

    /* Green metric */
    .metric-good .metric-value { color: #0d9668; }
    /* Red metric */
    .metric-warn .metric-value { color: #d44a3a; }

    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 6px 28px rgba(17,153,142,0.35);
    }
    .prediction-card .label {
        color: rgba(255,255,255,0.85);
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .prediction-card .amount {
        color: #fff;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.2rem 0 0;
    }

    /* Section header */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #2c5364;
        margin: 1.5rem 0 0.6rem;
        padding-bottom: 0.35rem;
        border-bottom: 3px solid #11998e;
        display: inline-block;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027 0%, #1a3040 100%);
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #c8dfe8 !important;
    }

    /* Model badge */
    .model-badge {
        display: inline-block;
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Hero
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🏥 Medical Cost Predictor</h1>
    <p>Predict insurance charges &bull; Analyze model performance &bull; Explore feature insights</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Load artifacts
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource(show_spinner=False)
def load_model():
    path = os.path.join(BASE_DIR, "insurance_model.pkl")
    return joblib.load(path) if os.path.isfile(path) else None


@st.cache_data(show_spinner=False)
def load_metrics():
    path = os.path.join(BASE_DIR, "model_metrics.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_predictions():
    path = os.path.join(BASE_DIR, "test_predictions.csv")
    return pd.read_csv(path) if os.path.isfile(path) else None


@st.cache_data(show_spinner=False)
def load_dataset():
    path = os.path.join(BASE_DIR, "insurance.csv")
    return pd.read_csv(path) if os.path.isfile(path) else None


model = load_model()
metrics_data = load_metrics()
pred_df = load_predictions()
dataset = load_dataset()

if model is None or metrics_data is None:
    st.error(
        "**Trained model or metrics not found.** Run `python train.py` first."
    )
    st.stop()

# ============================================================
# Plotly theme helper
# ============================================================
COLORS = {
    "primary": "#11998e",
    "secondary": "#2c5364",
    "accent": "#38ef7d",
    "warning": "#e74c3c",
    "bg": "#fafcfd",
    "grid": "#eef2f5",
    "text": "#2c3e50",
}

MODEL_COLORS = {
    "Random Forest": "#11998e",
    "Gradient Boosting": "#6c5ce7",
    "Linear Regression": "#e17055",
}


def styled_layout(title="", height=400, **kwargs):
    """Return a common Plotly layout dict."""
    return dict(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=40 if title else 15, b=20),
        title=dict(text=title, font=dict(size=15, color=COLORS["text"])) if title else None,
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        plot_bgcolor=COLORS["bg"],
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"]),
        **kwargs,
    )


# ============================================================
# Sidebar - Prediction Inputs
# ============================================================
with st.sidebar:
    st.markdown("## 🔮 Predict Your Cost")
    st.markdown("---")

    age = st.slider("🎂 Age", 18, 100, 30)
    sex = st.selectbox("⚧ Sex", ["male", "female"])
    bmi = st.number_input("⚖️ BMI", 10.0, 60.0, 25.0, 0.1, format="%.1f")
    children = st.selectbox("👶 Children", list(range(6)))
    smoker = st.selectbox("🚬 Smoker", ["no", "yes"])
    region = st.selectbox("📍 Region", ["north", "south", "east", "west", "central", "northeast"])

    st.markdown("---")
    predict_btn = st.button("🔮  Predict Insurance Cost", use_container_width=True, type="primary")

    # Show best model info
    st.markdown("---")
    st.markdown("### 🏆 Best Model")
    st.markdown(f'<span class="model-badge">{metrics_data["best_model"]}</span>', unsafe_allow_html=True)
    best_test = metrics_data["models"][metrics_data["best_model"]]["test"]
    st.markdown(f"**R²** = {best_test['r2']:.4f}")
    st.markdown(f"**MAE** = \u20b9{best_test['mae']:,.0f}")

# ============================================================
# Main Content — Tabbed Dashboard
# ============================================================
tab_predict, tab_perf, tab_compare, tab_residuals, tab_features, tab_learning, tab_sensitivity, tab_data = st.tabs([
    "🔮 Prediction",
    "📊 Performance Metrics",
    "🏆 Model Comparison",
    "📉 Residual Analysis",
    "🔍 Feature Insights",
    "📈 Learning Curves",
    "🔬 Sensitivity",
    "📋 Data Explorer",
])

# ============================================================
# TAB 1: Prediction
# ============================================================
with tab_predict:
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown('<p class="section-header">💰 Your Estimate</p>', unsafe_allow_html=True)

        if predict_btn:
            input_df = pd.DataFrame([{
                "age": age, "bmi": bmi, "children": children,
                "sex": sex, "smoker": smoker, "region": region,
            }])
            prediction = model.predict(input_df)[0]

            st.markdown(f"""
            <div class="prediction-card">
                <div class="label">Estimated Annual Charges</div>
                <div class="amount">\u20b9{prediction:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            # Contextual insights
            insights = []
            if smoker == "yes":
                insights.append("🚬 **Smoking** is the #1 cost driver — expect significantly higher premiums.")
            if bmi > 30:
                insights.append("⚖️ **BMI > 30** (obese range) increases predicted costs substantially.")
            if age >= 55:
                insights.append("🎂 **Age 55+** is associated with higher medical expenses.")
            if children >= 3:
                insights.append("👶 **3+ children** adds noticeable cost to premiums.")

            if insights:
                st.markdown("#### 💡 Key Factors")
                for i in insights:
                    st.markdown(i)
        else:
            st.info("👈 Fill in the sidebar and press **Predict Insurance Cost**.")

    with col_b:
        st.markdown('<p class="section-header">📊 Quick Model Summary</p>', unsafe_allow_html=True)

        best = metrics_data["best_model"]
        m = metrics_data["models"][best]["test"]

        # Summary metric cards (2x2 grid)
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown(f"""
            <div class="metric-card metric-good">
                <div class="metric-label">R² Score</div>
                <div class="metric-value">{m['r2']:.4f}</div>
                <div class="metric-sub">Variance Explained</div>
            </div>""", unsafe_allow_html=True)
        with r1c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MAE</div>
                <div class="metric-value">\u20b9{m['mae']:,.0f}</div>
                <div class="metric-sub">Avg Prediction Error</div>
            </div>""", unsafe_allow_html=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RMSE</div>
                <div class="metric-value">\u20b9{m['rmse']:,.0f}</div>
                <div class="metric-sub">Root Mean Sq Error</div>
            </div>""", unsafe_allow_html=True)
        with r2c2:
            cv_r2 = metrics_data["models"][best]["cv_r2_mean"]
            st.markdown(f"""
            <div class="metric-card metric-good">
                <div class="metric-label">CV R² (5-Fold)</div>
                <div class="metric-value">{cv_r2:.4f}</div>
                <div class="metric-sub">Cross-Validated</div>
            </div>""", unsafe_allow_html=True)

# ============================================================
# TAB 2: Performance Metrics (ALL metrics for the best model)
# ============================================================
with tab_perf:
    st.markdown('<p class="section-header">📊 Comprehensive Performance Metrics</p>', unsafe_allow_html=True)

    best = metrics_data["best_model"]
    train_m = metrics_data["models"][best]["train"]
    test_m = metrics_data["models"][best]["test"]

    st.markdown(f"**Best Model:** `{best}`  |  **Train size:** {metrics_data['dataset_info']['train_rows']}  |  **Test size:** {metrics_data['dataset_info']['test_rows']}")

    # ── All metrics in styled cards ──
    st.markdown("#### Test Set Metrics")
    row1 = st.columns(4)
    row2 = st.columns(4)
    row3 = st.columns(4)

    metric_list = [
        ("R² Score", f"{test_m['r2']:.4f}", "1.0 = perfect", "metric-good" if test_m['r2'] > 0.9 else ""),
        ("Adjusted R²", f"{test_m['adjusted_r2']:.4f}", "Penalizes extra features", "metric-good" if test_m['adjusted_r2'] > 0.9 else ""),
        ("Explained Variance", f"{test_m['explained_variance']:.4f}", "Proportion explained", "metric-good" if test_m['explained_variance'] > 0.9 else ""),
        ("MAPE", f"{test_m['mape']:.2f}%", "Mean Abs % Error", "metric-good" if test_m['mape'] < 10 else "metric-warn" if test_m['mape'] > 25 else ""),
        ("MAE", f"\u20b9{test_m['mae']:,.2f}", "Mean Absolute Error", ""),
        ("MSE", f"{test_m['mse']:,.0f}", "Mean Squared Error", ""),
        ("RMSE", f"\u20b9{test_m['rmse']:,.2f}", "Root Mean Sq Error", ""),
        ("Median AE", f"\u20b9{test_m['median_ae']:,.2f}", "Median Abs Error", ""),
        ("Max Error", f"\u20b9{test_m['max_error']:,.2f}", "Worst prediction", "metric-warn"),
        ("Residual Mean", f"\u20b9{test_m['residual_mean']:,.2f}", "Bias indicator", "metric-good" if abs(test_m['residual_mean']) < 500 else ""),
        ("Residual Std", f"\u20b9{test_m['residual_std']:,.2f}", "Prediction spread", ""),
    ]

    all_card_cols = row1 + row2 + row3
    for i, (label, value, sub, css_class) in enumerate(metric_list):
        if i < len(all_card_cols):
            with all_card_cols[i]:
                st.markdown(f"""
                <div class="metric-card {css_class}">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

    # ── Train vs Test comparison ──
    st.markdown("---")
    st.markdown("#### Train vs Test — Overfitting Check")

    compare_metrics = ["r2", "mae", "rmse", "mape", "median_ae", "explained_variance"]
    compare_labels = ["R²", "MAE ($)", "RMSE ($)", "MAPE (%)", "Median AE ($)", "Expl. Variance"]

    train_vals = [train_m[k] for k in compare_metrics]
    test_vals = [test_m[k] for k in compare_metrics]

    fig_overfit = go.Figure()
    fig_overfit.add_trace(go.Bar(
        name="Train", x=compare_labels, y=train_vals,
        marker_color=COLORS["primary"], opacity=0.85,
        text=[f"{v:.4f}" if v < 10 else f"{v:,.0f}" for v in train_vals],
        textposition="outside",
    ))
    fig_overfit.add_trace(go.Bar(
        name="Test", x=compare_labels, y=test_vals,
        marker_color=COLORS["secondary"], opacity=0.85,
        text=[f"{v:.4f}" if v < 10 else f"{v:,.0f}" for v in test_vals],
        textposition="outside",
    ))
    fig_overfit.update_layout(
        **styled_layout(height=380),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Value",
    )
    st.plotly_chart(fig_overfit, use_container_width=True)

    gap = train_m["r2"] - test_m["r2"]
    if gap < 0.02:
        st.success(f"✅ R² gap = {gap:.4f} — No significant overfitting detected.")
    elif gap < 0.05:
        st.warning(f"⚠️ R² gap = {gap:.4f} — Slight overfitting. Consider regularization.")
    else:
        st.error(f"🔴 R² gap = {gap:.4f} — Significant overfitting. Model may not generalize well.")

    # ── Cross-validation detail ──
    st.markdown("---")
    st.markdown("#### 5-Fold Cross-Validation Scores")

    cv_data = metrics_data["models"][best]
    cv_cols = st.columns(3)

    cv_metrics_display = [
        ("R²", cv_data["cv_r2_scores"], cv_data["cv_r2_mean"], cv_data["cv_r2_std"]),
        ("MAE ($)", cv_data["cv_mae_scores"], cv_data["cv_mae_mean"], cv_data["cv_mae_std"]),
        ("RMSE ($)", cv_data["cv_rmse_scores"], cv_data["cv_rmse_mean"], cv_data["cv_rmse_std"]),
    ]

    for col, (name, scores, mean, std) in zip(cv_cols, cv_metrics_display):
        with col:
            fig_cv = go.Figure()
            fold_labels = [f"Fold {i+1}" for i in range(len(scores))]
            fig_cv.add_trace(go.Bar(
                x=fold_labels, y=scores,
                marker_color=[COLORS["primary"]] * len(scores),
                text=[f"{s:.4f}" if s < 10 else f"\u20b9{s:,.0f}" for s in scores],
                textposition="outside",
            ))
            fig_cv.add_hline(
                y=mean, line_dash="dash", line_color=COLORS["warning"],
                annotation_text=f"Mean: {mean:.4f}" if mean < 10 else f"Mean: \u20b9{mean:,.0f}",
            )
            fig_cv.update_layout(**styled_layout(title=f"CV {name}", height=280))
            st.plotly_chart(fig_cv, use_container_width=True)
            st.caption(f"Mean: {mean:.4f} | Std: {std:.4f}" if mean < 10 else f"Mean: \u20b9{mean:,.0f} | Std: \u20b9{std:,.0f}")

# ============================================================
# TAB 3: Model Comparison
# ============================================================
with tab_compare:
    st.markdown('<p class="section-header">🏆 Multi-Model Comparison</p>', unsafe_allow_html=True)

    model_names = list(metrics_data["models"].keys())

    # Comparison table
    rows = []
    for mn in model_names:
        t = metrics_data["models"][mn]["test"]
        cv = metrics_data["models"][mn]
        rows.append({
            "Model": mn,
            "R²": t["r2"],
            "Adj R²": t["adjusted_r2"],
            "MAE (\u20b9)": t["mae"],
            "RMSE (\u20b9)": t["rmse"],
            "MAPE (%)": t["mape"],
            "Median AE (\u20b9)": t["median_ae"],
            "Max Error (\u20b9)": t["max_error"],
            "CV R² Mean": cv["cv_r2_mean"],
            "CV R² Std": cv["cv_r2_std"],
        })

    comp_df = pd.DataFrame(rows)
    st.dataframe(
        comp_df.style.highlight_max(subset=["R²", "Adj R²", "CV R² Mean"], color="#c3f5e5")
                     .highlight_min(subset=["MAE ($)", "RMSE ($)", "MAPE (%)", "Max Error ($)"], color="#c3f5e5")
                     .format({
                         "R²": "{:.4f}", "Adj R²": "{:.4f}",
                         "MAE ($)": "${:,.2f}", "RMSE ($)": "${:,.2f}",
                         "MAPE (%)": "{:.2f}%", "Median AE ($)": "${:,.2f}",
                         "Max Error ($)": "${:,.2f}",
                         "CV R² Mean": "{:.4f}", "CV R² Std": "{:.4f}",
                     }),
        use_container_width=True,
        hide_index=True,
    )

    # Radar chart
    st.markdown("#### Model Radar Comparison")
    categories = ["R²", "1-MAPE", "Expl. Var", "CV R²", "1-NormMAE"]

    fig_radar = go.Figure()
    for mn in model_names:
        t = metrics_data["models"][mn]["test"]
        cv = metrics_data["models"][mn]
        # Normalize metrics to [0,1] range for radar
        max_mae = max(metrics_data["models"][m]["test"]["mae"] for m in model_names)
        values = [
            t["r2"],
            max(0, 1 - t["mape"] / 100),
            t["explained_variance"],
            cv["cv_r2_mean"],
            max(0, 1 - t["mae"] / max_mae) if max_mae > 0 else 0,
        ]
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=mn,
            line_color=MODEL_COLORS.get(mn, "#999"),
            opacity=0.7,
        ))

    fig_radar.update_layout(
        **styled_layout(height=450),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Side-by-side bar charts for key metrics
    st.markdown("#### Key Metrics Breakdown")
    fig_bars = make_subplots(rows=1, cols=3, subplot_titles=["R² Score", "MAE ($)", "RMSE ($)"])

    for i, metric_key in enumerate(["r2", "mae", "rmse"]):
        vals = [metrics_data["models"][mn]["test"][metric_key] for mn in model_names]
        colors = [MODEL_COLORS.get(mn, "#999") for mn in model_names]
        fmt = [f"{v:.4f}" for v in vals] if metric_key == "r2" else [f"\u20b9{v:,.0f}" for v in vals]
        fig_bars.add_trace(go.Bar(
            x=model_names, y=vals, marker_color=colors,
            text=fmt, textposition="outside", showlegend=False,
        ), row=1, col=i + 1)

    fig_bars.update_layout(**styled_layout(height=350))
    st.plotly_chart(fig_bars, use_container_width=True)

# ============================================================
# TAB 4: Residual Analysis
# ============================================================
with tab_residuals:
    st.markdown('<p class="section-header">📉 Residual Analysis</p>', unsafe_allow_html=True)

    if pred_df is not None:
        r_col1, r_col2 = st.columns(2)

        # 4a. Actual vs Predicted scatter
        with r_col1:
            fig_avp = go.Figure()
            fig_avp.add_trace(go.Scatter(
                x=pred_df["actual"], y=pred_df["predicted"],
                mode="markers",
                marker=dict(
                    color=pred_df["residual"].abs(),
                    colorscale="Tealgrn_r",
                    size=7, opacity=0.7, line=dict(width=0.5, color="white"),
                    colorbar=dict(title="| Error |"),
                ),
                hovertemplate="Actual: \u20b9%{x:,.0f}<br>Predicted: \u20b9%{y:,.0f}<extra></extra>",
            ))
            min_val = min(pred_df["actual"].min(), pred_df["predicted"].min())
            max_val = max(pred_df["actual"].max(), pred_df["predicted"].max())
            fig_avp.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode="lines", line=dict(dash="dash", color="#e74c3c", width=2),
                name="Perfect Prediction",
            ))
            fig_avp.update_layout(
                **styled_layout(title="Actual vs Predicted", height=420),
                xaxis_title="Actual Charges (\u20b9)",
                yaxis_title="Predicted Charges (\u20b9)",
                showlegend=False,
            )
            st.plotly_chart(fig_avp, use_container_width=True)

        # 4b. Residuals vs Predicted
        with r_col2:
            fig_rvp = go.Figure()
            fig_rvp.add_trace(go.Scatter(
                x=pred_df["predicted"], y=pred_df["residual"],
                mode="markers",
                marker=dict(
                    color=pred_df["residual"],
                    colorscale="RdYlGn",
                    size=7, opacity=0.7, line=dict(width=0.5, color="white"),
                    cmid=0,
                ),
                hovertemplate="Predicted: \u20b9%{x:,.0f}<br>Residual: \u20b9%{y:,.0f}<extra></extra>",
            ))
            fig_rvp.add_hline(y=0, line_dash="dash", line_color="#e74c3c", line_width=2)
            fig_rvp.update_layout(
                **styled_layout(title="Residuals vs Predicted", height=420),
                xaxis_title="Predicted Charges (\u20b9)",
                yaxis_title="Residual (\u20b9)",
            )
            st.plotly_chart(fig_rvp, use_container_width=True)

        r_col3, r_col4 = st.columns(2)

        # 4c. Residual distribution
        with r_col3:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=pred_df["residual"],
                nbinsx=40,
                marker_color=COLORS["primary"],
                opacity=0.8,
                name="Residuals",
            ))
            # Overlay normal curve
            x_range = np.linspace(pred_df["residual"].min(), pred_df["residual"].max(), 200)
            mu, sigma = pred_df["residual"].mean(), pred_df["residual"].std()
            normal_curve = stats.norm.pdf(x_range, mu, sigma) * len(pred_df) * (pred_df["residual"].max() - pred_df["residual"].min()) / 40
            fig_hist.add_trace(go.Scatter(
                x=x_range, y=normal_curve,
                mode="lines", line=dict(color=COLORS["warning"], width=2, dash="dot"),
                name="Normal Fit",
            ))
            fig_hist.update_layout(
                **styled_layout(title="Residual Distribution", height=380),
                xaxis_title="Residual (\u20b9)",
                yaxis_title="Count",
                bargap=0.05,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # 4d. QQ-style plot
        with r_col4:
            sorted_res = np.sort(pred_df["residual"].values)
            theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_res)))

            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=theoretical_q, y=sorted_res,
                mode="markers",
                marker=dict(color=COLORS["primary"], size=5, opacity=0.6),
                name="Sample Quantiles",
            ))
            # Reference line
            slope, intercept = np.polyfit(theoretical_q, sorted_res, 1)
            fig_qq.add_trace(go.Scatter(
                x=[theoretical_q.min(), theoretical_q.max()],
                y=[slope * theoretical_q.min() + intercept, slope * theoretical_q.max() + intercept],
                mode="lines", line=dict(color=COLORS["warning"], dash="dash", width=2),
                name="Normal Reference",
            ))
            fig_qq.update_layout(
                **styled_layout(title="Q-Q Plot (Normality Check)", height=380),
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles (\u20b9)",
            )
            st.plotly_chart(fig_qq, use_container_width=True)

        # 4e. Residual statistics
        st.markdown("#### 📐 Residual Statistics")
        res = pred_df["residual"]
        stat_cols = st.columns(5)
        res_stats = [
            ("Mean", f"\u20b9{res.mean():,.2f}"),
            ("Std Dev", f"\u20b9{res.std():,.2f}"),
            ("Skewness", f"{res.skew():.3f}"),
            ("Kurtosis", f"{res.kurtosis():.3f}"),
            ("Shapiro-Wilk p", f"{stats.shapiro(res.sample(min(500, len(res)), random_state=42))[1]:.4f}"),
        ]
        for col, (label, val) in zip(stat_cols, res_stats):
            col.metric(label, val)

    else:
        st.warning("No test predictions found. Re-run `python train.py`.")

# ============================================================
# TAB 5: Feature Insights
# ============================================================
with tab_features:
    st.markdown('<p class="section-header">🔍 Feature Importance & Insights</p>', unsafe_allow_html=True)

    fi_col1, fi_col2 = st.columns([3, 2])

    with fi_col1:
        # Side-by-side importances from RF and GBR
        all_imp = metrics_data.get("all_importances", {})
        feature_names = metrics_data["feature_names"]

        if all_imp:
            fig_fi = go.Figure()
            for m_name, imp_dict in all_imp.items():
                sorted_features = sorted(imp_dict.items(), key=lambda x: x[1])
                fig_fi.add_trace(go.Bar(
                    y=[f[0] for f in sorted_features],
                    x=[f[1] for f in sorted_features],
                    orientation="h",
                    name=m_name,
                    marker_color=MODEL_COLORS.get(m_name, "#999"),
                    opacity=0.85,
                ))

            fig_fi.update_layout(
                **styled_layout(title="Feature Importance Comparison", height=480),
                barmode="group",
                xaxis_title="Importance",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            # Fallback: single model importances
            imp = metrics_data["feature_importances"]
            sorted_imp = sorted(imp.items(), key=lambda x: x[1])
            fig_fi = go.Figure(go.Bar(
                y=[f[0] for f in sorted_imp],
                x=[f[1] for f in sorted_imp],
                orientation="h",
                marker=dict(
                    color=[f[1] for f in sorted_imp],
                    colorscale=[[0, COLORS["accent"]], [1, COLORS["primary"]]],
                ),
                text=[f"{f[1]:.3f}" for f in sorted_imp],
                textposition="outside",
            ))
            fig_fi.update_layout(**styled_layout(title="Feature Importance", height=450))
            st.plotly_chart(fig_fi, use_container_width=True)

    with fi_col2:
        # Importance breakdown as percentages
        st.markdown("#### 📋 Importance Ranking")
        imp = metrics_data["feature_importances"]
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        total = sum(v for _, v in sorted_imp)

        for rank, (name, val) in enumerate(sorted_imp, 1):
            pct = (val / total * 100) if total > 0 else 0
            bar_width = int(pct * 2.5)
            st.markdown(
                f"**{rank}. {name}** — {pct:.1f}%\n\n"
                f"`{'█' * bar_width}{'░' * (100 - bar_width * 4 // 10)}`"
            )

    # Correlation heatmap
    if dataset is not None:
        st.markdown("---")
        st.markdown("#### 🌡️ Feature Correlation Heatmap")

        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
        corr = dataset[numeric_cols].corr()

        fig_heatmap = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="Tealgrn",
            text=np.round(corr.values, 3),
            texttemplate="%{text}",
            textfont=dict(size=12),
            zmin=-1, zmax=1,
            colorbar=dict(title="Corr"),
        ))
        fig_heatmap.update_layout(**styled_layout(title="", height=400))
        st.plotly_chart(fig_heatmap, use_container_width=True)

# ============================================================
# TAB 6: Learning Curves
# ============================================================
with tab_learning:
    st.markdown('<p class="section-header">📈 Learning Curves</p>', unsafe_allow_html=True)
    st.markdown("Learning curves show how model performance changes with training set size — useful for diagnosing bias vs variance.")

    lc = metrics_data.get("learning_curves")
    if lc:
        train_sizes = lc["train_sizes"]
        train_mean = np.array(lc["train_scores_mean"])
        train_std = np.array(lc["train_scores_std"])
        val_mean = np.array(lc["val_scores_mean"])
        val_std = np.array(lc["val_scores_std"])

        fig_lc = go.Figure()

        # Training score band
        fig_lc.add_trace(go.Scatter(
            x=train_sizes + train_sizes[::-1],
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]).tolist(),
            fill="toself", fillcolor="rgba(17,153,142,0.15)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        fig_lc.add_trace(go.Scatter(
            x=train_sizes, y=train_mean.tolist(),
            mode="lines+markers", name="Training Score",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=6),
        ))

        # Validation score band
        fig_lc.add_trace(go.Scatter(
            x=train_sizes + train_sizes[::-1],
            y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]).tolist(),
            fill="toself", fillcolor="rgba(108,92,231,0.15)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        fig_lc.add_trace(go.Scatter(
            x=train_sizes, y=val_mean.tolist(),
            mode="lines+markers", name="Validation Score",
            line=dict(color="#6c5ce7", width=3),
            marker=dict(size=6),
        ))

        fig_lc.update_layout(
            **styled_layout(height=450),
            xaxis_title="Training Set Size",
            yaxis_title="R² Score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_lc, use_container_width=True)

        # Interpretation
        gap = train_mean[-1] - val_mean[-1]
        if gap < 0.02:
            st.success("✅ Training and validation curves converge — good bias-variance balance.")
        elif gap < 0.05:
            st.info("ℹ️ Small gap — model has slight variance. More data might help slightly.")
        else:
            st.warning("⚠️ Notable gap — model may benefit from regularization or more training data.")
    else:
        st.warning("Learning curve data not found. Re-run `python train.py`.")

# ============================================================
# TAB 7: Sensitivity Analysis
# ============================================================
with tab_sensitivity:
    st.markdown('<p class="section-header">🔬 Sensitivity Analysis</p>', unsafe_allow_html=True)
    st.markdown("Vary **one factor at a time**, holding everything else at your current profile values.")

    def _base():
        return {"age": age, "bmi": bmi, "children": children, "sex": sex, "smoker": smoker, "region": region}

    s_col1, s_col2 = st.columns(2)

    # Age sweep
    with s_col1:
        ages = list(range(18, 81))
        preds = model.predict(pd.DataFrame([dict(_base(), age=a) for a in ages]))
        fig = go.Figure(go.Scatter(
            x=ages, y=preds, mode="lines",
            line=dict(color=COLORS["primary"], width=3),
            fill="tozeroy", fillcolor="rgba(17,153,142,0.1)",
        ))
        fig.update_layout(
            **styled_layout(title="Cost vs Age", height=340),
            xaxis_title="Age", yaxis_title="Predicted Charges (\u20b9)",
            yaxis=dict(tickprefix="\u20b9", tickformat=",", gridcolor=COLORS["grid"]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # BMI sweep
    with s_col2:
        bmis = [round(b, 1) for b in np.arange(15, 50.5, 0.5)]
        preds = model.predict(pd.DataFrame([dict(_base(), bmi=b) for b in bmis]))
        fig = go.Figure(go.Scatter(
            x=bmis, y=preds, mode="lines",
            line=dict(color=COLORS["secondary"], width=3),
            fill="tozeroy", fillcolor="rgba(44,83,100,0.1)",
        ))
        fig.update_layout(
            **styled_layout(title="Cost vs BMI", height=340),
            xaxis_title="BMI", yaxis_title="Predicted Charges (\u20b9)",
            yaxis=dict(tickprefix="\u20b9", tickformat=",", gridcolor=COLORS["grid"]),
        )
        st.plotly_chart(fig, use_container_width=True)

    s_col3, s_col4 = st.columns(2)

    # Children sweep
    with s_col3:
        ch_list = list(range(6))
        preds = model.predict(pd.DataFrame([dict(_base(), children=c) for c in ch_list]))
        fig = go.Figure(go.Bar(
            x=[str(c) for c in ch_list], y=preds,
            marker_color=[COLORS["primary"], "#1abc9c", "#16a085", "#27ae60", "#2ecc71", COLORS["accent"]],
            text=[f"\u20b9{p:,.0f}" for p in preds], textposition="outside",
        ))
        fig.update_layout(
            **styled_layout(title="Cost vs Children", height=340),
            xaxis_title="Number of Children", yaxis_title="Predicted Charges (\u20b9)",
            yaxis=dict(tickprefix="\u20b9", tickformat=",", gridcolor=COLORS["grid"]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Smoker impact
    with s_col4:
        pred_ns = model.predict(pd.DataFrame([dict(_base(), smoker="no")]))[0]
        pred_s = model.predict(pd.DataFrame([dict(_base(), smoker="yes")]))[0]
        fig = go.Figure(go.Bar(
            x=["Non-Smoker", "Smoker"], y=[pred_ns, pred_s],
            marker_color=[COLORS["accent"], COLORS["warning"]],
            text=[f"\u20b9{pred_ns:,.0f}", f"\u20b9{pred_s:,.0f}"],
            textposition="outside",
            width=0.5,
        ))
        fig.update_layout(
            **styled_layout(title="Smoker Impact", height=340),
            yaxis=dict(tickprefix="\u20b9", tickformat=",", range=[0, max(pred_ns, pred_s) * 1.25], gridcolor=COLORS["grid"]),
        )
        st.plotly_chart(fig, use_container_width=True)

        diff = pred_s - pred_ns
        st.metric("Cost Difference", f"\u20b9{diff:,.2f}", f"+{diff / max(pred_ns, 1) * 100:.0f}%", delta_color="inverse")

    # Region comparison
    st.markdown("---")
    regions = ["north", "south", "east", "west", "central", "northeast"]
    preds_region = model.predict(pd.DataFrame([dict(_base(), region=r) for r in regions]))
    fig_reg = go.Figure(go.Bar(
        x=regions, y=preds_region,
        marker_color=[COLORS["primary"], COLORS["secondary"], "#6c5ce7", "#e17055"],
        text=[f"\u20b9{p:,.0f}" for p in preds_region], textposition="outside",
    ))
    fig_reg.update_layout(
        **styled_layout(title="Cost by Region", height=340),
        xaxis_title="Region", yaxis_title="Predicted Charges (\u20b9)",
        yaxis=dict(tickprefix="\u20b9", tickformat=",", gridcolor=COLORS["grid"]),
    )
    st.plotly_chart(fig_reg, use_container_width=True)

# ============================================================
# TAB 8: Data Explorer
# ============================================================
with tab_data:
    st.markdown('<p class="section-header">📋 Data Explorer</p>', unsafe_allow_html=True)

    if dataset is not None:
        d_col1, d_col2 = st.columns([1, 1])

        with d_col1:
            st.markdown("#### Dataset Preview")
            st.dataframe(dataset.head(50), use_container_width=True, height=350)

        with d_col2:
            st.markdown("#### Descriptive Statistics")
            st.dataframe(dataset.describe().round(2), use_container_width=True, height=350)

        st.markdown("---")
        st.markdown("#### Distribution Plots")

        dist_cols = st.columns(3)
        for i, col_name in enumerate(["age", "bmi", "charges"]):
            with dist_cols[i]:
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=dataset[col_name], nbinsx=35,
                    marker_color=COLORS["primary"], opacity=0.8,
                ))
                fig_dist.update_layout(**styled_layout(title=f"{col_name.title()} Distribution", height=300))
                st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Categorical Breakdowns")
        cat_cols = st.columns(3)
        for i, col_name in enumerate(["sex", "smoker", "region"]):
            with cat_cols[i]:
                vc = dataset[col_name].value_counts()
                fig_pie = go.Figure(go.Pie(
                    labels=vc.index.tolist(), values=vc.values.tolist(),
                    hole=0.45,
                    marker=dict(colors=px.colors.qualitative.Pastel),
                    textinfo="label+percent",
                ))
                fig_pie.update_layout(**styled_layout(title=f"{col_name.title()}", height=300))
                st.plotly_chart(fig_pie, use_container_width=True)

        # Charges by category box plots
        st.markdown("---")
        st.markdown("#### Charges by Category")
        box_cols = st.columns(3)
        for i, col_name in enumerate(["sex", "smoker", "region"]):
            with box_cols[i]:
                fig_box = px.box(
                    dataset, x=col_name, y="charges",
                    color=col_name,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_box.update_layout(**styled_layout(title=f"Charges by {col_name.title()}", height=350))
                st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("Dataset `insurance.csv` not found.")

# ============================================================
# Footer
# ============================================================
st.markdown("""
<div class="footer">
    Built with Streamlit &bull; scikit-learn &bull; Plotly &bull;
    Models: Random Forest &bull; Gradient Boosting &bull; Linear Regression
</div>
""", unsafe_allow_html=True)
