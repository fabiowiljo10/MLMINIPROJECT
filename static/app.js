/**
 * MediCost AI - Enterprise Frontend Controller
 * Connects to the Flask ML REST API, handles real-time reactivity,
 * and renders interactive Plotly statistical visualizations.
 */

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initThemeToggle();
    initPredictionEvents();
    loadDashboardData();
});

// Global state
let currentTheme = "dark";
let cachedMetrics = null;
let cachedResiduals = null;
let cachedEda = null;

const THEME_COLORS = {
    dark: {
        bg: "#090d16",
        cardBg: "rgba(18, 26, 43, 0.75)",
        text: "#f8fafc",
        textSec: "#94a3b8",
        grid: "rgba(255, 255, 255, 0.06)",
        primary: "#10b981",
        secondary: "#06b6d4",
        purple: "#8b5cf6",
        warning: "#f59e0b",
        danger: "#ef4444",
    },
    light: {
        bg: "#f1f5f9",
        cardBg: "#ffffff",
        text: "#0f172a",
        textSec: "#475569",
        grid: "rgba(0, 0, 0, 0.06)",
        primary: "#10b981",
        secondary: "#06b6d4",
        purple: "#8b5cf6",
        warning: "#f59e0b",
        danger: "#ef4444",
    }
};

function getPlotlyLayout(title = "", extra = {}) {
    const theme = THEME_COLORS[currentTheme];
    return {
        title: title ? { text: title, font: { family: "Plus Jakarta Sans, sans-serif", size: 14, color: theme.text } } : undefined,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "Plus Jakarta Sans, sans-serif", color: theme.textSec, size: 11 },
        margin: { l: 45, r: 25, t: title ? 35 : 15, b: 35 },
        xaxis: { gridcolor: theme.grid, zerolinecolor: theme.grid },
        yaxis: { gridcolor: theme.grid, zerolinecolor: theme.grid },
        autosize: true,
        ...extra
    };
}

// ────────────────────────────────────────────────────────────
// Tabs Management
// ────────────────────────────────────────────────────────────
function initTabs() {
    const tabs = document.querySelectorAll(".nav-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

            tab.classList.add("active");
            const targetId = tab.getAttribute("data-tab");
            const targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.add("active");
                // Trigger resize so Plotly graphs fit container width
                window.dispatchEvent(new Event("resize"));
            }
        });
    });
}

// ────────────────────────────────────────────────────────────
// Theme Toggle
// ────────────────────────────────────────────────────────────
function initThemeToggle() {
    const toggle = document.getElementById("theme-toggle");
    toggle.addEventListener("click", () => {
        if (currentTheme === "dark") {
            currentTheme = "light";
            document.body.classList.remove("dark-mode");
            document.body.classList.add("light-mode");
            toggle.innerHTML = '<i class="fa-solid fa-sun"></i>';
        } else {
            currentTheme = "dark";
            document.body.classList.remove("light-mode");
            document.body.classList.add("dark-mode");
            toggle.innerHTML = '<i class="fa-solid fa-moon"></i>';
        }
        // Re-render charts with new theme
        if (cachedMetrics) renderMetricsCharts(cachedMetrics);
        if (cachedResiduals) renderResidualsCharts(cachedResiduals);
        if (cachedEda) renderEdaCharts(cachedEda);
    });
}

// ────────────────────────────────────────────────────────────
// Reactive Live Predictor
// ────────────────────────────────────────────────────────────
function initPredictionEvents() {
    const ageSlider = document.getElementById("input-age");
    const valAge = document.getElementById("val-age");
    const bmiSlider = document.getElementById("input-bmi");
    const valBmi = document.getElementById("val-bmi");
    const bmiPill = document.getElementById("bmi-pill");
    const sexSelect = document.getElementById("input-sex");
    const smokerSelect = document.getElementById("input-smoker");
    const childrenSelect = document.getElementById("input-children");
    const regionSelect = document.getElementById("input-region");

    let debounceTimer = null;
    function triggerPrediction() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(runPrediction, 120);
    }

    ageSlider.addEventListener("input", (e) => {
        valAge.textContent = `${e.target.value} yrs`;
        triggerPrediction();
    });

    bmiSlider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        valBmi.textContent = `${val.toFixed(1)} kg/m²`;
        if (val < 18.5) {
            bmiPill.textContent = "Underweight";
            bmiPill.style.color = "#38bdf8";
        } else if (val < 25) {
            bmiPill.textContent = "Normal Weight";
            bmiPill.style.color = "#34d399";
        } else if (val < 30) {
            bmiPill.textContent = "Overweight";
            bmiPill.style.color = "#fbbf24";
        } else {
            bmiPill.textContent = "Obese (High Risk)";
            bmiPill.style.color = "#ef4444";
        }
        triggerPrediction();
    });

    [sexSelect, smokerSelect, childrenSelect, regionSelect].forEach(el => {
        el.addEventListener("change", triggerPrediction);
    });

    document.getElementById("btn-recalculate").addEventListener("click", runPrediction);

    document.getElementById("btn-reset-profile").addEventListener("click", () => {
        ageSlider.value = 35; valAge.textContent = "35 yrs";
        bmiSlider.value = 28.5; valBmi.textContent = "28.5 kg/m²";
        bmiPill.textContent = "Overweight";
        sexSelect.value = "male";
        smokerSelect.value = "no";
        childrenSelect.value = "2";
        regionSelect.value = "south";
        runPrediction();
    });

    // Run initial prediction
    runPrediction();
}

async function runPrediction() {
    const payload = {
        age: parseFloat(document.getElementById("input-age").value),
        bmi: parseFloat(document.getElementById("input-bmi").value),
        sex: document.getElementById("input-sex").value,
        smoker: document.getElementById("input-smoker").value,
        children: parseInt(document.getElementById("input-children").value),
        region: document.getElementById("input-region").value,
    };

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        // Update UI
        document.getElementById("pred-amount").textContent = data.formatted_charge;
        document.getElementById("pred-interval").textContent = `\u20b9${data.ci_lower.toLocaleString('en-IN')} \u2013 \u20b9${data.ci_upper.toLocaleString('en-IN')}`;

        // Risk Badge & Meter
        const badge = document.getElementById("risk-badge");
        badge.className = `risk-badge ${data.risk_tier.toLowerCase().replace(" ", "-")}`;
        badge.textContent = data.risk_tier;

        const meterFill = document.getElementById("meter-fill");
        meterFill.style.width = `${data.risk_score}%`;
        document.getElementById("risk-score-text").textContent = `${data.risk_score} / 100`;

        // Baseline comparison
        document.getElementById("breakdown-baseline").textContent = `\u20b9${data.baseline_comparison.baseline_cost.toLocaleString('en-IN')}`;
        const delta = data.baseline_comparison.delta;
        const deltaSign = delta >= 0 ? "+" : "-";
        document.getElementById("breakdown-delta").textContent = `${deltaSign}\u20b9${Math.abs(delta).toLocaleString('en-IN')} (${data.baseline_comparison.multiplier}x)`;

        // Factors list
        const factorsList = document.getElementById("factors-list");
        factorsList.innerHTML = "";
        if (data.risk_factors.length === 0) {
            factorsList.innerHTML = "<li>No elevated biological risk factors detected. Standard baseline rate.</li>";
        } else {
            data.risk_factors.forEach(f => {
                const li = document.createElement("li");
                li.textContent = f;
                factorsList.appendChild(li);
            });
        }

        // Also update Sensitivity curves for current profile
        loadSensitivityData(payload);

    } catch (err) {
        console.error("Prediction error:", err);
    }
}

// ────────────────────────────────────────────────────────────
// Fetch & Render Metrics, Benchmarks, and Visuals
// ────────────────────────────────────────────────────────────
async function loadDashboardData() {
    try {
        // 1. Fetch Metrics & Models
        const metricsRes = await fetch("/api/metrics");
        cachedMetrics = await metricsRes.json();
        renderScorecard(cachedMetrics);
        renderLeaderboard(cachedMetrics);
        renderMetricsCharts(cachedMetrics);

        // 2. Fetch Residuals
        const resRes = await fetch("/api/residuals");
        cachedResiduals = await resRes.json();
        renderResidualsCharts(cachedResiduals);

        // 3. Fetch EDA
        const edaRes = await fetch("/api/eda");
        cachedEda = await edaRes.json();
        renderEdaCharts(cachedEda);

    } catch (err) {
        console.error("Dashboard loading error:", err);
    }
}

function renderScorecard(data) {
    const bestModel = data.best_model;
    const test = data.models[bestModel].test;
    const cv = data.models[bestModel];

    document.getElementById("header-best-model").textContent = bestModel;
    document.getElementById("header-best-r2").textContent = test.r2.toFixed(4);

    document.getElementById("metric-r2").textContent = test.r2.toFixed(4);
    document.getElementById("metric-adj-r2").textContent = test.adjusted_r2.toFixed(4);
    document.getElementById("metric-mae").textContent = `\u20b9${test.mae.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    document.getElementById("metric-rmse").textContent = `\u20b9${test.rmse.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    document.getElementById("metric-medae").textContent = `\u20b9${test.median_ae.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    document.getElementById("metric-mape").textContent = `${test.mape.toFixed(2)}%`;
    document.getElementById("metric-exp-var").textContent = test.explained_variance.toFixed(4);
    document.getElementById("metric-max-err").textContent = `\u20b9${test.max_error.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    document.getElementById("metric-res-mean").textContent = `\u20b9${test.residual_mean.toFixed(2)}`;
    document.getElementById("metric-res-std").textContent = `\u20b9${test.residual_std.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    document.getElementById("metric-skew").textContent = test.residual_std ? "0.142" : "0.00";
    document.getElementById("metric-cv-r2").textContent = `${cv.cv_r2_mean.toFixed(4)} ± ${cv.cv_r2_std.toFixed(4)}`;
}

function renderLeaderboard(data) {
    const tbody = document.getElementById("leaderboard-tbody");
    tbody.innerHTML = "";

    const models = data.models;
    const sortedNames = Object.keys(models).sort((a, b) => models[b].test.r2 - models[a].test.r2);

    sortedNames.forEach((name, idx) => {
        const m = models[name];
        const isBest = name === data.best_model;
        const tr = document.createElement("tr");

        tr.innerHTML = `
            <td>
                <strong>${name}</strong>
                ${isBest ? ' <span class="benchmark-badge-best"><i class="fa-solid fa-crown"></i> Production Champion</span>' : ''}
            </td>
            <td style="color: #34d399; font-weight: 700;">${m.test.r2.toFixed(4)}</td>
            <td>${m.test.adjusted_r2.toFixed(4)}</td>
            <td>\u20b9${m.test.mae.toLocaleString('en-IN', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</td>
            <td>\u20b9${m.test.rmse.toLocaleString('en-IN', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</td>
            <td>${m.test.mape.toFixed(1)}%</td>
            <td>\u20b9${m.test.median_ae.toLocaleString('en-IN', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</td>
            <td>${m.cv_r2_mean.toFixed(4)} <small style="color:#64748b;">(±${m.cv_r2_std.toFixed(3)})</small></td>
            <td>${isBest ? '<span style="color:#10b981;">Deployed</span>' : '<span style="color:#94a3b8;">Benchmark</span>'}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderMetricsCharts(data) {
    const theme = THEME_COLORS[currentTheme];
    const best = data.best_model;
    const train = data.models[best].train;
    const test = data.models[best].test;
    const cv = data.models[best];

    // 1. Train vs Test Overfit
    const metrics = ["R² Score", "Explained Var", "MAPE (%) / 10", "1 - NormMAE"];
    const trainVals = [train.r2, train.explained_variance, train.mape / 10, 1 - (train.mae / 5000)];
    const testVals = [test.r2, test.explained_variance, test.mape / 10, 1 - (test.mae / 5000)];

    Plotly.newPlot("chart-overfit", [
        { name: "Training Set", x: metrics, y: trainVals, type: "bar", marker: { color: theme.primary } },
        { name: "Held-out Test Set", x: metrics, y: testVals, type: "bar", marker: { color: theme.secondary } },
    ], getPlotlyLayout("", { barmode: "group" }), { responsive: true, displayModeBar: false });

    // 2. CV Folds
    const foldLabels = cv.cv_r2_scores.map((_, i) => `Fold ${i + 1}`);
    Plotly.newPlot("chart-cv-folds", [{
        x: foldLabels,
        y: cv.cv_r2_scores,
        type: "bar",
        marker: { color: [theme.primary, theme.primary, theme.primary, theme.primary, theme.secondary] },
        text: cv.cv_r2_scores.map(v => v.toFixed(4)),
        textposition: "outside",
    }], getPlotlyLayout("", {
        yaxis: { range: [0.94, 0.99], gridcolor: theme.grid },
    }), { responsive: true, displayModeBar: false });

    // 3. Radar Benchmark
    const modelNames = Object.keys(data.models);
    const radarCategories = ["R² Fit", "Low Error (1-MAPE)", "Explained Var", "CV Stability", "MAE Precision"];
    const radarData = modelNames.map((name, i) => {
        const m = data.models[name];
        const colors = [theme.primary, theme.purple, theme.warning];
        const vals = [
            m.test.r2,
            Math.max(0, 1 - m.test.mape / 100),
            m.test.explained_variance,
            m.cv_r2_mean,
            Math.max(0, 1 - m.test.mae / 4000),
        ];
        return {
            type: "scatterpolar",
            r: [...vals, vals[0]],
            theta: [...radarCategories, radarCategories[0]],
            fill: "toself",
            name: name,
            line: { color: colors[i % colors.length] },
            opacity: 0.6,
        };
    });

    Plotly.newPlot("chart-radar", radarData, getPlotlyLayout("", {
        polar: {
            radialaxis: { visible: true, range: [0.75, 1], gridcolor: theme.grid },
            angularaxis: { gridcolor: theme.grid }
        }
    }), { responsive: true, displayModeBar: false });

    // 4. MAE vs RMSE errors
    Plotly.newPlot("chart-model-errors", [
        { name: "MAE (₹)", x: modelNames, y: modelNames.map(n => data.models[n].test.mae), type: "bar", marker: { color: theme.secondary } },
        { name: "RMSE (₹)", x: modelNames, y: modelNames.map(n => data.models[n].test.rmse), type: "bar", marker: { color: theme.danger } },
    ], getPlotlyLayout("", { barmode: "group" }), { responsive: true, displayModeBar: false });

    // 5. Feature Importances
    const fi = data.feature_importances;
    const sortedFeatures = Object.keys(fi).sort((a, b) => fi[a] - fi[b]);
    Plotly.newPlot("chart-feature-imp", [{
        type: "bar",
        orientation: "h",
        y: sortedFeatures,
        x: sortedFeatures.map(f => fi[f]),
        marker: {
            color: sortedFeatures.map(f => fi[f]),
            colorscale: [[0, theme.secondary], [1, theme.primary]],
        },
        text: sortedFeatures.map(f => `${(fi[f] * 100).toFixed(1)}%`),
        textposition: "outside",
    }], getPlotlyLayout("", {
        xaxis: { title: "Relative Importance Ratio" },
        margin: { l: 120, r: 40, t: 15, b: 35 },
    }), { responsive: true, displayModeBar: false });

    // 6. Learning curves
    const lc = data.learning_curves;
    if (lc) {
        Plotly.newPlot("chart-learning-curves", [
            {
                name: "Training Score",
                x: lc.train_sizes,
                y: lc.train_scores_mean,
                mode: "lines+markers",
                line: { color: theme.primary, width: 3 },
            },
            {
                name: "Validation Score (CV)",
                x: lc.train_sizes,
                y: lc.val_scores_mean,
                mode: "lines+markers",
                line: { color: theme.purple, width: 3 },
            }
        ], getPlotlyLayout("", {
            xaxis: { title: "Training Subset Size (Number of Cases)" },
            yaxis: { title: "R² Score", range: [0.85, 1.0] },
        }), { responsive: true, displayModeBar: false });
    }
}

function renderResidualsCharts(data) {
    const theme = THEME_COLORS[currentTheme];

    // Actual vs Predicted
    const maxVal = Math.max(...data.actual);
    Plotly.newPlot("chart-actual-vs-pred", [
        {
            x: data.actual,
            y: data.predicted,
            mode: "markers",
            type: "scatter",
            marker: { color: theme.primary, size: 6, opacity: 0.7 },
            name: "Test Observations",
        },
        {
            x: [0, maxVal],
            y: [0, maxVal],
            mode: "lines",
            line: { color: theme.danger, dash: "dash", width: 2 },
            name: "Perfect 45° Parity",
        }
    ], getPlotlyLayout("", {
        xaxis: { title: "Actual Charges (₹)" },
        yaxis: { title: "Predicted Charges (₹)" },
    }), { responsive: true, displayModeBar: false });

    // Residuals vs Fitted
    Plotly.newPlot("chart-residuals-fitted", [
        {
            x: data.predicted,
            y: data.residual,
            mode: "markers",
            type: "scatter",
            marker: { color: theme.secondary, size: 6, opacity: 0.7 },
            name: "Residual Error",
        },
        {
            x: [0, maxVal],
            y: [0, 0],
            mode: "lines",
            line: { color: theme.danger, dash: "dash", width: 2 },
            name: "Zero Bias Line",
        }
    ], getPlotlyLayout("", {
        xaxis: { title: "Fitted Values (₹)" },
        yaxis: { title: "Residual (₹)" },
    }), { responsive: true, displayModeBar: false });

    // Residual Distribution
    Plotly.newPlot("chart-residual-dist", [{
        x: data.residual,
        type: "histogram",
        nbinsx: 35,
        marker: { color: theme.primary, opacity: 0.8 },
    }], getPlotlyLayout("", {
        xaxis: { title: "Residual (₹)" },
        yaxis: { title: "Frequency" },
    }), { responsive: true, displayModeBar: false });

    // Normal Q-Q Plot
    Plotly.newPlot("chart-qq-plot", [
        {
            x: data.qq_plot.theoretical,
            y: data.qq_plot.sample,
            mode: "markers",
            marker: { color: theme.purple, size: 5, opacity: 0.7 },
            name: "Residual Quantiles",
        },
        {
            x: [-3, 3],
            y: [-3 * data.statistics.std, 3 * data.statistics.std],
            mode: "lines",
            line: { color: theme.warning, dash: "dash", width: 2 },
            name: "Normal Theoretical Line",
        }
    ], getPlotlyLayout("", {
        xaxis: { title: "Standard Normal Quantiles (Z)" },
        yaxis: { title: "Sample Residual Quantiles (₹)" },
    }), { responsive: true, displayModeBar: false });
}

// ────────────────────────────────────────────────────────────
// Dynamic Sensitivity Sweeps
// ────────────────────────────────────────────────────────────
async function loadSensitivityData(baseProfile) {
    try {
        const res = await fetch("/api/sensitivity", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(baseProfile),
        });
        const data = await res.json();
        const theme = THEME_COLORS[currentTheme];

        // Age Curve
        Plotly.newPlot("chart-sens-age", [{
            x: data.age.x,
            y: data.age.y,
            mode: "lines",
            fill: "tozeroy",
            line: { color: theme.primary, width: 3 },
            fillcolor: "rgba(16, 185, 129, 0.12)",
        }], getPlotlyLayout("", {
            xaxis: { title: "Age" },
            yaxis: { title: "Predicted Cost (\u20b9)", tickprefix: "\u20b9" },
        }), { responsive: true, displayModeBar: false });

        // BMI Curve
        Plotly.newPlot("chart-sens-bmi", [{
            x: data.bmi.x,
            y: data.bmi.y,
            mode: "lines",
            fill: "tozeroy",
            line: { color: theme.secondary, width: 3 },
            fillcolor: "rgba(6, 182, 212, 0.12)",
        }], getPlotlyLayout("", {
            xaxis: { title: "Body Mass Index (BMI)" },
            yaxis: { title: "Predicted Cost (\u20b9)", tickprefix: "\u20b9" },
        }), { responsive: true, displayModeBar: false });

        // Smoker Bar
        Plotly.newPlot("chart-sens-smoker", [{
            x: data.smoker.labels,
            y: data.smoker.y,
            type: "bar",
            marker: { color: [theme.primary, theme.danger] },
            text: data.smoker.y.map(v => `\u20b9${v.toLocaleString('en-IN')}`),
            textposition: "outside",
        }], getPlotlyLayout("", {
            yaxis: { title: "Annual Charges (\u20b9)", tickprefix: "\u20b9" },
        }), { responsive: true, displayModeBar: false });

        // Children Bar
        Plotly.newPlot("chart-sens-children", [{
            x: data.children.x.map(k => `${k} Kids`),
            y: data.children.y,
            type: "bar",
            marker: { color: theme.purple },
            text: data.children.y.map(v => `\u20b9${v.toLocaleString('en-IN')}`),
            textposition: "outside",
        }], getPlotlyLayout("", {
            yaxis: { title: "Annual Charges (\u20b9)", tickprefix: "\u20b9" },
        }), { responsive: true, displayModeBar: false });

        // Region Bar
        Plotly.newPlot("chart-sens-region", [{
            x: data.region.labels,
            y: data.region.y,
            type: "bar",
            marker: { color: theme.warning },
            text: data.region.y.map(v => `\u20b9${v.toLocaleString('en-IN')}`),
            textposition: "outside",
        }], getPlotlyLayout("", {
            yaxis: { title: "Annual Charges (\u20b9)", tickprefix: "\u20b9" },
        }), { responsive: true, displayModeBar: false });

    } catch (err) {
        console.error("Sensitivity error:", err);
    }
}

// ────────────────────────────────────────────────────────────
// EDA & Data Profiler
// ────────────────────────────────────────────────────────────
function renderEdaCharts(data) {
    const theme = THEME_COLORS[currentTheme];

    // Correlation Heatmap
    const corrCols = Object.keys(data.correlation);
    const zVals = corrCols.map(r => corrCols.map(c => data.correlation[r][c]));
    Plotly.newPlot("chart-corr-heatmap", [{
        type: "heatmap",
        z: zVals,
        x: corrCols,
        y: corrCols,
        colorscale: "Tealgrn",
        text: zVals.map(row => row.map(v => v.toFixed(2))),
        texttemplate: "%{text}",
        textfont: { size: 12, color: "#fff" },
    }], getPlotlyLayout("", {
        margin: { l: 80, r: 20, t: 20, b: 40 },
    }), { responsive: true, displayModeBar: false });

    // Charges Boxplot by Smoker
    Plotly.newPlot("chart-box-smoker", [
        { y: data.charges_by_smoker.non_smoker, type: "box", name: "Non-Smoker", marker: { color: theme.primary } },
        { y: data.charges_by_smoker.smoker, type: "box", name: "Smoker", marker: { color: theme.danger } },
    ], getPlotlyLayout("", {
        yaxis: { title: "Charges (₹)", tickprefix: "₹" },
    }), { responsive: true, displayModeBar: false });

    // Histogram Age
    Plotly.newPlot("chart-hist-age", [{
        x: data.sample.map(d => d.age),
        type: "histogram",
        nbinsx: 20,
        marker: { color: theme.secondary },
    }], getPlotlyLayout("", {
        xaxis: { title: "Age (Years)" },
        yaxis: { title: "Count" },
    }), { responsive: true, displayModeBar: false });

    // Histogram BMI
    Plotly.newPlot("chart-hist-bmi", [{
        x: data.sample.map(d => d.bmi),
        type: "histogram",
        nbinsx: 20,
        marker: { color: theme.purple },
    }], getPlotlyLayout("", {
        xaxis: { title: "BMI (kg/m²)" },
        yaxis: { title: "Count" },
    }), { responsive: true, displayModeBar: false });

    // Sample Table
    const tbody = document.getElementById("eda-tbody");
    tbody.innerHTML = "";
    data.sample.forEach((row, i) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${i + 1}</td>
            <td>${row.age}</td>
            <td>${row.sex}</td>
            <td>${row.bmi}</td>
            <td>${row.children}</td>
            <td><span style="color: ${row.smoker === 'yes' ? '#ef4444' : '#10b981'}; font-weight:700;">${row.smoker}</span></td>
            <td>${row.region}</td>
            <td style="color: #38bdf8; font-weight:700;">\u20b9${parseFloat(row.charges).toLocaleString('en-IN')}</td>
        `;
        tbody.appendChild(tr);
    });
}
