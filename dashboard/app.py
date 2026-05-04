"""
app.py
------
SA Macroeconomic Credit Stress Monitor — Streamlit Dashboard

Four sections:
  1. Live Stress Gauge      — current regime + probability
  2. Historical Timeline    — 24 years of model predictions vs actuals
  3. Feature Intelligence   — SHAP importance + accuracy strip
  4. Scenario Scorer        — interactive form → live /predict call + waterfall

Run locally (from repo root):
    streamlit run dashboard/app.py

With a live Cloud Run API:
    API_BASE_URL=https://your-url.run.app streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Make src/ importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.api_client import (
    get_health,
    get_historical,
    get_feature_importance,
    post_predict,
)
from dashboard.charts import (
    regime_gauge,
    historical_chart,
    feature_importance_chart,
    shap_waterfall,
    confusion_strip,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SA Credit Stress Monitor",
    page_icon="🇿🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Metric cards */
  .metric-card {
    background: #F4F6FA;
    border-radius: 12px;
    padding: 20px 24px;
    border-left: 4px solid #002395;
  }
  .metric-card h3 { margin: 0 0 4px 0; font-size: 13px; color: #6B7280; font-weight: 500; }
  .metric-card p  { margin: 0; font-size: 28px; font-weight: 700; color: #111827; }

  /* Regime badge */
  .regime-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }

  /* Section headers */
  .section-header {
    font-size: 20px;
    font-weight: 700;
    color: #002395;
    margin: 8px 0 4px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #E5E7EB;
  }

  /* Driver pill */
  .driver-pill {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    margin: 2px 3px;
    font-weight: 500;
  }

  /* Hide Streamlit default branding */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Cached data fetchers ───────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_historical():
    return get_historical()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_importance():
    return get_feature_importance()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_health():
    return get_health()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Flag_of_South_Africa.svg/320px-Flag_of_South_Africa.svg.png",
        use_column_width=True,
    )
    st.markdown("## SA Credit Stress Monitor")
    st.markdown(
        "XGBoost early warning system for South African credit stress regimes. "
        "Trained on 24 years of real macro data (2000–2023).\n\n"
        "**Data sources:**\n"
        "- SARB · Stats SA\n"
        "- World Bank Open Data\n"
        "- FRED (St. Louis Fed)\n"
        "- yfinance (ZAR/USD · JSE)\n\n"
        "**Model:** XGBoost · SHAP explainability\n\n"
        "**Deployment:** GCP Cloud Run · GitHub Actions"
    )

    st.divider()

    # Model status
    try:
        health = fetch_health()
        st.success(f"✓ Model loaded  |  v{health['model_version']}")
        st.caption(f"{health['n_features']} features  ·  trained {health['train_years']}")
    except Exception as e:
        st.error(f"API error: {repr(e)}")

    st.divider()
    st.markdown(
        "Built by **Fikile Sondach**\n\n"
        "[GitHub ↗](https://github.com/Fikilesondach/sa-credit-stress-monitor)",
        unsafe_allow_html=False,
    )


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#002395; margin-bottom:4px;'>🇿🇦 SA Macroeconomic Credit Stress Monitor</h1>"
    "<p style='color:#6B7280; margin-top:0;'>XGBoost early warning system · 2000–2023 · SHAP-explained</p>",
    unsafe_allow_html=True,
)
st.divider()


# ── Load historical data ───────────────────────────────────────────────────────
with st.spinner("Loading model data... (first run may train the model — ~30s)"):
    try:
        hist_data   = fetch_historical()
        imp_data    = fetch_importance()
        hist_list   = hist_data["data"]
        latest_year = hist_list[-1]
    except Exception as e:
        st.error(f"Failed to load data: {repr(e)}")
        import traceback; st.code(traceback.format_exc()); st.info("Fix: run `PYTHONPATH=. python -m src.models.train` then `PYTHONPATH=. python -m src.models.explain` from your repo root, then refresh."); st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Current Regime
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">📡 Current Regime (Latest Available: 2023)</p>', unsafe_allow_html=True)

prob          = latest_year["stress_prob"]
actual_stress = latest_year["actual_stress"]
regime_label  = (
    "Acute Stress"    if prob >= 0.70 else
    "Elevated Stress" if prob >= 0.50 else
    "Watch"           if prob >= 0.30 else
    "Stable"
)
regime_colour = {
    "Stable": "#007A4D", "Watch": "#FFB612",
    "Elevated Stress": "#E8620A", "Acute Stress": "#DE3831"
}[regime_label]

col_gauge, col_metrics = st.columns([1, 2], gap="large")

with col_gauge:
    st.plotly_chart(
        regime_gauge(prob, regime_label, regime_colour),
        use_container_width=True, config={"displayModeBar": False}
    )

with col_metrics:
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
          <h3>Stress Probability</h3>
          <p style="color:{regime_colour}">{prob:.1%}</p>
        </div>""", unsafe_allow_html=True)

    with m2:
        total  = hist_data["total_years"]
        stress = hist_data["stress_years"]
        st.markdown(f"""
        <div class="metric-card">
          <h3>Historical Base Rate</h3>
          <p>{stress}/{total} years</p>
        </div>""", unsafe_allow_html=True)

    with m3:
        gini_val = fetch_health().get("n_features", "—")
        st.markdown(f"""
        <div class="metric-card">
          <h3>Model Features</h3>
          <p>{gini_val}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f'<span class="regime-badge" style="background:{regime_colour}22; color:{regime_colour}; border:1.5px solid {regime_colour};">'
        f"{'⚠️' if prob >= 0.5 else '✅'} {regime_label}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>**Top 3 Drivers in 2023:**", unsafe_allow_html=True)
    drivers_html = "".join(
        f'<span class="driver-pill">{d.replace("_", " ").title()}</span>'
        for d in latest_year["top_3_drivers"]
    )
    st.markdown(drivers_html, unsafe_allow_html=True)

    actual_label = "🔴 Actual: Stress Year" if actual_stress else "🟢 Actual: Stable Year"
    st.caption(actual_label)

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Historical Timeline
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">📈 Historical Stress Probability (2000–2023)</p>', unsafe_allow_html=True)

st.plotly_chart(
    historical_chart(hist_list),
    use_container_width=True,
    config={"displayModeBar": True, "displaylogo": False},
)

# Accuracy strip
st.plotly_chart(
    confusion_strip(hist_list),
    use_container_width=True,
    config={"displayModeBar": False},
)

# Quick stats table
with st.expander("📋 Full historical data table"):
    df_hist = pd.DataFrame(hist_list)
    df_hist["predicted"]    = (df_hist["stress_prob"] >= 0.5).astype(int)
    df_hist["correct"]      = (df_hist["predicted"] == df_hist["actual_stress"])
    df_hist["stress_prob"]  = df_hist["stress_prob"].map("{:.1%}".format)
    df_hist["top_3_drivers"] = df_hist["top_3_drivers"].apply(
        lambda x: " · ".join(d.replace("_", " ").title() for d in x)
    )
    df_hist.columns = ["Year", "Stress Prob", "Actual Stress", "Top 3 Drivers", "Predicted", "Correct"]
    st.dataframe(
        df_hist[["Year", "Stress Prob", "Actual Stress", "Predicted", "Correct", "Top 3 Drivers"]],
        use_container_width=True, hide_index=True,
    )

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Feature Intelligence
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">🔬 Feature Intelligence</p>', unsafe_allow_html=True)

col_imp, col_txt = st.columns([3, 1], gap="large")

with col_imp:
    st.plotly_chart(
        feature_importance_chart(imp_data["features"], top_n=12),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with col_txt:
    st.markdown("<br><br>", unsafe_allow_html=True)
    top3 = imp_data["features"][:3]
    st.markdown("**Top 3 Global Drivers:**")
    for f in top3:
        clean = f["feature"].replace("_pct_gdp", " (% GDP)").replace("_", " ").title()
        st.markdown(f"**{f['rank']}.** {clean}  \n`SHAP: {f['importance']:.4f}`")
        st.markdown("")

    st.info(
        "Importance = mean |SHAP value| across all 24 years. "
        "Higher = stronger average impact on the model's output.",
        icon="ℹ️",
    )

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Scenario Scorer
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">🧪 Scenario Scorer — Score Any Macro Environment</p>', unsafe_allow_html=True)
st.caption(
    "Enter any combination of SA macro indicators to get a real-time stress probability "
    "and SHAP waterfall explaining what's driving the prediction."
)

# Defaults = 2023 actual values
DEFAULTS = {
    "gdp_growth": 0.7,
    "cpi_inflation": 6.1,
    "unemployment_rate": 32.9,
    "repo_rate": 8.25,
    "npl_ratio": 4.1,
    "private_credit_pct_gdp": 140.2,
    "govt_debt_pct_gdp": 73.8,
    "current_account_pct_gdp": -1.6,
    "zar_usd": 18.45,
    "vix_avg": 16.9,
}

# Quick presets
preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
preset = None
with preset_col1:
    if st.button("📉 2009 GFC", use_container_width=True):
        preset = {
            "gdp_growth": -1.5, "cpi_inflation": 7.1, "unemployment_rate": 24.0,
            "repo_rate": 7.0, "npl_ratio": 5.9, "private_credit_pct_gdp": 163.2,
            "govt_debt_pct_gdp": 30.1, "current_account_pct_gdp": -4.0,
            "zar_usd": 8.47, "vix_avg": 31.5,
        }
with preset_col2:
    if st.button("💥 2020 COVID", use_container_width=True):
        preset = {
            "gdp_growth": -6.3, "cpi_inflation": 3.3, "unemployment_rate": 29.2,
            "repo_rate": 3.5, "npl_ratio": 4.9, "private_credit_pct_gdp": 141.5,
            "govt_debt_pct_gdp": 69.4, "current_account_pct_gdp": 2.0,
            "zar_usd": 16.46, "vix_avg": 29.3,
        }
with preset_col3:
    if st.button("🌱 2006 Boom", use_container_width=True):
        preset = {
            "gdp_growth": 5.6, "cpi_inflation": 4.7, "unemployment_rate": 25.5,
            "repo_rate": 9.0, "npl_ratio": 1.1, "private_credit_pct_gdp": 161.9,
            "govt_debt_pct_gdp": 27.8, "current_account_pct_gdp": -5.3,
            "zar_usd": 6.77, "vix_avg": 12.8,
        }
with preset_col4:
    if st.button("📊 2023 Current", use_container_width=True):
        preset = DEFAULTS.copy()

vals = preset if preset else DEFAULTS

st.markdown("<br>", unsafe_allow_html=True)

# Input form
with st.form("scenario_form"):
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("**Real Economy**")
        gdp_growth = st.slider(
            "GDP Growth (%)", -10.0, 10.0, float(vals["gdp_growth"]), 0.1,
            help="Real annual GDP growth rate"
        )
        unemployment_rate = st.slider(
            "Unemployment Rate (%)", 15.0, 45.0, float(vals["unemployment_rate"]), 0.1,
            help="% of total labour force (Stats SA QLFS)"
        )
        cpi_inflation = st.slider(
            "CPI Inflation (%)", 0.0, 20.0, float(vals["cpi_inflation"]), 0.1,
            help="Annual CPI inflation rate"
        )
        repo_rate = st.slider(
            "SARB Repo Rate (%)", 3.0, 15.0, float(vals["repo_rate"]), 0.25,
            help="SARB repo rate at year-end"
        )
        vix_avg = st.slider(
            "VIX (annual avg)", 5.0, 60.0, float(vals["vix_avg"]), 0.5,
            help="CBOE VIX — global risk/fear index"
        )

    with col_b:
        st.markdown("**Banking & Fiscal**")
        npl_ratio = st.slider(
            "NPL Ratio (%)", 0.5, 12.0, float(vals["npl_ratio"]), 0.1,
            help="Non-performing loans as % of gross loans"
        )
        private_credit_pct_gdp = st.slider(
            "Private Credit (% GDP)", 80.0, 200.0, float(vals["private_credit_pct_gdp"]), 0.5,
            help="Domestic credit to private sector"
        )
        govt_debt_pct_gdp = st.slider(
            "Govt Debt (% GDP)", 20.0, 100.0, float(vals["govt_debt_pct_gdp"]), 0.5,
            help="Central government debt as % of GDP"
        )
        current_account_pct_gdp = st.slider(
            "Current Account (% GDP)", -12.0, 8.0, float(vals["current_account_pct_gdp"]), 0.1,
            help="Negative = deficit"
        )
        zar_usd = st.slider(
            "ZAR/USD Rate", 4.0, 25.0, float(vals["zar_usd"]), 0.05,
            help="Annual average ZAR per USD"
        )

    submitted = st.form_submit_button(
        "🔮 Score This Scenario", use_container_width=True, type="primary"
    )


# ── Prediction output ──────────────────────────────────────────────────────────
if submitted:
    payload = {
        "gdp_growth":              gdp_growth,
        "cpi_inflation":           cpi_inflation,
        "unemployment_rate":       unemployment_rate,
        "repo_rate":               repo_rate,
        "npl_ratio":               npl_ratio,
        "private_credit_pct_gdp":  private_credit_pct_gdp,
        "govt_debt_pct_gdp":       govt_debt_pct_gdp,
        "current_account_pct_gdp": current_account_pct_gdp,
        "zar_usd":                 zar_usd,
        "vix_avg":                 vix_avg,
    }

    with st.spinner("Scoring scenario..."):
        try:
            result = post_predict(payload)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    pred_prob   = result["stress_probability"]
    pred_regime = result["regime"]
    pred_colour = pred_regime["colour"]

    st.markdown("<br>", unsafe_allow_html=True)

    # Result header
    res_col1, res_col2 = st.columns([1, 2], gap="large")

    with res_col1:
        st.plotly_chart(
            regime_gauge(pred_prob, pred_regime["label"], pred_colour),
            use_container_width=True, config={"displayModeBar": False}
        )

    with res_col2:
        st.markdown(f"### {pred_regime['label']}")
        st.markdown(pred_regime["description"])
        st.markdown(f"**Stress probability: `{pred_prob:.1%}`**")

        verdict = (
            "🔴 **Model predicts: STRESS**" if result["stress_predicted"] == 1
            else "🟢 **Model predicts: STABLE**"
        )
        st.markdown(verdict)

        st.markdown("**Top 3 drivers for this prediction:**")
        drivers_html = "".join(
            f'<span class="driver-pill">{d.replace("_", " ").title()}</span>'
            for d in result["top_drivers"]
        )
        st.markdown(drivers_html + "<br>", unsafe_allow_html=True)

    # SHAP waterfall
    st.plotly_chart(
        shap_waterfall(result["shap_values"], top_n=12),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.caption(
        f"Model version: {result['model_version']}  ·  "
        f"In-sample Gini: {result['gini_coefficient']:.3f}  ·  "
        "SHAP values computed via TreeExplainer"
    )
