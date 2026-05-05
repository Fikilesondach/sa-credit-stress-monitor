# SA Macroeconomic Credit Stress Monitor

> **Live demo:** [sa-credit-stress-monitor.streamlit.app](https://sa-credit-stress-monitor-fsc7e8mz5xcqmmyxepn3dg.streamlit.app)  
> **API docs:** [Cloud Run — Swagger UI](https://sa-credit-stress-monitor-13790884044.africa-south1.run.app/docs)  
> **Stack:** Python · XGBoost · SHAP · FastAPI · Streamlit · Docker · GCP Cloud Run

---

## What This Is

An end-to-end, production-deployed machine learning system that monitors South African macroeconomic conditions and classifies the current credit stress regime — Stable, Watch, Elevated Stress, or Acute Stress — using 24 years of real public data (2000–2023).

This is not a synthetic demo. All training data is sourced from the SARB, Stats SA, World Bank, IMF, and FRED. The model is deployed as a live REST API on GCP Cloud Run (Johannesburg region) and surfaced through an interactive Streamlit dashboard with real-time scenario scoring, SHAP explainability, and a 24-year historical timeline.

---

## The Problem

South African credit risk does not move in isolation from the macroeconomy. NPL cycles, repo rate shocks, rand depreciation, and unemployment spikes interact in non-linear ways that simple threshold rules miss. Most credit stress tools in the SA market are either proprietary, backward-looking, or built on synthetic data.

This project builds a reproducible, explainable, open-source alternative — grounded in real SA macro history and trained to detect the conditions that preceded every major credit stress event since 2000.

---

## Model Architecture

### Data

16 raw macroeconomic features ingested from four public sources:

| Feature | Source |
|---|---|
| GDP growth, CPI, unemployment, repo rate | SARB / Stats SA |
| NPL ratio, private credit growth, government debt, current account | World Bank Open Data |
| VIX (global risk proxy), USD broad index | FRED (St. Louis Fed) |
| ZAR/USD spot rate, JSE All Share annual return | yfinance |

The ingestion pipeline runs in three modes: `seed` (offline, 24-year hand-sourced fallback), `live` (API pull), and `auto` (live where available, seed as fallback per source).

### Feature Engineering

116 features are derived from 16 raw inputs:

- **Lag features** (t-1, t-2) — stress is typically preceded by deterioration the prior year
- **Rolling statistics** (2yr and 3yr mean + std) — captures regime persistence vs. volatility
- **Interaction terms** — economically motivated cross-features: real rate × GDP growth, NPL × unemployment, fiscal deficit × current account deficit, ZAR depreciation × VIX, misery index (CPI + unemployment)
- **Stress momentum** — lagged stress label, because stress years cluster
- **WoE encoding** — Weight of Evidence on top predictors using `optbinning`, converting continuous variables into monotonic scorecard-style predictors consistent with actuarial credit modelling practice

Feature selection retains the top 20 by XGBoost gain importance, estimated once before cross-validation to prevent leakage on the 24-observation dataset.

### Model

**XGBoost binary classifier** trained to predict credit stress years. A year is labelled stress if: NPL ≥ 4%, OR GDP growth < 0%, OR unemployment > 30% with NPL > 3.5%. This produces 11 historically labelled stress years across the 2000–2023 window.

Key design decisions for a small-N actuarial setting:

- Walk-forward `TimeSeriesSplit` cross-validation (5 folds) — the model never trains on future data
- Heavy regularisation (`max_depth=2`, L1 + L2, `min_child_weight=3`) appropriate for 24 observations
- In-sample Gini coefficient: **0.825**

### Explainability

Full SHAP suite via `TreeExplainer`:

- Global importance (mean |SHAP| per feature)
- Beeswarm plot (direction + magnitude across all years)
- Per-prediction waterfall charts (what drove 2020 vs 2023 specifically)
- Historical prediction chart — model probability vs. actual stress events, annotated for the 2001 rand crisis, 2009 GFC, 2015–2018 Zuma-era SOE crisis, 2020 COVID shock, and 2022–2023 load shedding peak

### Regime Classification

Stress probability is mapped to four regimes with SA-flag colour coding:

| Probability | Regime | Colour |
|---|---|---|
| < 0.30 | Stable | Green |
| 0.30 – 0.50 | Watch | Gold |
| 0.50 – 0.70 | Elevated Stress | Orange |
| > 0.70 | Acute Stress | Red |

---

## API Reference

Base URL: `https://sa-credit-stress-monitor-13790884044.africa-south1.run.app`

### `GET /health`
Model status, version, feature count, training window.

### `POST /predict`
Takes a macro snapshot, returns stress probability, regime label, top 3 SHAP drivers, and full SHAP attribution across all 20 features.

```json
{
  "gdp_growth": -1.5,
  "cpi_inflation": 5.9,
  "unemployment_rate": 32.9,
  "repo_rate": 8.25,
  "npl_ratio": 4.8,
  "govt_debt_pct_gdp": 73.0,
  "current_account_pct_gdp": -2.1,
  "zar_usd": 18.63,
  "vix_avg": 21.4,
  "private_credit_pct_gdp": 62.1
}
```

**Response:**
```json
{
  "stress_probability": 0.612,
  "stress_predicted": 1,
  "regime": {
    "label": "Elevated Stress",
    "colour": "#E8620A",
    "description": "Significant macro stress indicators present. Credit risk is rising."
  },
  "top_drivers": ["repo_rate_roll3std", "gdp_growth_roll2m", "misery_index"],
  "shap_values": { ... },
  "model_version": "1.0.0"
}
```

### `GET /historical`
All 24 years of model predictions vs. actual stress labels.

### `GET /feature-importance`
Global SHAP ranking across the 20 selected features.

Full interactive docs at `/docs` (Swagger UI).

---

## Project Structure

```
sa-credit-stress-monitor/
├── src/
│   ├── ingestion/      seed_data · world_bank · fred · market · pipeline
│   ├── features/       engineer
│   ├── models/         train · explain
│   └── api/            main · schemas · model_loader
├── dashboard/
│   ├── app.py
│   ├── charts.py
│   ├── api_client.py
│   └── requirements.txt
├── tests/              test_api · conftest
├── data/
│   ├── raw/            sa_macro_dataset.parquet
│   └── processed/      xgb_model.joblib · model_artefact.json · shap_summary.json
├── .github/workflows/  ci.yml
├── .streamlit/         config.toml · secrets.toml.template
├── Dockerfile
├── deploy_gcp.sh
├── runtime.txt
├── packages.txt
└── requirements.txt
```

---

## CI/CD Pipeline

GitHub Actions runs four jobs on every push to `main`:

1. **Test** — trains the model in seed mode, runs 30 pytest tests covering all endpoints, schema validation, and economic sanity checks (2006 boom should score low, 2009 GFC should score high)
2. **Build** — builds a Docker image (`linux/amd64`, multi-stage, non-root user) and pushes to Google Container Registry
3. **Deploy** — deploys to Cloud Run (`africa-south1`, Johannesburg) and smoke-tests the live `/health` endpoint
4. **Retrain** (scheduled weekly, Monday 02:00 UTC) — pulls latest live data, retrains, tests, commits artefacts, rebuilds the image, and redeploys

---

## Local Development

```bash
# Clone and install
git clone https://github.com/Fikilesondach/sa-credit-stress-monitor.git
cd sa-credit-stress-monitor
pip install -r requirements.txt

# Train model (seed mode — no API keys needed)
PYTHONPATH=. python -m src.models.train
PYTHONPATH=. python -m src.models.explain

# Run API
PYTHONPATH=. uvicorn src.api.main:app --reload --port 8080

# Run dashboard (connects to local API in direct mode)
streamlit run dashboard/app.py

# Run tests
PYTHONPATH=. pytest tests/test_api.py -v
```

For live data ingestion, create a `.env` file:

```
FRED_API_KEY=your_free_key_from_fred.stlouisfed.org
```

---

## GCP Deployment

First-time setup (creates service account, configures IAM, builds and deploys):

```bash
chmod +x deploy_gcp.sh
./deploy_gcp.sh sa-credit-stress-prod
```

The script pauses at step 5 to print the base64 service account key for GitHub Secrets, then completes the Docker build, GCR push, and Cloud Run deploy automatically.

---

## Technical Notes

**Why XGBoost over a logistic regression scorecard?** With 24 observations the model is heavily regularised regardless of family. XGBoost with `max_depth=2` and strong L1/L2 penalties produces a near-additive model that SHAP can decompose cleanly — giving the interpretability of a scorecard with the flexibility to capture the NPL × unemployment interaction that a linear model misses.

**Why WoE encoding?** The actuarial credit tradition uses WoE because it converts raw macroeconomic magnitudes into monotonic, unit-free signals aligned with default risk direction. This is especially valuable for rolling features where the raw scale changes meaning across regimes.

**Why `numpy<2.0`?** NumPy 2.0 introduced breaking changes to the C API that affected pyarrow, scikit-learn, and plotly.express at the time this project was built. The pin will be lifted once all upstream packages declare 2.0 support.

---

## Data Sources

| Source | Data | Access |
|---|---|---|
| SARB / Stats SA | GDP, CPI, unemployment, repo rate, NPL | Hand-sourced (seed), free |
| World Bank Open Data | NPL, private credit, government debt, current account | Free API, no key |
| FRED (St. Louis Fed) | VIX, USD broad index | Free API, free key |
| Yahoo Finance | ZAR/USD, JSE All Share | Free, no key |

---

## Author

**Fikile Sondach**   
BSc Actuarial Science · IFoA CT1, CT2, CT3, CT7, CT8  
[github.com/Fikilesondach](https://github.com/Fikilesondach)
