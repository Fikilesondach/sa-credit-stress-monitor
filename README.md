# SA Macroeconomic Credit Stress Early Warning System

A cloud-deployed ML pipeline that monitors South African macroeconomic conditions
and predicts credit stress regimes using real public data sources.

**Live demo:** *(deployed link goes here after Step 3)*

---

## What it does

Ingests real SA macro data → trains an XGBoost classifier → serves predictions
via a FastAPI REST endpoint → visualises current regime probability on a Streamlit dashboard.

| Component        | Technology                        |
|------------------|-----------------------------------|
| Data ingestion   | World Bank API · FRED · yfinance  |
| ML model         | XGBoost + SHAP explainability     |
| API              | FastAPI · Pydantic                |
| Dashboard        | Streamlit · Plotly                |
| Cloud deploy     | GCP Cloud Run · Docker            |
| CI/CD            | GitHub Actions                    |

---

## Data sources (all real, all free)

| Source   | Data                                         | API Key? |
|----------|----------------------------------------------|----------|
| World Bank Open Data | GDP growth, CPI, unemployment, NPL ratio, govt debt | No |
| FRED (St. Louis Fed) | VIX, USD broad index, SA discount rate | Free key |
| yfinance | ZAR/USD spot, JSE All Share Index | No |
| SARB / Stats SA | Repo rate, prime rate (via seed data) | No |

---

## Project structure

```
sa-credit-stress-monitor/
├── data/
│   ├── raw/              ← ingested datasets (parquet)
│   └── processed/        ← feature-engineered, model-ready data
├── src/
│   ├── ingestion/
│   │   ├── seed_data.py  ← real historical SA macro data (2000–2023)
│   │   ├── world_bank.py ← World Bank API pull
│   │   ├── fred.py       ← FRED API pull
│   │   ├── market.py     ← yfinance market data pull
│   │   └── pipeline.py   ← master orchestrator
│   ├── features/         ← feature engineering (Step 2)
│   ├── models/           ← XGBoost training + SHAP (Step 2)
│   └── api/              ← FastAPI app (Step 3)
├── notebooks/            ← EDA and model exploration
├── tests/
├── Dockerfile            ← (Step 3)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/Fikilesondach/sa-credit-stress-monitor
cd sa-credit-stress-monitor
pip install -r requirements.txt
cp .env.example .env       # add your FRED_API_KEY

# Run ingestion (seed mode — no API key needed)
PYTHONPATH=. python -m src.ingestion.pipeline --mode seed --save

# Run ingestion (auto mode — live APIs where available)
PYTHONPATH=. python -m src.ingestion.pipeline --mode auto --save
```

---

## Credit stress label definition

A year is labelled **stress = 1** when any of the following hold:
- Bank NPL ratio ≥ 4.0%
- Real GDP growth < 0%
- Unemployment > 30% AND NPL > 3.5%

Historical stress years (2000–2023): **2001, 2002, 2008, 2009, 2010, 2016, 2017, 2019, 2020, 2022, 2023**

---

## Build phases

- [x] **Step 1** — Data ingestion pipeline (this file)
- [ ] **Step 2** — Feature engineering + XGBoost model + SHAP
- [ ] **Step 3** — FastAPI endpoint + Docker
- [ ] **Step 4** — Streamlit dashboard
- [ ] **Step 5** — GCP Cloud Run deployment + GitHub Actions CI/CD

---

*Built by Fikile Sondach · Senior BI Analyst → Credit Risk Data Scientist*
