"""
schemas.py
----------
Pydantic v2 request/response models for the SA Credit Stress Monitor API.

Every field is annotated with a description so the auto-generated
OpenAPI docs at /docs are self-explanatory to any analyst or engineer
reading them.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request schemas ────────────────────────────────────────────────────────────

class MacroInput(BaseModel):
    """
    Macroeconomic snapshot for a single year.
    All values are real (not nominal) where applicable.
    """
    gdp_growth:               float = Field(..., description="Real GDP growth rate (%)", examples=[0.7])
    cpi_inflation:            float = Field(..., description="CPI inflation rate (% annual)", examples=[6.1])
    unemployment_rate:        float = Field(..., description="Unemployment rate (% of total labour force)", examples=[32.9])
    repo_rate:                float = Field(..., description="SARB repo rate, year-end (%)", examples=[8.25])
    npl_ratio:                float = Field(..., description="Bank non-performing loans (% of gross loans)", examples=[4.1])
    private_credit_pct_gdp:   float = Field(..., description="Domestic credit to private sector (% GDP)", examples=[140.2])
    govt_debt_pct_gdp:        float = Field(..., description="Central government debt (% GDP)", examples=[73.8])
    current_account_pct_gdp:  float = Field(..., description="Current account balance (% GDP, negative = deficit)", examples=[-1.6])
    zar_usd:                  float = Field(..., description="ZAR/USD annual average exchange rate", examples=[18.45])
    vix_avg:                  float = Field(..., description="CBOE VIX annual average (global risk proxy)", examples=[16.9])

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


# ── Response schemas ───────────────────────────────────────────────────────────

class StressRegime(BaseModel):
    label:       str   = Field(..., description="Human-readable regime label")
    colour:      str   = Field(..., description="Hex colour for dashboard rendering")
    description: str   = Field(..., description="Plain-English interpretation")


class PredictionResponse(BaseModel):
    stress_probability: float       = Field(..., description="Model's credit stress probability (0–1)")
    stress_predicted:   int         = Field(..., description="Binary prediction: 1=stress, 0=stable")
    gini_coefficient:   float       = Field(..., description="Model in-sample Gini (0.825) — for reference")
    regime:             StressRegime
    top_drivers:        List[str]   = Field(..., description="Top 3 SHAP feature drivers for this prediction")
    shap_values:        Dict[str, float] = Field(..., description="SHAP attribution for each model feature")
    model_version:      str         = Field(..., description="Model version string")


class HistoricalEntry(BaseModel):
    year:           int
    stress_prob:    float
    actual_stress:  int
    top_3_drivers:  List[str]


class HistoricalResponse(BaseModel):
    data:         List[HistoricalEntry]
    stress_years: int
    total_years:  int


class FeatureImportanceEntry(BaseModel):
    feature:     str
    importance:  float
    rank:        int


class ImportanceResponse(BaseModel):
    method:   str = "mean_abs_shap"
    features: List[FeatureImportanceEntry]


class HealthResponse(BaseModel):
    status:          str
    model_version:   str
    model_loaded:    bool
    n_features:      int
    train_years:     str
    ingestion_mode:  str
