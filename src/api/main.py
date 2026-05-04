"""
main.py
-------
SA Credit Stress Monitor — FastAPI REST API

Endpoints:
  GET  /health                → model status + metadata
  POST /predict               → stress probability for a given macro snapshot
  GET  /historical            → all historical predictions (2000–2023)
  GET  /feature-importance    → global SHAP feature ranking
  GET  /docs                  → interactive Swagger UI (auto-generated)
  GET  /redoc                 → ReDoc documentation

Run locally:
    PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000

Then visit: http://localhost:8000/docs
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict

import numpy as np
import shap
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    MacroInput,
    PredictionResponse,
    StressRegime,
    HistoricalResponse,
    HistoricalEntry,
    ImportanceResponse,
    FeatureImportanceEntry,
    HealthResponse,
)
from src.api import model_loader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ── Lifespan: load model artefacts once at startup ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artefacts on startup, release on shutdown."""
    logger.info("Starting SA Credit Stress Monitor API...")
    model_loader.load_model_artefacts()
    logger.info("API ready.")
    yield
    logger.info("API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SA Macroeconomic Credit Stress Monitor",
    description=(
        "Real-time credit stress early warning system for South Africa.\n\n"
        "Uses an XGBoost classifier trained on 24 years of real SA macroeconomic "
        "data (SARB · Stats SA · World Bank · FRED) to predict whether current "
        "conditions constitute a **credit stress regime**.\n\n"
        "Built by [Fikile Sondach](https://github.com/Fikilesondach) · "
        "Senior BI Analyst → Credit Risk Data Scientist"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit dashboard and any frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "SA Credit Stress Monitor API",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    tags=["System"],
)
async def health():
    """
    Returns model status, version, and metadata.
    Use this to verify the API is running and the model is loaded.
    """
    meta = model_loader.get_meta()
    return HealthResponse(
        status         = "ok",
        model_version  = meta.get("model_version", "unknown"),
        model_loaded   = model_loader._model is not None,
        n_features     = len(model_loader.get_feature_names()),
        train_years    = "2000–2023",
        ingestion_mode = "seed",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict credit stress probability",
    tags=["Prediction"],
)
async def predict(inputs: MacroInput):
    """
    Given a macroeconomic snapshot, returns:
    - **stress_probability**: model's predicted probability (0–1)
    - **stress_predicted**: binary prediction (0=stable, 1=stress)
    - **regime**: named stress regime with colour code
    - **top_drivers**: top 3 SHAP features driving this prediction
    - **shap_values**: full SHAP attribution for all model features

    The example values in the schema correspond to SA's 2023 macro conditions.
    """
    try:
        model         = model_loader.get_model()
        feature_names = model_loader.get_feature_names()

        # Build feature vector
        X = model_loader.build_feature_vector(inputs.model_dump())

        # Predict
        prob        = float(model.predict_proba(X)[0, 1])
        prediction  = int(prob >= 0.5)
        regime_dict = model_loader.classify_regime(prob)

        # SHAP for this observation
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X)[0]
        shap_dict  = {
            feat: round(float(val), 6)
            for feat, val in zip(feature_names, shap_vals)
        }
        top_drivers = sorted(shap_dict, key=lambda k: abs(shap_dict[k]), reverse=True)[:3]

        meta = model_loader.get_meta()

        return PredictionResponse(
            stress_probability = round(prob, 4),
            stress_predicted   = prediction,
            gini_coefficient   = meta.get("cv_summary", {}).get("gini", {}).get("mean", 0.0),
            regime             = StressRegime(**regime_dict),
            top_drivers        = top_drivers,
            shap_values        = shap_dict,
            model_version      = meta.get("model_version", "1.0.0"),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/historical",
    response_model=HistoricalResponse,
    summary="Historical stress predictions (2000–2023)",
    tags=["Analysis"],
)
async def historical():
    """
    Returns model stress probabilities and actual stress labels
    for every year from 2000 to 2023.

    Useful for the dashboard time-series chart and for validating
    that the model correctly identified historical stress events:
    rand crisis (2002), GFC (2008–09), SOE crisis (2016–17),
    COVID (2020), load shedding peak (2022–23).
    """
    shap_summary = model_loader.get_shap_summary()
    entries      = shap_summary.get("per_year_explanations", [])

    data = [
        HistoricalEntry(
            year          = e["year"],
            stress_prob   = e["stress_prob"],
            actual_stress = e["actual_stress"],
            top_3_drivers = e["top_3_drivers"],
        )
        for e in entries
    ]

    stress_years = sum(1 for e in data if e.actual_stress == 1)
    return HistoricalResponse(
        data         = data,
        stress_years = stress_years,
        total_years  = len(data),
    )


@app.get(
    "/feature-importance",
    response_model=ImportanceResponse,
    summary="Global SHAP feature importance ranking",
    tags=["Analysis"],
)
async def feature_importance():
    """
    Returns the global feature importance ranking by mean absolute SHAP value.

    Higher values = stronger influence on the model's predictions overall.
    This is model-agnostic (based on SHAP, not XGBoost's internal gain metric)
    and is the industry standard for credit risk model explainability.
    """
    shap_summary  = model_loader.get_shap_summary()
    global_imp    = shap_summary.get("global_feature_importance", {})

    features = [
        FeatureImportanceEntry(
            feature    = feat,
            importance = round(imp, 6),
            rank       = rank,
        )
        for rank, (feat, imp) in enumerate(
            sorted(global_imp.items(), key=lambda x: x[1], reverse=True),
            start=1,
        )
    ]

    return ImportanceResponse(features=features)


# ── Error handlers ────────────────────────────────────────────────────────────

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "docs": "/docs"},
    )

@app.exception_handler(500)
async def server_error(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error — check API logs."},
    )
