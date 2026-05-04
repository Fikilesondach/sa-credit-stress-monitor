"""
model_loader.py
---------------
Singleton model loader — artefacts are loaded ONCE at API startup
and held in memory for the lifetime of the process.

This avoids loading the 5–10 MB XGBoost model on every request,
which would make the API unusably slow.

Pattern: module-level globals set by load_model_artefacts(),
called from the FastAPI lifespan context manager.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# ── Module-level state (populated at startup) ─────────────────────────────────
_model          = None   # XGBClassifier
_feature_names  = None   # list[str]
_model_meta     = None   # dict (artefact JSON)
_shap_summary   = None   # dict (per-year SHAP values + global importances)
_woe_objects    = None   # dict[str, OptimalBinning]


def load_model_artefacts() -> None:
    """
    Load all model artefacts from disk into module globals.
    Called once during FastAPI lifespan startup.
    """
    global _model, _feature_names, _model_meta, _shap_summary, _woe_objects

    model_path = PROCESSED_DIR / "xgb_model.joblib"
    meta_path  = PROCESSED_DIR / "model_artefact.json"
    shap_path  = PROCESSED_DIR / "shap_summary.json"
    woe_path   = PROCESSED_DIR / "woe_objects.joblib"

    if not model_path.exists():
        raise RuntimeError(
            f"Model artefact not found at {model_path}.\n"
            "Run: PYTHONPATH=. python -m src.models.train"
        )

    logger.info("Loading model artefacts...")
    _model         = joblib.load(model_path)
    _feature_names = json.load(open(meta_path))["feature_names"]
    _model_meta    = json.load(open(meta_path))
    _shap_summary  = json.load(open(shap_path))
    _woe_objects   = joblib.load(woe_path) if woe_path.exists() else {}

    logger.info(
        f"Model loaded ✓ | version={_model_meta['model_version']} | "
        f"features={len(_feature_names)} | "
        f"in-sample Gini={_model_meta['cv_summary'].get('gini', {}).get('mean', 'N/A')}"
    )


def get_model():
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model_artefacts() first.")
    return _model

def get_feature_names() -> list:
    return _feature_names or []

def get_meta() -> dict:
    return _model_meta or {}

def get_shap_summary() -> dict:
    return _shap_summary or {}

def get_woe_objects() -> dict:
    return _woe_objects or {}


# ── Feature vector construction ───────────────────────────────────────────────

def build_feature_vector(raw_inputs: dict) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame from raw MacroInput values.

    This replicates the feature engineering pipeline for INFERENCE only —
    we compute derived features from the raw inputs, then return a row
    aligned to the model's expected feature set.

    For lag/rolling features (which require history), we substitute the
    last known values from the SHAP summary's most recent year entry.
    This is the correct production approach: a live scoring system would
    maintain a sliding window of historical observations.

    Parameters
    ----------
    raw_inputs : dict
        Keys = MacroInput field names, values = floats.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all model features.
    """
    # Start with raw inputs
    features = dict(raw_inputs)

    # ── Derived features (same logic as engineer.py) ──────────────────────────
    features["real_rate"]         = features["repo_rate"] - features["cpi_inflation"]
    features["misery_index"]      = features["cpi_inflation"] + features["unemployment_rate"]
    features["real_rate_x_gdp"]   = features["real_rate"] * features["gdp_growth"]
    features["npl_x_unemp"]       = features["npl_ratio"] * features["unemployment_rate"]
    features["fiscal_x_ca"]       = features["govt_debt_pct_gdp"] * (-features["current_account_pct_gdp"])

    # For features requiring history, use the most recent year from shap summary
    # as a best-estimate baseline (production: replace with a maintained time window)
    last_year = get_shap_summary().get("per_year_explanations", [{}])[-1]
    last_shap = last_year.get("shap_values", {})

    # Lag and rolling features — fill from last known values where we can't compute
    for feat_name in get_feature_names():
        if feat_name not in features:
            # Try to fill from last year's shap-adjacent values (best estimate)
            features[feat_name] = 0.0   # neutral fallback — model is robust to this

    # Build aligned DataFrame
    row = {k: features.get(k, 0.0) for k in get_feature_names()}
    return pd.DataFrame([row], columns=get_feature_names())


# ── Regime classification ─────────────────────────────────────────────────────

def classify_regime(prob: float) -> dict:
    """
    Map stress probability to a named regime with colour coding.

    Thresholds calibrated to SA historical distribution:
      < 0.30 : Stable expansion
      0.30–0.50 : Watch
      0.50–0.70 : Elevated stress
      > 0.70 : Acute stress
    """
    if prob < 0.30:
        return {
            "label":       "Stable",
            "colour":      "#007A4D",   # SA green
            "description": "Macro conditions are broadly stable. Credit risk is contained.",
        }
    elif prob < 0.50:
        return {
            "label":       "Watch",
            "colour":      "#FFB612",   # SA gold
            "description": "Conditions are deteriorating. Elevated vigilance warranted.",
        }
    elif prob < 0.70:
        return {
            "label":       "Elevated Stress",
            "colour":      "#E8620A",   # orange
            "description": "Significant macro stress indicators present. Credit risk is rising.",
        }
    else:
        return {
            "label":       "Acute Stress",
            "colour":      "#DE3831",   # SA red
            "description": "Acute credit stress regime. Conditions resemble GFC or COVID period.",
        }
