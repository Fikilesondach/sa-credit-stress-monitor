"""
api_client.py
-------------
Thin wrapper around the SA Credit Stress Monitor API.

Two modes:
  - API mode   : calls the live FastAPI service (production / Cloud Run)
  - Direct mode: calls the model loader directly (local dev without running the server)

Set API_BASE_URL in .env or Streamlit secrets to switch between them.
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure repo root is on the path (works whether you run from repo root or dashboard/)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Default: direct mode for local development
# Override with: export API_BASE_URL=https://your-cloud-run-url.run.app
API_BASE_URL = os.getenv("API_BASE_URL", "direct")

PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def _ensure_artefacts():
    """
    Auto-train if model artefacts are missing OR corrupted.
    Corruption (EOFError) happens when joblib files are copied across
    machines or environments — the fix is always to retrain locally.
    """
    model_path = PROCESSED_DIR / "xgb_model.joblib"
    shap_path  = PROCESSED_DIR / "shap_summary.json"

    need_train = not model_path.exists()
    if not need_train:
        try:
            import joblib
            joblib.load(model_path)
        except Exception:
            logger.warning("Model artefact corrupt — retraining...")
            model_path.unlink(missing_ok=True)
            need_train = True

    if need_train:
        logger.info("Training model...")
        from src.models.train import train
        train(mode="seed", save=True)

    need_shap = not shap_path.exists()
    if not need_shap:
        try:
            import json as _json
            _json.load(open(shap_path))
        except Exception:
            logger.warning("SHAP artefact corrupt — regenerating...")
            shap_path.unlink(missing_ok=True)
            need_shap = True

    if need_shap:
        logger.info("Generating SHAP artefacts...")
        from src.models.explain import explain
        explain(mode="seed")


def _load_if_needed():
    """Load model into memory if not already loaded."""
    _ensure_artefacts()
    from src.api import model_loader
    if model_loader._model is None:
        model_loader.load_model_artefacts()
    return model_loader


def _direct_health():
    ml = _load_if_needed()
    meta = ml.get_meta()
    return {
        "status": "ok",
        "model_version": meta.get("model_version", "1.0.0"),
        "model_loaded": True,
        "n_features": len(ml.get_feature_names()),
        "train_years": "2000-2023",
        "ingestion_mode": "seed",
    }


def _direct_historical():
    ml = _load_if_needed()
    shap_summary = ml.get_shap_summary()
    entries = shap_summary.get("per_year_explanations", [])
    data = [
        {
            "year":           e["year"],
            "stress_prob":    e["stress_prob"],
            "actual_stress":  e["actual_stress"],
            "top_3_drivers":  e["top_3_drivers"],
        }
        for e in entries
    ]
    return {
        "data":         data,
        "stress_years": sum(1 for d in data if d["actual_stress"]),
        "total_years":  len(data),
    }


def _direct_importance():
    ml = _load_if_needed()
    shap_summary = ml.get_shap_summary()
    global_imp = shap_summary.get("global_feature_importance", {})
    features = [
        {"feature": f, "importance": round(v, 6), "rank": i + 1}
        for i, (f, v) in enumerate(
            sorted(global_imp.items(), key=lambda x: x[1], reverse=True)
        )
    ]
    return {"method": "mean_abs_shap", "features": features}


def _direct_predict(payload: dict):
    import shap as shap_lib
    ml = _load_if_needed()
    model         = ml.get_model()
    feature_names = ml.get_feature_names()
    X             = ml.build_feature_vector(payload)
    prob          = float(model.predict_proba(X)[0, 1])
    prediction    = int(prob >= 0.5)
    regime_dict   = ml.classify_regime(prob)
    explainer     = shap_lib.TreeExplainer(model)
    shap_vals     = explainer.shap_values(X)[0]
    shap_dict     = {
        feat: round(float(val), 6)
        for feat, val in zip(feature_names, shap_vals)
    }
    top_drivers = sorted(shap_dict, key=lambda k: abs(shap_dict[k]), reverse=True)[:3]
    meta = ml.get_meta()
    return {
        "stress_probability": round(prob, 4),
        "stress_predicted":   prediction,
        "gini_coefficient":   meta.get("cv_summary", {}).get("gini", {}).get("mean", 0.0),
        "regime":             regime_dict,
        "top_drivers":        top_drivers,
        "shap_values":        shap_dict,
        "model_version":      meta.get("model_version", "1.0.0"),
    }


# ── Public interface ──────────────────────────────────────────────────────────

def get_health() -> dict:
    if API_BASE_URL == "direct":
        return _direct_health()
    import requests
    return requests.get(f"{API_BASE_URL}/health", timeout=10).json()


def get_historical() -> dict:
    if API_BASE_URL == "direct":
        return _direct_historical()
    import requests
    return requests.get(f"{API_BASE_URL}/historical", timeout=10).json()


def get_feature_importance() -> dict:
    if API_BASE_URL == "direct":
        return _direct_importance()
    import requests
    return requests.get(f"{API_BASE_URL}/feature-importance", timeout=10).json()


def post_predict(payload: dict) -> dict:
    if API_BASE_URL == "direct":
        return _direct_predict(payload)
    import requests
    return requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=15).json()
