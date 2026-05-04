"""
test_api.py
-----------
Pytest integration tests for the SA Credit Stress Monitor API.

Uses FastAPI's built-in TestClient (via httpx) — no live server needed.
Tests run against real model artefacts loaded from data/processed/.

Run:
    PYTHONPATH=. pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

# ── Fixture: 2023 SA macro snapshot (stress year, prob ≈ 0.56) ───────────────
PAYLOAD_2023 = {
    "gdp_growth":              0.7,
    "cpi_inflation":           6.1,
    "unemployment_rate":       32.9,
    "repo_rate":               8.25,
    "npl_ratio":               4.1,
    "private_credit_pct_gdp":  140.2,
    "govt_debt_pct_gdp":       73.8,
    "current_account_pct_gdp": -1.6,
    "zar_usd":                 18.45,
    "vix_avg":                 16.9,
}

# ── Fixture: 2006 stable snapshot (boom year, prob should be low) ─────────────
PAYLOAD_2006 = {
    "gdp_growth":              5.6,
    "cpi_inflation":           4.7,
    "unemployment_rate":       25.5,
    "repo_rate":               9.0,
    "npl_ratio":               1.1,
    "private_credit_pct_gdp":  161.9,
    "govt_debt_pct_gdp":       27.8,
    "current_account_pct_gdp": -5.3,
    "zar_usd":                 6.77,
    "vix_avg":                 12.8,
}


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_model_loaded(self):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_fields_present(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_version" in data
        assert "n_features" in data
        assert data["status"] == "ok"

    def test_health_n_features_positive(self):
        data = client.get("/health").json()
        assert data["n_features"] > 0


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_200(self):
        resp = client.post("/predict", json=PAYLOAD_2023)
        assert resp.status_code == 200

    def test_predict_probability_in_range(self):
        data = client.post("/predict", json=PAYLOAD_2023).json()
        assert 0.0 <= data["stress_probability"] <= 1.0

    def test_predict_binary_label(self):
        data = client.post("/predict", json=PAYLOAD_2023).json()
        assert data["stress_predicted"] in [0, 1]

    def test_predict_regime_present(self):
        data = client.post("/predict", json=PAYLOAD_2023).json()
        assert "regime" in data
        assert "label" in data["regime"]
        assert "colour" in data["regime"]
        assert "description" in data["regime"]

    def test_predict_top_drivers_length(self):
        data = client.post("/predict", json=PAYLOAD_2023).json()
        assert len(data["top_drivers"]) == 3

    def test_predict_shap_values_present(self):
        data = client.post("/predict", json=PAYLOAD_2023).json()
        assert isinstance(data["shap_values"], dict)
        assert len(data["shap_values"]) > 0

    def test_predict_2023_is_stress(self):
        """2023 was a stress year — model should predict elevated probability."""
        data = client.post("/predict", json=PAYLOAD_2023).json()
        assert data["stress_probability"] > 0.45, (
            f"Expected elevated stress for 2023, got {data['stress_probability']}"
        )

    def test_predict_2006_is_stable(self):
        """2006 was a boom year — model should predict low stress probability."""
        data = client.post("/predict", json=PAYLOAD_2006).json()
        assert data["stress_probability"] < 0.55, (
            f"Expected low stress for 2006 boom conditions, got {data['stress_probability']}"
        )

    def test_predict_missing_field_returns_422(self):
        """Pydantic should reject payloads with missing required fields."""
        incomplete = {k: v for k, v in PAYLOAD_2023.items() if k != "npl_ratio"}
        resp = client.post("/predict", json=incomplete)
        assert resp.status_code == 422

    def test_predict_wrong_type_returns_422(self):
        """Pydantic should reject non-numeric values."""
        bad_payload = {**PAYLOAD_2023, "gdp_growth": "not_a_number"}
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_predict_regime_colour_is_hex(self):
        data = client.post("/predict", json=PAYLOAD_2023).json()
        colour = data["regime"]["colour"]
        assert colour.startswith("#"), f"Expected hex colour, got: {colour}"
        assert len(colour) == 7


# ── /historical ───────────────────────────────────────────────────────────────

class TestHistorical:
    def test_historical_returns_200(self):
        resp = client.get("/historical")
        assert resp.status_code == 200

    def test_historical_data_length(self):
        data = client.get("/historical").json()
        assert len(data["data"]) == 24   # 2000–2023

    def test_historical_stress_count(self):
        data = client.get("/historical").json()
        assert data["stress_years"] == 11

    def test_historical_years_ascending(self):
        data = client.get("/historical").json()
        years = [e["year"] for e in data["data"]]
        assert years == sorted(years)

    def test_historical_probs_in_range(self):
        data = client.get("/historical").json()
        for entry in data["data"]:
            assert 0.0 <= entry["stress_prob"] <= 1.0

    def test_historical_gfc_2009_is_stress(self):
        """GFC 2009 must be labelled as actual stress."""
        data = client.get("/historical").json()
        gfc  = next(e for e in data["data"] if e["year"] == 2009)
        assert gfc["actual_stress"] == 1

    def test_historical_2005_is_stable(self):
        """2005 was a stable boom year."""
        data = client.get("/historical").json()
        y2005 = next(e for e in data["data"] if e["year"] == 2005)
        assert y2005["actual_stress"] == 0


# ── /feature-importance ───────────────────────────────────────────────────────

class TestFeatureImportance:
    def test_importance_returns_200(self):
        resp = client.get("/feature-importance")
        assert resp.status_code == 200

    def test_importance_features_present(self):
        data = client.get("/feature-importance").json()
        assert "features" in data
        assert len(data["features"]) > 0

    def test_importance_sorted_descending(self):
        data    = client.get("/feature-importance").json()
        scores  = [f["importance"] for f in data["features"]]
        assert scores == sorted(scores, reverse=True)

    def test_importance_ranks_sequential(self):
        data  = client.get("/feature-importance").json()
        ranks = [f["rank"] for f in data["features"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_importance_top_feature_is_macro(self):
        """Top driver should be a recognisable macro indicator."""
        data     = client.get("/feature-importance").json()
        top_feat = data["features"][0]["feature"]
        macro_keywords = [
            "npl", "gdp", "unemployment", "repo", "current_account",
            "govt_debt", "vix", "zar", "inflation", "credit",
        ]
        assert any(kw in top_feat for kw in macro_keywords), (
            f"Unexpected top feature: {top_feat}"
        )

    def test_importance_method_is_shap(self):
        data = client.get("/feature-importance").json()
        assert data["method"] == "mean_abs_shap"


# ── Root ──────────────────────────────────────────────────────────────────────

class TestRoot:
    def test_root_returns_200(self):
        assert client.get("/").status_code == 200

    def test_root_has_docs_link(self):
        data = client.get("/").json()
        assert "docs" in data
