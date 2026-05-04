"""
explain.py
----------
SHAP (SHapley Additive exPlanations) for the SA Credit Stress XGBoost model.

Produces three explanation artefacts:
  1. Global importance  — SHAP mean |value| bar chart (what drives the model overall)
  2. Beeswarm plot      — SHAP value distribution per feature (direction + magnitude)
  3. Waterfall plots    — per-year local explanations (what drove THAT prediction)
  4. Historical stress  — model probability vs actual stress label across 2000-2023
  5. SHAP summary JSON  — machine-readable feature attributions for the API/dashboard

All charts saved to data/processed/shap_*.png

SHAP reference: https://shap.readthedocs.io
Run:
    PYTHONPATH=. python -m src.models.explain
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (SA flag inspired) ─────────────────────────────────────────
GREEN  = "#007A4D"
RED    = "#DE3831"
GOLD   = "#FFB612"
NAVY   = "#002395"
WHITE  = "#FFFFFF"
GREY   = "#4A4A4A"


# ── Load artefacts ─────────────────────────────────────────────────────────────

def load_artefacts():
    model_path = PROCESSED_DIR / "xgb_model.joblib"
    meta_path  = PROCESSED_DIR / "model_artefact.json"

    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found. Run: PYTHONPATH=. python -m src.models.train"
        )
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    return model, meta


# ── SHAP computation ──────────────────────────────────────────────────────────

def compute_shap_values(model, X: pd.DataFrame):
    """
    Compute SHAP values using TreeExplainer (exact, fast for XGBoost).

    Returns
    -------
    explainer : shap.TreeExplainer
    shap_values : np.ndarray  (n_samples × n_features)
    expected_value : float    (model baseline log-odds)
    """
    explainer     = shap.TreeExplainer(model)
    shap_values   = explainer.shap_values(X)
    expected_value = explainer.expected_value
    logger.info(
        f"SHAP computed | shape={shap_values.shape} | "
        f"baseline={expected_value:.4f}"
    )
    return explainer, shap_values, expected_value


# ── Plot 1: Global SHAP importance bar chart ───────────────────────────────────

def plot_global_importance(shap_values, X: pd.DataFrame, top_n: int = 15) -> str:
    """Mean |SHAP| bar chart — most impactful features overall."""
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns,
    ).sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(mean_abs.index, mean_abs.values, color=GREEN, edgecolor="white", height=0.7)
    ax.bar_label(bars, fmt="%.4f", padding=4, color=GREY, fontsize=9)

    ax.set_xlabel("Mean |SHAP Value|  (average impact on model output)", fontsize=11)
    ax.set_title(
        "SA Credit Stress Monitor\nGlobal Feature Importance (SHAP)",
        fontsize=13, fontweight="bold", color=NAVY, pad=15,
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#F9F9F9")
    fig.patch.set_facecolor(WHITE)

    # Watermark
    ax.text(0.99, 0.01, "SA Credit Stress Monitor | github.com/Fikilesondach",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="lightgrey")

    out = PROCESSED_DIR / "shap_global_importance.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return str(out)


# ── Plot 2: Beeswarm / summary plot ───────────────────────────────────────────

def plot_beeswarm(shap_values, X: pd.DataFrame, top_n: int = 15) -> str:
    """SHAP beeswarm — shows direction AND magnitude of each feature's impact."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Select top-N by mean |SHAP|
    mean_abs  = np.abs(shap_values).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[-top_n:]
    shap_top  = shap_values[:, top_idx]
    X_top     = X.iloc[:, top_idx]
    feat_names = X.columns[top_idx].tolist()

    shap.summary_plot(
        shap_top, X_top,
        feature_names=feat_names,
        show=False,
        plot_size=(10, 6),
        color_bar_label="Feature value",
    )

    plt.title(
        "SA Credit Stress Monitor\nSHAP Beeswarm — Direction & Magnitude",
        fontsize=13, fontweight="bold", color=NAVY, pad=10,
    )
    out = PROCESSED_DIR / "shap_beeswarm.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return str(out)


# ── Plot 3: Historical probability vs actual ───────────────────────────────────

def plot_historical_predictions(
    model, X: pd.DataFrame, y: pd.Series
) -> str:
    """
    Model stress probability vs actual stress label across all years.

    This is the key chart for the portfolio — shows the model
    tracking real SA credit stress events (GFC, load shedding, COVID).
    """
    y_prob = model.predict_proba(X)[:, 1]
    years  = X.index.year

    fig, ax = plt.subplots(figsize=(13, 5))

    # Shade actual stress periods
    for i, (yr, stress) in enumerate(zip(years, y.values)):
        if stress == 1:
            ax.axvspan(yr - 0.5, yr + 0.5, color=RED, alpha=0.15, linewidth=0)

    # Probability line
    ax.plot(years, y_prob, color=NAVY, linewidth=2.5, marker="o",
            markersize=5, label="Stress probability (model)")

    # Decision threshold
    ax.axhline(0.5, color=GREY, linestyle="--", linewidth=1, alpha=0.6, label="Threshold (0.5)")

    # Label key events
    events = {
        2002: "Rand crisis",
        2009: "GFC",
        2016: "Zuma era\nSO crisis",
        2020: "COVID-19",
        2023: "Load\nshedding peak",
    }
    for yr, label in events.items():
        if yr in years.tolist():
            prob = y_prob[years.tolist().index(yr)]
            ax.annotate(
                label, xy=(yr, prob),
                xytext=(yr, min(prob + 0.22, 0.95)),
                fontsize=7.5, ha="center", color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8),
            )

    # Legend
    stress_patch = mpatches.Patch(color=RED, alpha=0.3, label="Actual stress year")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [stress_patch], labels + ["Actual stress year"],
              fontsize=9, loc="upper left")

    ax.set_xlim(years.min() - 0.7, years.max() + 0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Credit Stress Probability", fontsize=11)
    ax.set_title(
        "SA Macroeconomic Credit Stress Monitor\n"
        "XGBoost Predicted Stress Probability vs Actual Stress Events (2000–2023)",
        fontsize=13, fontweight="bold", color=NAVY, pad=15,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#F9F9F9")
    fig.patch.set_facecolor(WHITE)
    ax.text(0.99, 0.01, "SA Credit Stress Monitor | github.com/Fikilesondach",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="lightgrey")

    out = PROCESSED_DIR / "shap_historical_predictions.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return str(out)


# ── Plot 4: Waterfall for a specific year ─────────────────────────────────────

def plot_waterfall(
    explainer, shap_values, X: pd.DataFrame,
    year: int = 2023,
) -> str:
    """Local SHAP waterfall for a specific year — what drove THAT prediction."""
    years = X.index.year.tolist()
    if year not in years:
        year = years[-1]
        logger.warning(f"Year not found — using {year}")

    idx = years.index(year)

    explanation = shap.Explanation(
        values         = shap_values[idx],
        base_values    = explainer.expected_value,
        data           = X.iloc[idx].values,
        feature_names  = X.columns.tolist(),
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(
        f"SA Credit Stress Monitor\nSHAP Waterfall — {year} Prediction Explained",
        fontsize=12, fontweight="bold", color=NAVY, pad=10,
    )

    out = PROCESSED_DIR / f"shap_waterfall_{year}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return str(out)


# ── Export SHAP summary as JSON (for API + dashboard) ─────────────────────────

def export_shap_json(
    shap_values, X: pd.DataFrame, y: pd.Series, model
) -> dict:
    """
    Machine-readable SHAP summary for the FastAPI endpoint and Streamlit dashboard.

    Includes per-year attributions and global feature rankings.
    """
    y_prob     = model.predict_proba(X)[:, 1]
    mean_shap  = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X.columns
    ).sort_values(ascending=False)

    per_year = []
    for i, (date, row) in enumerate(X.iterrows()):
        year_shap = {
            feat: round(float(shap_values[i, j]), 6)
            for j, feat in enumerate(X.columns)
        }
        per_year.append({
            "year":           int(date.year),
            "stress_prob":    round(float(y_prob[i]), 4),
            "actual_stress":  int(y.iloc[i]),
            "shap_values":    year_shap,
            "top_3_drivers":  list(
                pd.Series(year_shap).abs().sort_values(ascending=False).head(3).index
            ),
        })

    summary = {
        "global_feature_importance": {
            feat: round(float(imp), 6)
            for feat, imp in mean_shap.head(15).items()
        },
        "per_year_explanations": per_year,
    }

    out = PROCESSED_DIR / "shap_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"SHAP JSON saved: {out}")
    return summary


# ── Master explain function ───────────────────────────────────────────────────

def explain(mode: str = "seed") -> dict:
    """
    Full SHAP explanation pipeline.

    Returns the SHAP summary dict (also saved to disk).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.ingestion.pipeline import load_macro_dataset
    from src.features.engineer  import build_feature_matrix

    logger.info("=" * 65)
    logger.info("SA CREDIT STRESS MONITOR — SHAP EXPLAINABILITY")
    logger.info("=" * 65)

    model, artefact_meta = load_artefacts()
    selected_features    = artefact_meta["feature_names"]

    df_raw    = load_macro_dataset(mode=mode)
    X_full, y, _ = build_feature_matrix(df_raw)
    X         = X_full[selected_features]   # align to model's feature set

    # Compute SHAP
    explainer, shap_values, expected_value = compute_shap_values(model, X)

    # Generate all plots
    logger.info("\n── Generating SHAP charts ──")
    plot_global_importance(shap_values, X)
    plot_beeswarm(shap_values, X)
    plot_historical_predictions(model, X, y)
    plot_waterfall(explainer, shap_values, X, year=2023)
    plot_waterfall(explainer, shap_values, X, year=2020)  # COVID year

    # Export JSON
    summary = export_shap_json(shap_values, X, y, model)

    logger.info("\n── SHAP complete. Artefacts in data/processed/ ──")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    summary = explain(mode="seed")

    print("\n" + "=" * 65)
    print("GLOBAL FEATURE IMPORTANCE (Mean |SHAP|)")
    print("=" * 65)
    for feat, val in list(summary["global_feature_importance"].items())[:10]:
        print(f"  {feat:<45} {val:.6f}")

    print("\nMost Recent Year Explanation (2023):")
    last = summary["per_year_explanations"][-1]
    print(f"  Year:          {last['year']}")
    print(f"  Stress prob:   {last['stress_prob']:.4f}")
    print(f"  Actual stress: {last['actual_stress']}")
    print(f"  Top 3 drivers: {last['top_3_drivers']}")

    print("\nCharts saved to: data/processed/shap_*.png")
