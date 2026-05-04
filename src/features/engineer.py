"""
engineer.py
-----------
Feature engineering for the SA Credit Stress Early Warning System.

Transforms the raw annual macro dataset into a rich model-ready feature matrix.

Transformations applied:
  1. Lag features       — t-1 and t-2 values of key indicators
                          (early warning: stress is often preceded by deterioration)
  2. Rolling statistics — 2yr and 3yr rolling mean and std of each indicator
                          (captures regime persistence vs volatility)
  3. Interaction terms  — economically motivated cross-features
                          (e.g. real rate × NPL, fiscal deficit × unemployment)
  4. Stress momentum    — lagged stress label (was last year a stress year?)
  5. WoE encoding       — Weight of Evidence on top features using optbinning
                          (Fikile's existing toolkit — converts continuous vars to
                          monotonic credit-score-style predictors)
  6. Scaling            — StandardScaler for tree models (XGBoost doesn't need it
                          strictly, but needed for any downstream logistic baselines)

Run:
    PYTHONPATH=. python -m src.features.engineer
"""

import logging
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Features to generate lags and rolling stats for
CORE_FEATURES = [
    "gdp_growth",
    "cpi_inflation",
    "unemployment_rate",
    "repo_rate",
    "npl_ratio",
    "private_credit_pct_gdp",
    "govt_debt_pct_gdp",
    "current_account_pct_gdp",
    "zar_usd",
    "vix_avg",
    "real_rate",
    "npl_yoy_chg",
    "credit_growth_gap",
    "zar_yoy_depr",
]

# Features to apply WoE encoding to (top predictors, continuous)
WOE_FEATURES = [
    "npl_ratio",
    "gdp_growth",
    "unemployment_rate",
    "real_rate",
    "govt_debt_pct_gdp",
    "zar_yoy_depr",
    "vix_avg",
]

TARGET = "credit_stress"


# ─── 1. Lag features ──────────────────────────────────────────────────────────

def _add_lags(df: pd.DataFrame, lags: list = [1, 2]) -> pd.DataFrame:
    """Add t-k lagged versions of CORE_FEATURES."""
    new_cols = {}
    for col in CORE_FEATURES:
        if col not in df.columns:
            continue
        for k in lags:
            new_cols[f"{col}_lag{k}"] = df[col].shift(k)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ─── 2. Rolling statistics ────────────────────────────────────────────────────

def _add_rolling(df: pd.DataFrame, windows: list = [2, 3]) -> pd.DataFrame:
    """Add rolling mean and std for CORE_FEATURES."""
    new_cols = {}
    for col in CORE_FEATURES:
        if col not in df.columns:
            continue
        for w in windows:
            # min_periods=1 avoids NaN for first window
            new_cols[f"{col}_roll{w}m"] = df[col].rolling(w, min_periods=1).mean()
            new_cols[f"{col}_roll{w}std"] = df[col].rolling(w, min_periods=1).std().fillna(0)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ─── 3. Interaction terms ─────────────────────────────────────────────────────

def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Economically motivated interaction features.

    These encode joint deterioration — two bad things happening simultaneously
    is more than additive in credit stress contexts.
    """
    interactions = {}

    # Monetary tightening into weak growth
    if {"real_rate", "gdp_growth"}.issubset(df.columns):
        interactions["real_rate_x_gdp"] = df["real_rate"] * df["gdp_growth"]

    # Banking stress × labour market
    if {"npl_ratio", "unemployment_rate"}.issubset(df.columns):
        interactions["npl_x_unemp"] = df["npl_ratio"] * df["unemployment_rate"]

    # Fiscal stress × external vulnerability
    if {"govt_debt_pct_gdp", "current_account_pct_gdp"}.issubset(df.columns):
        interactions["fiscal_x_ca"] = df["govt_debt_pct_gdp"] * (-df["current_account_pct_gdp"])

    # Currency weakness × global risk
    if {"zar_yoy_depr", "vix_avg"}.issubset(df.columns):
        interactions["zar_x_vix"] = df["zar_yoy_depr"] * df["vix_avg"]

    # Credit impulse × NPL momentum (credit expansion into deteriorating loans)
    if {"credit_growth_gap", "npl_yoy_chg"}.issubset(df.columns):
        interactions["credit_x_npl_mom"] = df["credit_growth_gap"] * df["npl_yoy_chg"]

    # Inflation × unemployment (misery index)
    if {"cpi_inflation", "unemployment_rate"}.issubset(df.columns):
        interactions["misery_index"] = df["cpi_inflation"] + df["unemployment_rate"]

    return pd.concat([df, pd.DataFrame(interactions, index=df.index)], axis=1)


# ─── 4. Stress momentum ───────────────────────────────────────────────────────

def _add_stress_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged stress label and cumulative stress count.

    Stress periods tend to be persistent — last year's stress is a strong
    predictor of this year's stress.
    """
    df = df.copy()
    df["stress_lag1"]       = df[TARGET].shift(1).fillna(0)
    df["stress_lag2"]       = df[TARGET].shift(2).fillna(0)
    df["stress_cumulative"] = df[TARGET].shift(1).expanding().sum().fillna(0)
    df["stress_run"]        = (
        df[TARGET].shift(1)
        .groupby((df[TARGET].shift(1) != df[TARGET].shift(1).shift(1)).cumsum())
        .cumcount()
        .fillna(0)
    )
    return df


# ─── 5. WoE encoding ──────────────────────────────────────────────────────────

def _add_woe(
    df: pd.DataFrame,
    target: pd.Series,
    woe_features: list = WOE_FEATURES,
    n_bins: int = 4,
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply Weight of Evidence (WoE) encoding using optbinning.

    WoE is standard in credit risk scorecards and ensures monotonic
    transformations that are interpretable alongside SHAP values.

    Returns
    -------
    df_woe : pd.DataFrame
        DataFrame with added _woe columns.
    woe_objects : dict
        Fitted OptimalBinning objects keyed by feature name
        (serialise these for production scoring).
    """
    try:
        from optbinning import OptimalBinning
    except ImportError:
        logger.warning("optbinning not installed — skipping WoE encoding. "
                       "pip install optbinning")
        return df, {}

    woe_objects = {}
    new_cols    = {}

    for col in woe_features:
        if col not in df.columns:
            logger.warning(f"WoE: column '{col}' not found — skipping.")
            continue

        x     = df[col].values
        y     = target.values
        valid = ~np.isnan(x) & ~np.isnan(y.astype(float))

        if valid.sum() < 8:
            logger.warning(f"WoE: insufficient data for '{col}' — skipping.")
            continue

        try:
            optb = OptimalBinning(
                name=col,
                dtype="numerical",
                max_n_bins=n_bins,
                min_bin_size=0.15,   # at least 15% of sample per bin
                monotonic_trend="auto",
            )
            optb.fit(x[valid], y[valid])
            woe_vals = np.full(len(df), np.nan)
            woe_vals[valid] = optb.transform(x[valid], metric="woe")
            new_cols[f"{col}_woe"] = woe_vals
            woe_objects[col]       = optb
            logger.info(f"  WoE encoded: {col} → {optb.n_bins} bins")
        except Exception as e:
            logger.warning(f"  WoE failed for '{col}': {e}")

    df_woe = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df_woe, woe_objects


# ─── 6. Master engineer function ──────────────────────────────────────────────

def build_feature_matrix(
    df_raw: pd.DataFrame,
    scale:  bool = False,
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw macro dataset from ingestion pipeline (output of load_macro_dataset).
    scale : bool
        If True, StandardScaler is applied to all numeric features.
        Not required for XGBoost, but useful for logistic baselines.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix, ready for model training.
    y : pd.Series
        Binary credit stress target.
    meta : dict
        WoE objects, feature list, and scaler (for production scoring).
    """
    logger.info("Starting feature engineering...")
    df = df_raw.copy()

    # Steps 1–4
    df = _add_lags(df)
    df = _add_rolling(df)
    df = _add_interactions(df)
    df = _add_stress_momentum(df)

    # Step 5 — WoE (fitted on full data; in production fit on train only)
    y_full    = df[TARGET]
    df, woe_objects = _add_woe(df, y_full)

    # Drop rows with NaN targets
    df = df.dropna(subset=[TARGET])
    y  = df[TARGET].astype(int)

    # Drop the raw target + any completely empty columns
    X = df.drop(columns=[TARGET])
    X = X.dropna(axis=1, how="all")

    # Drop first 2 rows (NaN-heavy due to lags) only if they have >50% nulls
    null_pct = X.isnull().mean(axis=1)
    X = X[null_pct <= 0.5]
    y = y[X.index]

    # Forward-fill any remaining NaNs (conservative: carry last known value)
    X = X.ffill().bfill()

    # Optional scaling
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            index=X.index,
            columns=X.columns,
        )
        X = X_scaled

    logger.info(
        f"Feature matrix built | shape={X.shape} | "
        f"stress_balance={y.value_counts().to_dict()}"
    )
    logger.info(f"Features ({len(X.columns)}): {list(X.columns)}")

    meta = {
        "woe_objects":   woe_objects,
        "feature_names": list(X.columns),
        "scaler":        scaler,
        "target":        TARGET,
    }
    return X, y, meta


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.ingestion.pipeline import load_macro_dataset

    df_raw = load_macro_dataset(mode="seed")
    X, y, meta = build_feature_matrix(df_raw)

    print("\n" + "=" * 65)
    print("FEATURE MATRIX SUMMARY")
    print("=" * 65)
    print(f"Shape:           {X.shape}")
    print(f"Target balance:  {y.value_counts().to_dict()}")
    print(f"WoE features:    {list(meta['woe_objects'].keys())}")
    print(f"\nFirst 5 columns preview:")
    print(X.iloc[:, :5].tail(6).to_string())
    print(f"\nAll features ({len(X.columns)}):")
    for i, f in enumerate(X.columns, 1):
        print(f"  {i:3d}. {f}")
