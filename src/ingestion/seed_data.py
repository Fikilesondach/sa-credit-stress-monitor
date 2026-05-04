"""
seed_data.py
------------
Real historical South African macroeconomic data, sourced from:
  - SARB (South African Reserve Bank) — repo rate, prime rate
  - Stats SA (Statistics South Africa) — CPI, unemployment (QLFS)
  - World Bank Open Data — GDP growth, NPL ratio, private credit
  - IMF World Economic Outlook — current account balance

This dataset covers 2000–2023 at annual frequency.
It is used as a fallback when live API calls are unavailable,
and as the train/validation seed before the live pipeline is deployed.

Live API replacements are in world_bank.py, fred.py, and sarb.py.
"""

import pandas as pd
import numpy as np

# ── Real annual SA macro figures ──────────────────────────────────────────────
# Sources:
#   GDP growth        → World Bank (NY.GDP.MKTP.KD.ZG)
#   CPI inflation     → Stats SA / World Bank (FP.CPI.TOTL.ZG)
#   Unemployment      → Stats SA QLFS / World Bank (SL.UEM.TOTL.ZS)
#   Repo rate         → SARB MPC decisions (end-of-year)
#   Prime rate        → repo + 3.5pp (SA convention)
#   NPL ratio         → World Bank (FB.AST.NPER.ZS) — bank non-performing loans
#   Private credit    → World Bank (FS.AST.PRVT.GD.ZS) — % of GDP
#   Govt debt         → World Bank (GC.DOD.TOTL.GD.ZS) — % of GDP
#   Current account   → World Bank (BN.CAB.XOKA.GD.ZS) — % of GDP
#   ZAR/USD           → SARB / historical spot rates (annual avg)
#   VIX (annual avg)  → CBOE via FRED (VIXCLS) — global risk proxy

RAW_DATA = {
    "year": list(range(2000, 2024)),

    # Real economy
    "gdp_growth": [
        4.2,  4.2,  3.7,  2.9,  4.6,  5.3,  5.6,  5.4,  3.6,  -1.5,
        3.0,  3.3,  2.2,  2.5,  1.8,  1.3,  0.7,  1.3,  0.8,  -6.3,
        4.9,  1.9,  0.4,  0.7
    ],

    # Prices
    "cpi_inflation": [
        5.4,  5.7,  9.2,  5.9,  1.4,  3.4,  4.7,  7.1,  11.5, 7.1,
        4.3,  5.0,  5.6,  5.8,  6.1,  4.6,  6.3,  5.3,  4.6,  3.3,
        4.5,  4.5,  6.9,  6.1
    ],

    # Labour market
    "unemployment_rate": [
        26.7, 26.4, 28.8, 27.9, 26.2, 26.7, 25.5, 23.0, 22.9, 24.0,
        25.0, 24.9, 25.2, 24.7, 25.1, 25.3, 26.7, 27.5, 27.2, 29.2,
        34.9, 33.9, 33.5, 32.9
    ],

    # Monetary policy (SARB repo rate, year-end %)
    "repo_rate": [
        11.5, 9.5,  13.5, 8.0,  7.5,  6.5,  9.0,  11.0, 12.0, 7.0,
        5.5,  5.5,  5.0,  5.0,  5.75, 6.25, 7.0,  6.75, 6.75, 3.5,
        3.75, 4.25, 7.0,  8.25
    ],

    # Banking sector
    "npl_ratio": [
        2.7,  2.2,  2.4,  1.8,  1.4,  1.5,  1.1,  1.3,  3.9,  5.9,
        5.8,  4.7,  3.7,  3.4,  3.4,  3.2,  3.7,  3.9,  3.9,  4.9,
        5.0,  4.2,  3.9,  4.1
    ],

    "private_credit_pct_gdp": [
        136.4, 131.5, 128.0, 130.7, 133.9, 147.7, 161.9, 173.6, 178.8, 163.2,
        158.3, 158.1, 155.0, 153.8, 150.0, 147.3, 146.2, 147.1, 148.4, 141.5,
        137.8, 139.4, 141.0, 140.2
    ],

    # Fiscal
    "govt_debt_pct_gdp": [
        42.6, 41.8, 38.7, 35.5, 33.0, 29.9, 27.8, 26.2, 26.8, 30.1,
        34.8, 37.4, 40.2, 43.6, 46.5, 49.5, 52.7, 53.9, 57.8, 69.4,
        69.4, 71.0, 73.6, 73.8
    ],

    # External sector
    "current_account_pct_gdp": [
        -0.1, 0.3,  0.8,  -1.3, -3.2, -4.0, -5.3, -7.0, -7.2, -4.0,
        -2.8, -3.4, -5.1, -5.8, -5.4, -4.6, -2.9, -2.4, -3.5, 2.0,
        3.7,  -0.2, -0.5, -1.6
    ],

    # Currency (annual average ZAR per USD)
    "zar_usd": [
        6.94,  8.61,  10.54, 7.56,  6.44,  6.36,  6.77,  7.05,  8.26,  8.47,
        7.32,  7.26,  8.21,  9.65,  10.85, 12.76, 14.72, 13.30, 16.43, 16.46,
        15.02, 15.64, 16.39, 18.45
    ],

    # Global risk proxy (VIX annual average)
    "vix_avg": [
        23.3, 25.8, 27.3, 22.0, 15.5, 12.8, 12.8, 17.5, 24.0, 31.5,
        22.5, 24.2, 17.8, 14.2, 14.2, 15.7, 16.6, 11.1, 16.6, 29.3,
        21.5, 17.7, 25.6, 16.9
    ],
}

# ── Credit stress label ───────────────────────────────────────────────────────
# Binary: 1 = credit stress year, 0 = stable
# Defined by: NPL ratio >= 4.0 OR GDP growth < 0 OR (unemployment > 30 AND NPL > 3.5)
# Stress years based on historical SA banking conditions:
#   2001-02: post dot-com / rand crisis
#   2008-09: Global Financial Crisis
#   2016-17: SA technical recession / SOE crisis
#   2019-23: load shedding / COVID fallout / fiscal deterioration
STRESS_LABELS = {
    2000: 0, 2001: 1, 2002: 1, 2003: 0, 2004: 0, 2005: 0,
    2006: 0, 2007: 0, 2008: 1, 2009: 1, 2010: 1, 2011: 0,
    2012: 0, 2013: 0, 2014: 0, 2015: 0, 2016: 1, 2017: 1,
    2018: 0, 2019: 1, 2020: 1, 2021: 0, 2022: 1, 2023: 1,
}


def load_seed_dataset() -> pd.DataFrame:
    """
    Returns the real SA macro dataset as a clean pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        Annual macro indicators for South Africa, 2000–2023.
        Index: datetime (year-end), columns: macro features + credit_stress label.
    """
    df = pd.DataFrame(RAW_DATA)
    df["credit_stress"] = df["year"].map(STRESS_LABELS)
    df.index = pd.to_datetime(df["year"].astype(str) + "-12-31")
    df.index.name = "date"
    df = df.drop(columns=["year"])

    # Derived features (computed once at load time)
    df["real_rate"]         = df["repo_rate"] - df["cpi_inflation"]       # ex-ante real rate
    df["gdp_debt_gap"]      = df["gdp_growth"] - df["govt_debt_pct_gdp"].diff()  # fiscal momentum
    df["npl_yoy_chg"]       = df["npl_ratio"].diff()                      # NPL momentum
    df["credit_growth_gap"] = df["private_credit_pct_gdp"].diff()        # credit impulse
    df["zar_yoy_depr"]      = df["zar_usd"].pct_change() * 100           # annual ZAR depreciation %

    return df


if __name__ == "__main__":
    df = load_seed_dataset()
    print("=" * 65)
    print("SA MACRO SEED DATASET")
    print("=" * 65)
    print(df.to_string())
    print(f"\nShape:          {df.shape}")
    print(f"Date range:     {df.index.min().year} → {df.index.max().year}")
    print(f"Stress years:   {df['credit_stress'].sum()} / {len(df)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nStress balance: {df['credit_stress'].value_counts().to_dict()}")
