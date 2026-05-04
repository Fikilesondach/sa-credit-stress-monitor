"""
world_bank.py
-------------
Pulls South African macroeconomic indicators from the World Bank Open Data API.
No API key required — completely free and public.

API docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

Run locally:   python -m src.ingestion.world_bank
"""

import requests
import pandas as pd
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

BASE_URL   = "https://api.worldbank.org/v2"
COUNTRY    = "ZA"          # ISO 3166-1 alpha-2 for South Africa
PER_PAGE   = 100
START_YEAR = 2000

# World Bank indicator codes → human-readable column names
INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",             # GDP growth (% annual)
    "FP.CPI.TOTL.ZG":    "cpi_inflation",           # CPI inflation (% annual)
    "SL.UEM.TOTL.ZS":    "unemployment_rate",       # Unemployment (% total labour force)
    "FR.INR.LEND":       "lending_rate",            # Lending interest rate (%)
    "FB.AST.NPER.ZS":    "npl_ratio",               # Bank non-performing loans (% gross loans)
    "FS.AST.PRVT.GD.ZS": "private_credit_pct_gdp",  # Domestic credit to private sector (% GDP)
    "GC.DOD.TOTL.GD.ZS": "govt_debt_pct_gdp",       # Central govt debt (% GDP)
    "BN.CAB.XOKA.GD.ZS": "current_account_pct_gdp", # Current account balance (% GDP)
    "PA.NUS.FCRF":       "zar_usd_wb",              # Official ZAR/USD exchange rate
}


def _fetch_indicator(indicator_code: str, retries: int = 3) -> dict:
    """
    Fetch a single World Bank indicator for South Africa.

    Returns a dict of {year (int): value (float)}.
    """
    url = (
        f"{BASE_URL}/country/{COUNTRY}/indicator/{indicator_code}"
        f"?format=json&per_page={PER_PAGE}&mrv=30"
    )
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            payload = resp.json()

            if not payload or len(payload) < 2 or not payload[1]:
                logger.warning(f"No data returned for {indicator_code}")
                return {}

            return {
                int(entry["date"]): (
                    float(entry["value"]) if entry["value"] is not None else None
                )
                for entry in payload[1]
                if entry.get("date", "").isdigit()
            }

        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {indicator_code}: {e}")
            time.sleep(2 ** attempt)  # exponential backoff

    logger.error(f"All retries exhausted for {indicator_code}")
    return {}


def fetch_world_bank_data(
    start_year: int = START_YEAR,
    end_year:   Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch all World Bank indicators for SA and return a tidy annual DataFrame.

    Parameters
    ----------
    start_year : int
        First year of data to include (default 2000).
    end_year : int, optional
        Last year of data to include (defaults to current year).

    Returns
    -------
    pd.DataFrame
        Annual index (datetime, year-end), one column per indicator.
    """
    import datetime
    end_year = end_year or datetime.date.today().year

    all_series = {}
    for code, col_name in INDICATORS.items():
        logger.info(f"  Fetching {col_name} ({code})...")
        series = _fetch_indicator(code)
        all_series[col_name] = series
        time.sleep(0.3)  # be polite to the API

    # Build DataFrame
    years = list(range(start_year, end_year + 1))
    df = pd.DataFrame(index=years)
    for col, series in all_series.items():
        df[col] = df.index.map(series)

    df.index = pd.to_datetime(df.index.astype(str) + "-12-31")
    df.index.name = "date"
    df = df.sort_index()

    logger.info(f"World Bank pull complete. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print("Pulling World Bank data for South Africa...")
    df = fetch_world_bank_data()
    print(df.tail(10).to_string())
    print(f"\nMissing values:\n{df.isnull().sum()}")
