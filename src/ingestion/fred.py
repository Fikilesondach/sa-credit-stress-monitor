"""
fred.py
-------
Pulls global risk proxy indicators from FRED (Federal Reserve Bank of St. Louis).
Requires a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html

Set your key in .env:
    FRED_API_KEY=your_key_here

Series used:
  - VIXCLS     : CBOE Volatility Index (VIX) — daily, averaged to annual
  - DTWEXBGS   : USD broad trade-weighted index — daily, averaged to annual
  - INTDSRZAM193N : SA discount rate (SARB repo rate proxy via IMF/FRED)

Run locally:   python -m src.ingestion.fred
"""

import os
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

# FRED series codes → column names
FRED_SERIES = {
    "VIXCLS":          "vix_avg",          # CBOE VIX (fear index)
    "DTWEXBGS":        "usd_broad_index",   # USD broad trade-weighted index
    "INTDSRZAM193N":   "sarb_repo_fred",    # SA discount rate (SARB proxy via IMF)
}

START_DATE = "2000-01-01"


def fetch_fred_data(
    api_key:    Optional[str] = None,
    start_date: str = START_DATE,
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch FRED series and aggregate to annual frequency.

    Parameters
    ----------
    api_key : str, optional
        FRED API key. Falls back to FRED_API_KEY env variable.
    start_date : str
        ISO date string (YYYY-MM-DD).
    end_date : str, optional
        ISO date string (YYYY-MM-DD). Defaults to today.

    Returns
    -------
    pd.DataFrame
        Annual index (datetime, year-end), one column per FRED series.
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("Install fredapi: pip install fredapi")

    api_key = api_key or os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Get a free key at: "
            "https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then add it to your .env file: FRED_API_KEY=your_key_here"
        )

    fred = Fred(api_key=api_key)
    annual_series = {}

    for series_id, col_name in FRED_SERIES.items():
        logger.info(f"  Fetching {col_name} ({series_id})...")
        try:
            raw = fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
            # Resample daily → annual mean
            annual = raw.resample("YE").mean()
            annual.index = annual.index.to_period("Y").to_timestamp("Y")
            annual_series[col_name] = annual
        except Exception as e:
            logger.warning(f"  Failed to fetch {series_id}: {e}")

    df = pd.DataFrame(annual_series)
    df.index.name = "date"
    df = df[df.index.year >= int(start_date[:4])]
    df = df.sort_index()

    logger.info(f"FRED pull complete. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from dotenv import load_dotenv
    load_dotenv()
    print("Pulling FRED data...")
    df = fetch_fred_data()
    print(df.tail(10).to_string())
