"""
market.py
---------
Pulls market-based indicators via yfinance (no API key required).

Tickers:
  - ZAR=X    : ZAR/USD spot rate
  - ^J203.JO : JSE All Share Index (ALSI) — SA equity market proxy
  - JSE:GOVI : SA government bond ETF (GOVI) — sovereign yield proxy

Run locally:   python -m src.ingestion.market
"""

import logging
import pandas as pd
import yfinance as yf
from typing import Optional

logger = logging.getLogger(__name__)

TICKERS = {
    "ZAR=X":    "zar_usd",        # ZAR per 1 USD spot rate
    "^J203.JO": "jse_alsi",       # JSE All Share Index level
    "STXGOV.JO":"sa_bond_etf",    # SA govt bond ETF as yield proxy
}

START_DATE = "2000-01-01"


def fetch_market_data(
    start_date: str = START_DATE,
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Download market data and resample to annual frequency.

    Parameters
    ----------
    start_date : str
        ISO date string (YYYY-MM-DD).
    end_date : str, optional
        Defaults to today.

    Returns
    -------
    pd.DataFrame
        Annual index (datetime, year-end), market indicators as columns.
        Includes derived features: jse_annual_return, zar_annual_depr_pct.
    """
    import datetime
    end_date = end_date or str(datetime.date.today())

    frames = {}
    for ticker, col_name in TICKERS.items():
        logger.info(f"  Fetching {col_name} ({ticker})...")
        try:
            raw = yf.download(ticker, start=start_date, end=end_date,
                               progress=False, auto_adjust=True)
            if raw.empty:
                logger.warning(f"  No data for {ticker}")
                continue
            # Annual mean close price
            annual = raw["Close"].resample("YE").mean()
            annual.index = annual.index.to_period("Y").to_timestamp("Y")
            frames[col_name] = annual
        except Exception as e:
            logger.warning(f"  Failed to fetch {ticker}: {e}")

    if not frames:
        raise RuntimeError("No market data retrieved. Check internet connection.")

    df = pd.DataFrame(frames)
    df.index.name = "date"

    # Derived features
    if "jse_alsi" in df.columns:
        df["jse_annual_return"] = df["jse_alsi"].pct_change() * 100  # % annual return

    if "zar_usd" in df.columns:
        df["zar_annual_depr_pct"] = df["zar_usd"].pct_change() * 100  # % ZAR depreciation

    df = df[df.index.year >= int(start_date[:4])].sort_index()
    logger.info(f"Market data pull complete. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print("Pulling market data...")
    df = fetch_market_data()
    print(df.tail(10).to_string())
    print(f"\nMissing values:\n{df.isnull().sum()}")
