"""
pipeline.py
-----------
Master ingestion orchestrator for the SA Credit Stress Monitor.

Priority order for each data source:
  1. Live API pull (World Bank, FRED, yfinance)
  2. Fallback to seed_data.py (real historical figures, always available)

This means the pipeline ALWAYS produces a valid dataset, even with no internet
or missing API keys. Live data overrides seed columns where available.

Usage
-----
    # Full live pull (requires FRED_API_KEY in .env):
    from src.ingestion.pipeline import load_macro_dataset
    df = load_macro_dataset(mode="live")

    # Seed-only (no API keys needed):
    df = load_macro_dataset(mode="seed")

    # Auto (tries live, falls back to seed per source):
    df = load_macro_dataset(mode="auto")

CLI:
    python -m src.ingestion.pipeline --mode auto --save
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv

from src.ingestion.seed_data import load_seed_dataset

logger = logging.getLogger(__name__)
load_dotenv()

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_macro_dataset(
    mode: Literal["live", "seed", "auto"] = "auto",
    save: bool = False,
) -> pd.DataFrame:
    """
    Load the SA macroeconomic dataset.

    Parameters
    ----------
    mode : "live" | "seed" | "auto"
        - "live"  → pulls from World Bank, FRED, yfinance only (fails if unavailable)
        - "seed"  → uses embedded real historical data only
        - "auto"  → tries live sources, falls back to seed per source gracefully
    save : bool
        If True, saves the resulting DataFrame to data/raw/sa_macro_dataset.parquet

    Returns
    -------
    pd.DataFrame
        Annual macro dataset ready for feature engineering.
    """
    logger.info(f"Loading macro dataset (mode={mode})")

    # ── 1. Always start with the seed (guaranteed baseline) ──────────────────
    seed_df = load_seed_dataset()

    if mode == "seed":
        logger.info("Seed-only mode. Using embedded historical data.")
        df = seed_df
    else:
        # ── 2. Attempt live pulls and merge ──────────────────────────────────
        live_frames = []

        # World Bank
        try:
            from src.ingestion.world_bank import fetch_world_bank_data
            wb_df = fetch_world_bank_data()
            live_frames.append(("World Bank", wb_df))
            logger.info("✓ World Bank data pulled successfully")
        except Exception as e:
            logger.warning(f"✗ World Bank pull failed: {e}")
            if mode == "live":
                raise

        # FRED
        if os.getenv("FRED_API_KEY"):
            try:
                from src.ingestion.fred import fetch_fred_data
                fred_df = fetch_fred_data()
                live_frames.append(("FRED", fred_df))
                logger.info("✓ FRED data pulled successfully")
            except Exception as e:
                logger.warning(f"✗ FRED pull failed: {e}")
                if mode == "live":
                    raise
        else:
            logger.warning("FRED_API_KEY not set — skipping FRED pull. "
                           "Get a free key: https://fred.stlouisfed.org/docs/api/api_key.html")

        # Market data (yfinance)
        try:
            from src.ingestion.market import fetch_market_data
            mkt_df = fetch_market_data()
            live_frames.append(("Market", mkt_df))
            logger.info("✓ Market data pulled successfully")
        except Exception as e:
            logger.warning(f"✗ Market data pull failed: {e}")
            if mode == "live":
                raise

        # ── 3. Merge live onto seed (live wins where both exist) ─────────────
        df = seed_df.copy()
        for source_name, live_df in live_frames:
            # Align on year
            live_df.index = pd.to_datetime(
                live_df.index.year.astype(str) + "-12-31"
            )
            for col in live_df.columns:
                if col in df.columns:
                    df[col].update(live_df[col])   # overwrite with live values
                    logger.info(f"  Refreshed '{col}' from {source_name}")
                else:
                    df[col] = live_df[col]         # add new column
                    logger.info(f"  Added new column '{col}' from {source_name}")

    # ── 4. Final quality checks ───────────────────────────────────────────────
    _validate_dataset(df)

    # ── 5. Optionally save ────────────────────────────────────────────────────
    if save:
        out_path = OUTPUT_DIR / "sa_macro_dataset.parquet"
        df.to_parquet(out_path, index=True)
        logger.info(f"Dataset saved to {out_path}")

    return df


def _validate_dataset(df: pd.DataFrame) -> None:
    """Run basic sanity checks on the assembled dataset."""
    required_cols = [
        "gdp_growth", "cpi_inflation", "unemployment_rate",
        "repo_rate", "npl_ratio", "credit_stress",
    ]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Dataset missing required columns: {missing_required}")

    if df["credit_stress"].isnull().any():
        raise ValueError("credit_stress label contains nulls — check seed_data.py")

    high_null_cols = [
        c for c in df.columns
        if df[c].isnull().mean() > 0.4 and c != "credit_stress"
    ]
    if high_null_cols:
        logger.warning(f"Columns with >40% nulls: {high_null_cols}")

    logger.info(
        f"Dataset validated | shape={df.shape} | "
        f"stress_years={int(df['credit_stress'].sum())} / {len(df)}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="SA Macro Data Ingestion Pipeline")
    parser.add_argument(
        "--mode",
        choices=["live", "seed", "auto"],
        default="auto",
        help="Data sourcing mode",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output to data/raw/sa_macro_dataset.parquet",
    )
    args = parser.parse_args()

    df = load_macro_dataset(mode=args.mode, save=args.save)

    print("\n" + "=" * 65)
    print("FINAL DATASET SUMMARY")
    print("=" * 65)
    print(df.tail(8).to_string())
    print(f"\nShape:        {df.shape}")
    print(f"Date range:   {df.index.min().year} → {df.index.max().year}")
    print(f"Stress years: {int(df['credit_stress'].sum())} / {len(df)}")
    print(f"\nColumns:\n{list(df.columns)}")
