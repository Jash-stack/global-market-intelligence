"""
Cleaning module — normalises, deduplicates, interpolates, and validates
raw OHLCV data, producing cy_clean.parquet.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[1]
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

PRICE_COLS  = ["open", "high", "low", "close"]
OHLCV_COLS  = PRICE_COLS + ["volume"]


# ─────────────────────────────────────────────────────────────────────────────
# Step-by-step cleaning functions
# ─────────────────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    path = RAW_DIR / "ohlcv_raw.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    log.info("Loaded raw OHLCV: %d rows, %d tickers", len(df), df["ticker"].nunique())
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"]).copy()
    log.info("Duplicates removed: %d → %d rows", before, len(df))
    return df


def enforce_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex each ticker to a complete business-day calendar and forward-fill gaps."""
    all_dates = pd.bdate_range(df["date"].min(), df["date"].max())
    frames = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.set_index("date").reindex(all_dates)
        grp["ticker"] = ticker
        grp["sector"] = grp["sector"].ffill()
        grp[PRICE_COLS] = grp[PRICE_COLS].interpolate(method="time")
        grp["volume"]   = grp["volume"].fillna(0).astype(np.int64)
        frames.append(grp.reset_index().rename(columns={"index": "date"}))
    df_out = pd.concat(frames, ignore_index=True)
    log.info("Reindexed to trading calendar: %d rows", len(df_out))
    return df_out


def fix_ohlc_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure high >= close >= low and open within [low, high]."""
    df = df.copy()
    df["high"]  = df[["high",  "close", "open"]].max(axis=1)
    df["low"]   = df[["low",   "close", "open"]].min(axis=1)
    df["open"]  = df["open"].clip(lower=df["low"], upper=df["high"])
    return df


def remove_zero_price(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["close"] > 0].copy()
    log.info("Zero-price rows removed: %d → %d", before, len(df))
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["daily_return"]   = df.groupby("ticker")["close"].pct_change()
    df["log_return"]     = np.log(df.groupby("ticker")["close"].transform(lambda x: x / x.shift(1)))
    df["price_range"]    = df["high"] - df["low"]
    df["dollar_volume"]  = df["close"] * df["volume"]
    df["year"]           = df["date"].dt.year
    df["month"]          = df["date"].dt.month
    df["week"]           = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"]    = df["date"].dt.dayofweek
    return df


def clip_extreme_returns(df: pd.DataFrame, z_thresh: float = 6.0) -> pd.DataFrame:
    """Cap log-returns beyond z_thresh standard deviations (circuit-breaker level)."""
    df = df.copy()
    mu  = df["log_return"].mean()
    std = df["log_return"].std()
    lo, hi = mu - z_thresh * std, mu + z_thresh * std
    df["return_clipped"] = df["log_return"].clip(lo, hi)
    return df


def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """20-day & 60-day rolling volatility (annualised)."""
    df = df.sort_values(["ticker", "date"]).copy()
    for w in [20, 60]:
        df[f"vol_{w}d"] = (
            df.groupby("ticker")["log_return"]
            .transform(lambda x: x.rolling(w, min_periods=w // 2).std() * np.sqrt(252))
        )
    return df


def quality_report(df: pd.DataFrame) -> None:
    nulls = df[OHLCV_COLS].isnull().sum()
    log.info("Null counts after cleaning:\n%s", nulls.to_string())
    log.info("Date range: %s → %s", df["date"].min().date(), df["date"].max().date())
    log.info("Tickers: %d | Total rows: %d", df["ticker"].nunique(), len(df))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_cleaning() -> pd.DataFrame:
    log.info("=== CLEANING START ===")
    df = load_raw()
    df = drop_duplicates(df)
    df = enforce_trading_days(df)
    df = fix_ohlc_consistency(df)
    df = remove_zero_price(df)
    df = add_derived_columns(df)
    df = clip_extreme_returns(df)
    df = compute_rolling_stats(df)
    quality_report(df)

    out = PROC_DIR / "cy_clean.parquet"
    df.to_parquet(out, index=False)
    log.info("Saved clean data → %s", out)
    log.info("=== CLEANING COMPLETE ===")
    return df


if __name__ == "__main__":
    run_cleaning()
