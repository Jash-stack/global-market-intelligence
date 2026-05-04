"""
Feature engineering — lag returns, rolling stats, CAGR, HHI concentration,
technical indicators (RSI, MACD, Bollinger Bands, ATR).
Saves cy_features.parquet.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"


# ─────────────────────────────────────────────────────────────────────────────
# Lag & rolling features
# ─────────────────────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, lags: list[int] = [1, 2, 3, 5, 10, 21]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"return_lag_{lag}d"] = df.groupby("ticker")["daily_return"].shift(lag)
        df[f"close_lag_{lag}d"]  = df.groupby("ticker")["close"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"roll_mean_{w}d"]   = df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}d"]    = df.groupby("ticker")["daily_return"].transform(
            lambda x: x.rolling(w, min_periods=1).std())
        df[f"roll_max_{w}d"]    = df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(w, min_periods=1).max())
        df[f"roll_min_{w}d"]    = df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(w, min_periods=1).min())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicators
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    df[f"rsi_{period}"] = df.groupby("ticker")["close"].transform(
        lambda x: _rsi(x, period))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df.groupby("ticker")["close"].transform(
        lambda x: x.ewm(span=fast, adjust=False).mean())
    ema_slow = df.groupby("ticker")["close"].transform(
        lambda x: x.ewm(span=slow, adjust=False).mean())
    df["macd_line"]    = ema_fast - ema_slow
    df["macd_signal"]  = df.groupby("ticker")["macd_line"].transform(
        lambda x: x.ewm(span=signal, adjust=False).mean())
    df["macd_hist"]    = df["macd_line"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    roll_mean = df.groupby("ticker")["close"].transform(
        lambda x: x.rolling(window, min_periods=1).mean())
    roll_std  = df.groupby("ticker")["close"].transform(
        lambda x: x.rolling(window, min_periods=1).std())
    df["bb_upper"]  = roll_mean + n_std * roll_std
    df["bb_lower"]  = roll_mean - n_std * roll_std
    df["bb_middle"] = roll_mean
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_pct"]    = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range — proxy for volatility."""
    df = df.copy()
    prev_close = df.groupby("ticker")["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = df.groupby("ticker").apply(
        lambda g: tr.loc[g.index].ewm(span=period, adjust=False).mean()
    ).reset_index(level=0, drop=True)
    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Price momentum: 1m, 3m, 6m, 12m (in trading days)."""
    df = df.copy()
    for days, label in [(21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")]:
        df[f"momentum_{label}"] = df.groupby("ticker")["close"].transform(
            lambda x: x.pct_change(days) * 100)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CAGR (per ticker, full period)
# ─────────────────────────────────────────────────────────────────────────────

def add_cagr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cagr_map = {}
    for ticker, grp in df.groupby("ticker"):
        grp = grp.dropna(subset=["close"]).sort_values("date")
        if len(grp) < 2:
            cagr_map[ticker] = np.nan
            continue
        years = (grp["date"].iloc[-1] - grp["date"].iloc[0]).days / 365.25
        if years <= 0 or grp["close"].iloc[0] <= 0:
            cagr_map[ticker] = np.nan
        else:
            cagr_map[ticker] = round(
                (grp["close"].iloc[-1] / grp["close"].iloc[0]) ** (1 / years) - 1, 4)
    df["cagr"] = df["ticker"].map(cagr_map)
    log.info("CAGR computed for %d tickers", len(cagr_map))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# HHI — sector-level volume concentration per day
# ─────────────────────────────────────────────────────────────────────────────

def add_hhi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Herfindahl-Hirschman Index of dollar-volume within each sector-date.
    HHI = sum(market_share²); high HHI → concentrated sector.
    """
    df = df.copy()
    sector_tot = df.groupby(["date", "sector"])["dollar_volume"].transform("sum")
    share       = (df["dollar_volume"] / sector_tot.replace(0, np.nan))
    df["hhi"]   = df.groupby(["date", "sector"])["dollar_volume"].transform(
        lambda x: ((x / (x.sum() if x.sum() > 0 else np.nan)) ** 2).sum()
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Volume features
# ─────────────────────────────────────────────────────────────────────────────

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["volume_ma20"]   = df.groupby("ticker")["volume"].transform(
        lambda x: x.rolling(20, min_periods=1).mean())
    df["rel_volume"]    = df["volume"] / df["volume_ma20"].replace(0, np.nan)
    df["volume_shock"]  = (df["rel_volume"] > 3.0).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_engineering() -> pd.DataFrame:
    log.info("=== FEATURE ENGINEERING START ===")
    df = pd.read_parquet(PROC_DIR / "cy_outliers.parquet")

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_momentum(df)
    df = add_cagr(df)
    df = add_hhi(df)
    df = add_volume_features(df)

    out = PROC_DIR / "cy_features.parquet"
    df.to_parquet(out, index=False)
    log.info("Feature matrix: %d rows × %d cols → saved %s", len(df), len(df.columns), out)
    log.info("=== FEATURE ENGINEERING COMPLETE ===")
    return df


if __name__ == "__main__":
    run_feature_engineering()
