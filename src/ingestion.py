"""
Ingestion module — fetches OHLCV + fundamentals from Yahoo Finance (RapidAPI)
and persists raw + aggregated parquet files.

RapidAPI product : yh-finance (host: yh-finance.p.rapidapi.com)
Docs            : https://rapidapi.com/apidojo/api/yh-finance
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
RAW_DIR    = ROOT / "data" / "raw"
PROC_DIR   = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── API config ─────────────────────────────────────────────────────────────
API_KEY    = os.getenv("RAPIDAPI_KEY", "")
API_HOST   = "yh-finance.p.rapidapi.com"
BASE_URL   = f"https://{API_HOST}"
HEADERS    = {"X-RapidAPI-Key": API_KEY, "X-RapidAPI-Host": API_HOST}

# ── universe ───────────────────────────────────────────────────────────────
UNIVERSE = {
    "Technology":   ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMD"],
    "Finance":      ["JPM",  "BAC",  "GS",    "MS",   "WFC",  "BLK"],
    "Healthcare":   ["JNJ",  "PFE",  "UNH",   "ABBV", "MRK",  "BMY"],
    "Energy":       ["XOM",  "CVX",  "COP",   "SLB",  "EOG",  "PSX"],
    "Consumer":     ["WMT",  "COST", "PG",    "KO",   "PEP",  "MCD"],
    "Industrials":  ["CAT",  "BA",   "HON",   "UPS",  "GE",   "MMM"],
}
ALL_TICKERS = [t for tickers in UNIVERSE.values() for t in tickers]
SECTOR_MAP  = {t: s for s, tickers in UNIVERSE.items() for t in tickers}

INTERVALS    = {"1d": "1d"}
RANGE_PERIOD = "5y"


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict, retries: int = 3) -> dict | None:
    """GET with exponential-backoff retry."""
    if not API_KEY:
        return None
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 2 ** attempt
                log.warning("Rate-limited — sleeping %ss", wait)
                time.sleep(wait)
            else:
                log.error("HTTP %s for %s", resp.status_code, url)
                return None
        except requests.RequestException as exc:
            log.warning("Request error (attempt %d): %s", attempt + 1, exc)
            time.sleep(2 ** attempt)
    return None


def fetch_historical(ticker: str) -> pd.DataFrame | None:
    """Fetch 5-year daily OHLCV for one ticker."""
    data = _get("/stock/v3/get-historical-data", {"symbol": ticker, "region": "US"})
    if not data:
        return None
    prices = data.get("prices", [])
    if not prices:
        return None
    df = pd.DataFrame(prices)
    df = df[df["type"].isna()].copy() if "type" in df.columns else df.copy()
    df["date"]   = pd.to_datetime(df["date"], unit="s")
    df["ticker"] = ticker
    df["sector"] = SECTOR_MAP.get(ticker, "Unknown")
    keep = ["date", "ticker", "sector", "open", "high", "low", "close", "volume"]
    df   = df[[c for c in keep if c in df.columns]]
    return df.sort_values("date").reset_index(drop=True)


def fetch_fundamentals(ticker: str) -> dict | None:
    """Fetch key fundamentals (P/E, market cap, EPS, beta…)."""
    data = _get("/stock/v2/get-summary", {"symbol": ticker, "region": "US"})
    if not data:
        return None
    price   = data.get("price", {})
    summary = data.get("summaryDetail", {})
    stats   = data.get("defaultKeyStatistics", {})
    return {
        "ticker":         ticker,
        "sector":         SECTOR_MAP.get(ticker, "Unknown"),
        "market_cap_bn":  price.get("marketCap", {}).get("raw", np.nan) / 1e9
                          if price.get("marketCap") else np.nan,
        "pe_ratio":       summary.get("trailingPE", {}).get("raw", np.nan),
        "forward_pe":     summary.get("forwardPE",  {}).get("raw", np.nan),
        "eps_ttm":        stats.get("trailingEps",  {}).get("raw", np.nan),
        "beta":           summary.get("beta",        {}).get("raw", np.nan),
        "dividend_yield": summary.get("dividendYield", {}).get("raw", np.nan),
        "52w_high":       summary.get("fiftyTwoWeekHigh", {}).get("raw", np.nan),
        "52w_low":        summary.get("fiftyTwoWeekLow",  {}).get("raw", np.nan),
        "avg_volume":     summary.get("averageVolume",    {}).get("raw", np.nan),
        "name":           price.get("longName", ticker),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback (runs when no API key is present)
# ─────────────────────────────────────────────────────────────────────────────

def _seed_prices() -> dict[str, float]:
    return {
        "AAPL":200, "MSFT":380, "GOOGL":170, "NVDA":880, "META":480, "AMD":170,
        "JPM": 200, "BAC":  38, "GS":   480, "MS":  100, "WFC":  60, "BLK":810,
        "JNJ": 155, "PFE":  27, "UNH":  520, "ABBV":175, "MRK": 130, "BMY": 50,
        "XOM": 115, "CVX":  155,"COP":  120, "SLB":  50, "EOG": 125, "PSX":155,
        "WMT": 175, "COST": 720,"PG":   160, "KO":   60, "PEP": 170, "MCD":300,
        "CAT": 370, "BA":   180,"HON":  200, "UPS":  140,"GE":  160, "MMM":100,
    }


def _generate_synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate realistic GBM price paths for all tickers (5 years of daily data)."""
    log.warning("No RAPIDAPI_KEY found — generating synthetic data for demo.")
    rng      = np.random.default_rng(42)
    dates    = pd.bdate_range("2020-01-01", "2024-12-31")
    n        = len(dates)
    seed_px  = _seed_prices()

    rows, fund_rows = [], []
    for ticker, s0 in seed_px.items():
        sector = SECTOR_MAP.get(ticker, "Unknown")
        mu     = rng.uniform(0.0001, 0.0006)
        sigma  = rng.uniform(0.012, 0.028)
        shocks = rng.standard_normal(n)
        log_r  = (mu - 0.5 * sigma**2) + sigma * shocks
        prices = s0 * np.exp(np.cumsum(log_r))
        hi     = prices * (1 + rng.uniform(0.002, 0.015, n))
        lo     = prices * (1 - rng.uniform(0.002, 0.015, n))
        op     = prices * (1 + rng.uniform(-0.005, 0.005, n))
        vol    = rng.integers(3_000_000, 80_000_000, n)

        df_t = pd.DataFrame({
            "date":   dates, "ticker": ticker, "sector": sector,
            "open":   np.round(op, 2),  "high":  np.round(hi, 2),
            "low":    np.round(lo, 2),  "close": np.round(prices, 2),
            "volume": vol,
        })
        rows.append(df_t)

        fund_rows.append({
            "ticker":         ticker,
            "sector":         sector,
            "name":           ticker,
            "market_cap_bn":  round(prices[-1] * rng.integers(1_000_000, 10_000_000_000) / 1e9, 2),
            "pe_ratio":       round(rng.uniform(10, 45), 2),
            "forward_pe":     round(rng.uniform(8, 35), 2),
            "eps_ttm":        round(rng.uniform(1, 20), 2),
            "beta":           round(rng.uniform(0.5, 1.8), 2),
            "dividend_yield": round(rng.uniform(0, 0.05), 4),
            "52w_high":       round(prices[-252:].max(), 2),
            "52w_low":        round(prices[-252:].min(), 2),
            "avg_volume":     int(rng.integers(5_000_000, 60_000_000)),
        })

    ohlcv = pd.concat(rows, ignore_index=True)
    funds = pd.DataFrame(fund_rows)
    return ohlcv, funds


# ─────────────────────────────────────────────────────────────────────────────
# Aggregations
# ─────────────────────────────────────────────────────────────────────────────

def _build_sector_daily(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Value-weighted average close price by sector-date."""
    return (
        ohlcv.groupby(["date", "sector"])
        .agg(avg_close=("close", "mean"), total_volume=("volume", "sum"), count=("ticker", "nunique"))
        .reset_index()
    )


def _build_annual_returns(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Annual return % per ticker."""
    df = ohlcv.copy()
    df["year"] = df["date"].dt.year
    first = df.groupby(["ticker", "year"])["close"].first().rename("price_start")
    last  = df.groupby(["ticker", "year"])["close"].last().rename("price_end")
    ann   = pd.concat([first, last], axis=1).reset_index()
    ann["annual_return_pct"] = (ann["price_end"] / ann["price_start"] - 1) * 100
    ann["sector"] = ann["ticker"].map(SECTOR_MAP)
    return ann


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion() -> None:
    log.info("=== INGESTION START ===")

    # ── fetch / generate ──────────────────────────────────────────────────
    if API_KEY:
        log.info("API key found — fetching from Yahoo Finance RapidAPI")
        ohlcv_frames, fund_list = [], []
        for ticker in ALL_TICKERS:
            log.info("  fetching %s", ticker)
            df_t = fetch_historical(ticker)
            if df_t is not None:
                ohlcv_frames.append(df_t)
            fd = fetch_fundamentals(ticker)
            if fd:
                fund_list.append(fd)
            time.sleep(0.3)
        ohlcv = pd.concat(ohlcv_frames, ignore_index=True) if ohlcv_frames else pd.DataFrame()
        funds = pd.DataFrame(fund_list)
    else:
        ohlcv, funds = _generate_synthetic()

    if ohlcv.empty:
        log.error("No OHLCV data — aborting.")
        return

    # ── save raw ──────────────────────────────────────────────────────────
    ohlcv.to_parquet(RAW_DIR / "ohlcv_raw.parquet",        index=False)
    funds.to_parquet(RAW_DIR / "fundamentals_raw.parquet", index=False)
    log.info("Raw saved: %d rows OHLCV, %d tickers fundamentals", len(ohlcv), len(funds))

    # ── build & save aggregations ─────────────────────────────────────────
    sector_daily = _build_sector_daily(ohlcv)
    annual_ret   = _build_annual_returns(ohlcv)

    sector_daily.to_parquet(PROC_DIR / "sector_daily.parquet",  index=False)
    annual_ret.to_parquet(  PROC_DIR / "annual_returns.parquet", index=False)
    log.info("Processed aggregations saved.")
    log.info("=== INGESTION COMPLETE ===")


if __name__ == "__main__":
    run_ingestion()
