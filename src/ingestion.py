"""
Ingestion module — fetches live OHLCV + fundamentals from Yahoo Finance via yfinance.
No API key required.  Falls back to synthetic GBM data if network is unavailable.
"""

import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[1]
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = {
    "Technology":  ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMD"],
    "Finance":     ["JPM",  "BAC",  "GS",    "MS",   "WFC",  "BLK"],
    "Healthcare":  ["JNJ",  "PFE",  "UNH",   "ABBV", "MRK",  "BMY"],
    "Energy":      ["XOM",  "CVX",  "COP",   "SLB",  "EOG",  "PSX"],
    "Consumer":    ["WMT",  "COST", "PG",    "KO",   "PEP",  "MCD"],
    "Industrials": ["CAT",  "BA",   "HON",   "UPS",  "GE",   "MMM"],
}
ALL_TICKERS = [t for tickers in UNIVERSE.values() for t in tickers]
SECTOR_MAP  = {t: s for s, tickers in UNIVERSE.items() for t in tickers}
PERIOD      = "5y"


# ─────────────────────────────────────────────────────────────────────────────
# Live fetch via yfinance
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv_live() -> pd.DataFrame:
    import yfinance as yf

    log.info("Downloading OHLCV for %d tickers (5y daily) via yfinance …", len(ALL_TICKERS))
    tickers_str = " ".join(ALL_TICKERS)
    raw = yf.download(
        tickers_str,
        period=PERIOD,
        interval="1d",
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("yfinance returned empty dataframe")

    # yfinance multi-ticker returns (metric, ticker) MultiIndex columns
    # Flatten into long format
    frames = []
    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    for ticker in ALL_TICKERS:
        try:
            df_t = raw.xs(ticker, axis=1, level=1)[price_cols].copy()
        except KeyError:
            log.warning("  %s: not found in download, skipping", ticker)
            continue
        df_t = df_t.dropna(subset=["Close"])
        df_t = df_t.reset_index().rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        df_t["ticker"] = ticker
        df_t["sector"] = SECTOR_MAP.get(ticker, "Unknown")
        frames.append(df_t)
        log.info("  %-6s  %d rows  close=%.2f", ticker, len(df_t), df_t["close"].iloc[-1])

    ohlcv = pd.concat(frames, ignore_index=True)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.tz_localize(None)
    ohlcv["close"]  = ohlcv["close"].round(4)
    ohlcv["open"]   = ohlcv["open"].round(4)
    ohlcv["high"]   = ohlcv["high"].round(4)
    ohlcv["low"]    = ohlcv["low"].round(4)
    ohlcv["volume"] = ohlcv["volume"].fillna(0).astype(int)
    log.info("OHLCV download complete: %d rows, %d tickers", len(ohlcv), ohlcv["ticker"].nunique())
    return ohlcv[["date","ticker","sector","open","high","low","close","volume"]]


def fetch_fundamentals_live() -> pd.DataFrame:
    import yfinance as yf

    log.info("Fetching fundamentals for %d tickers …", len(ALL_TICKERS))
    rows = []
    for ticker in ALL_TICKERS:
        try:
            info = yf.Ticker(ticker).info
            rows.append({
                "ticker":         ticker,
                "sector":         SECTOR_MAP.get(ticker, "Unknown"),
                "name":           info.get("longName", ticker),
                "market_cap_bn":  round(info.get("marketCap", 0) / 1e9, 2) if info.get("marketCap") else np.nan,
                "pe_ratio":       info.get("trailingPE", np.nan),
                "forward_pe":     info.get("forwardPE",  np.nan),
                "eps_ttm":        info.get("trailingEps", np.nan),
                "beta":           info.get("beta", np.nan),
                "dividend_yield": info.get("dividendYield", np.nan),
                "52w_high":       info.get("fiftyTwoWeekHigh", np.nan),
                "52w_low":        info.get("fiftyTwoWeekLow",  np.nan),
                "avg_volume":     info.get("averageVolume", np.nan),
                "revenue_bn":     round(info.get("totalRevenue", 0) / 1e9, 2) if info.get("totalRevenue") else np.nan,
                "profit_margin":  info.get("profitMargins", np.nan),
                "roe":            info.get("returnOnEquity", np.nan),
                "debt_equity":    info.get("debtToEquity", np.nan),
                "current_ratio":  info.get("currentRatio", np.nan),
                "price_to_book":  info.get("priceToBook", np.nan),
                "analyst_target": info.get("targetMeanPrice", np.nan),
                "recommendation": info.get("recommendationKey", ""),
            })
            log.info("  %-6s  MCap=$%.1fBn  PE=%.1f  Beta=%.2f",
                     ticker,
                     rows[-1]["market_cap_bn"] if not np.isnan(rows[-1]["market_cap_bn"] or np.nan) else 0,
                     rows[-1]["pe_ratio"] if not np.isnan(rows[-1]["pe_ratio"] or np.nan) else 0,
                     rows[-1]["beta"] if not np.isnan(rows[-1]["beta"] or np.nan) else 0)
            time.sleep(0.3)
        except Exception as exc:
            log.warning("  %s fundamentals failed: %s", ticker, exc)
            rows.append({"ticker": ticker, "sector": SECTOR_MAP.get(ticker, "Unknown"), "name": ticker})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────

def _seed_prices() -> dict:
    return {
        "AAPL":200,"MSFT":380,"GOOGL":170,"NVDA":880,"META":480,"AMD":170,
        "JPM":200,"BAC":38,"GS":480,"MS":100,"WFC":60,"BLK":810,
        "JNJ":155,"PFE":27,"UNH":520,"ABBV":175,"MRK":130,"BMY":50,
        "XOM":115,"CVX":155,"COP":120,"SLB":50,"EOG":125,"PSX":155,
        "WMT":175,"COST":720,"PG":160,"KO":60,"PEP":170,"MCD":300,
        "CAT":370,"BA":180,"HON":200,"UPS":140,"GE":160,"MMM":100,
    }

def _generate_synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.warning("Network unavailable — generating synthetic GBM data for demo.")
    rng   = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    n     = len(dates)
    seed  = _seed_prices()
    rows, fund_rows = [], []
    for ticker, s0 in seed.items():
        sector = SECTOR_MAP[ticker]
        mu, sigma = rng.uniform(0.0001, 0.0006), rng.uniform(0.012, 0.028)
        prices = s0 * np.exp(np.cumsum((mu - .5*sigma**2) + sigma * rng.standard_normal(n)))
        rows.append(pd.DataFrame({
            "date": dates, "ticker": ticker, "sector": sector,
            "open": (prices*(1+rng.uniform(-0.005,0.005,n))).round(2),
            "high": (prices*(1+rng.uniform(0.002,0.015,n))).round(2),
            "low":  (prices*(1-rng.uniform(0.002,0.015,n))).round(2),
            "close": prices.round(2),
            "volume": rng.integers(3_000_000, 80_000_000, n),
        }))
        fund_rows.append({
            "ticker": ticker, "sector": sector, "name": ticker,
            "market_cap_bn": round(prices[-1]*rng.integers(1_000_000,10_000_000_000)/1e9,2),
            "pe_ratio": round(rng.uniform(10,45),2),
            "forward_pe": round(rng.uniform(8,35),2),
            "eps_ttm": round(rng.uniform(1,20),2),
            "beta": round(rng.uniform(0.5,1.8),2),
            "dividend_yield": round(rng.uniform(0,0.05),4),
            "52w_high": round(prices[-252:].max(),2),
            "52w_low":  round(prices[-252:].min(),2),
            "avg_volume": int(rng.integers(5_000_000,60_000_000)),
        })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(fund_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregations
# ─────────────────────────────────────────────────────────────────────────────

def _build_sector_daily(ohlcv: pd.DataFrame) -> pd.DataFrame:
    return (
        ohlcv.groupby(["date","sector"])
        .agg(avg_close=("close","mean"), total_volume=("volume","sum"),
             count=("ticker","nunique"))
        .reset_index()
    )

def _build_annual_returns(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    df["year"] = df["date"].dt.year
    first = df.groupby(["ticker","year"])["close"].first().rename("price_start")
    last  = df.groupby(["ticker","year"])["close"].last().rename("price_end")
    ann   = pd.concat([first, last], axis=1).reset_index()
    ann["annual_return_pct"] = (ann["price_end"]/ann["price_start"]-1)*100
    ann["sector"] = ann["ticker"].map(SECTOR_MAP)
    return ann


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion() -> None:
    log.info("=== INGESTION START ===")

    try:
        ohlcv = fetch_ohlcv_live()
        funds = fetch_fundamentals_live()
        log.info("Live data fetched successfully.")
    except Exception as exc:
        log.error("Live fetch failed (%s) — falling back to synthetic data.", exc)
        ohlcv, funds = _generate_synthetic()

    if ohlcv.empty:
        log.error("No OHLCV data — aborting.")
        return

    ohlcv.to_parquet(RAW_DIR / "ohlcv_raw.parquet",        index=False)
    funds.to_parquet(RAW_DIR / "fundamentals_raw.parquet", index=False)
    log.info("Raw saved: %d rows OHLCV, %d tickers fundamentals", len(ohlcv), len(funds))

    _build_sector_daily(ohlcv).to_parquet(PROC_DIR / "sector_daily.parquet",   index=False)
    _build_annual_returns(ohlcv).to_parquet(PROC_DIR / "annual_returns.parquet", index=False)
    log.info("Aggregations saved.")
    log.info("=== INGESTION COMPLETE ===")


if __name__ == "__main__":
    run_ingestion()
