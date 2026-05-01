"""
Outlier detection — IQR, Z-score, IsolationForest, and domain rules.
Flags suspicious price/return observations without dropping them.
Saves cy_outliers.parquet.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

FEATURE_COLS = ["daily_return", "log_return", "price_range", "dollar_volume", "vol_20d"]


# ─────────────────────────────────────────────────────────────────────────────
# Individual detectors
# ─────────────────────────────────────────────────────────────────────────────

def iqr_outliers(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Return boolean mask: True = outlier (beyond k*IQR fence)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - k * iqr) | (series > q3 + k * iqr)


def zscore_outliers(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Modified Z-score using median absolute deviation (robust)."""
    median   = series.median()
    mad      = (series - median).abs().median()
    if mad == 0:
        return pd.Series(False, index=series.index)
    mzscore  = 0.6745 * (series - median) / mad
    return mzscore.abs() > threshold


def isolation_forest_flags(df: pd.DataFrame, contamination: float = 0.02) -> pd.Series:
    """IsolationForest on multi-dimensional feature space; returns bool mask."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    sub  = df[cols].copy().fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(sub)
    clf = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=42, n_jobs=-1)
    preds = clf.fit_predict(X)          # -1 = outlier, 1 = normal
    return pd.Series(preds == -1, index=df.index)


def domain_rules(df: pd.DataFrame) -> pd.Series:
    """Business-logic flags: >25% single-day move, negative volume, etc."""
    flags = pd.Series(False, index=df.index)
    if "daily_return" in df.columns:
        flags |= df["daily_return"].abs() > 0.25      # >25% single-day swing
    if "volume" in df.columns:
        flags |= df["volume"] < 0                      # negative volume
    if "close" in df.columns:
        flags |= df["close"] <= 0                      # non-positive price
    if "high" in df.columns and "low" in df.columns:
        flags |= df["high"] < df["low"]                # inverted OHLC
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate & summarise
# ─────────────────────────────────────────────────────────────────────────────

def build_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker outlier rate table."""
    cols = [c for c in df.columns if c.endswith("_outlier")]
    df["any_outlier"] = df[cols].any(axis=1)
    summary = (
        df.groupby("ticker")[["any_outlier"] + cols]
        .mean()
        .mul(100)
        .round(2)
        .rename(columns=lambda c: c.replace("_outlier", "_pct"))
    )
    log.info("Outlier rates (%%)::\n%s", summary.to_string())
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_outlier_detection() -> pd.DataFrame:
    log.info("=== OUTLIER DETECTION START ===")
    df = pd.read_parquet(PROC_DIR / "cy_clean.parquet")

    # ── per-ticker IQR & Z-score on returns ──────────────────────────────
    iqr_flags, zscore_flags = [], []
    for _, grp in df.groupby("ticker"):
        if "daily_return" in grp.columns:
            iqr_flags.append(iqr_outliers(grp["daily_return"].fillna(0)))
            zscore_flags.append(zscore_outliers(grp["daily_return"].fillna(0)))
        else:
            iqr_flags.append(pd.Series(False, index=grp.index))
            zscore_flags.append(pd.Series(False, index=grp.index))

    df["iqr_outlier"]    = pd.concat(iqr_flags).reindex(df.index)
    df["zscore_outlier"] = pd.concat(zscore_flags).reindex(df.index)

    # ── global IsolationForest ────────────────────────────────────────────
    log.info("Fitting IsolationForest …")
    df["iforest_outlier"] = isolation_forest_flags(df)

    # ── domain rules ─────────────────────────────────────────────────────
    df["domain_outlier"] = domain_rules(df)

    # ── composite flag ───────────────────────────────────────────────────
    df["outlier_score"] = (
        df["iqr_outlier"].astype(int)
      + df["zscore_outlier"].astype(int)
      + df["iforest_outlier"].astype(int)
      + df["domain_outlier"].astype(int)
    )
    df["is_outlier"] = df["outlier_score"] >= 2   # flagged by ≥2 methods

    build_outlier_summary(df)

    out = PROC_DIR / "cy_outliers.parquet"
    df.to_parquet(out, index=False)
    log.info("Saved → %s  (flagged: %d / %d rows)", out,
             df["is_outlier"].sum(), len(df))
    log.info("=== OUTLIER DETECTION COMPLETE ===")
    return df


if __name__ == "__main__":
    run_outlier_detection()
