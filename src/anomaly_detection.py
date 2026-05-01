"""
Anomaly detection — IsolationForest (multivariate), DBSCAN (cluster-based),
and ARIMA residual-based detection.
Saves cy_anomalies.parquet.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble        import IsolationForest
from sklearn.cluster         import DBSCAN
from sklearn.preprocessing   import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[1]
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

ANOMALY_FEATURES = [
    "daily_return", "log_return", "vol_20d",
    "rel_volume", "bb_pct", "atr", "rsi_14",
    "macd_hist", "momentum_1m",
]


# ─────────────────────────────────────────────────────────────────────────────
# Method 1 — IsolationForest
# ─────────────────────────────────────────────────────────────────────────────

def isolation_forest_anomalies(df: pd.DataFrame, contamination: float = 0.015) -> pd.Series:
    cols = [c for c in ANOMALY_FEATURES if c in df.columns]
    X = df[cols].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = IsolationForest(n_estimators=300, contamination=contamination,
                          max_samples="auto", random_state=42, n_jobs=-1)
    labels = clf.fit_predict(Xs)
    scores = clf.score_samples(Xs)    # negative; lower = more anomalous
    joblib.dump(clf, MODEL_DIR / "anomaly_iforest.joblib")
    log.info("IsolationForest: %d anomalies flagged", (labels == -1).sum())
    return pd.Series(labels == -1, index=df.index, name="if_anomaly"), \
           pd.Series(-scores, index=df.index, name="if_score")


# ─────────────────────────────────────────────────────────────────────────────
# Method 2 — DBSCAN
# ─────────────────────────────────────────────────────────────────────────────

def dbscan_anomalies(df: pd.DataFrame, eps: float = 0.8, min_samples: int = 10) -> pd.Series:
    """Points labelled -1 by DBSCAN (noise) are anomalies."""
    cols = [c for c in ANOMALY_FEATURES[:4] if c in df.columns]   # subset for speed
    X = df[cols].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(Xs)
    anomaly = pd.Series(labels == -1, index=df.index, name="dbscan_anomaly")
    log.info("DBSCAN: %d noise points flagged (eps=%.2f, min_samples=%d)",
             anomaly.sum(), eps, min_samples)
    return anomaly


# ─────────────────────────────────────────────────────────────────────────────
# Method 3 — ARIMA residual anomalies
# ─────────────────────────────────────────────────────────────────────────────

def residual_anomalies(df: pd.DataFrame, z_thresh: float = 3.5) -> pd.Series:
    """
    For tickers with a saved ARIMA model, compute in-sample residuals.
    Flag dates where |residual| > z_thresh * MAD.
    """
    flags = pd.Series(False, index=df.index, name="residual_anomaly")
    model_files = list(MODEL_DIR.glob("arima_*.pkl"))
    if not model_files:
        log.warning("No ARIMA models found — skipping residual anomalies.")
        return flags

    for model_path in model_files:
        ticker = model_path.stem.replace("arima_", "")
        grp    = df[df["ticker"] == ticker].sort_values("date").dropna(subset=["close"])
        if grp.empty:
            continue
        try:
            model    = joblib.load(model_path)
            resid    = model.resid
            resid.index = pd.to_datetime(resid.index)
            mad      = (resid - resid.median()).abs().median()
            if mad == 0:
                continue
            mz       = 0.6745 * (resid - resid.median()) / mad
            anom_dates = mz[mz.abs() > z_thresh].index
            mask = (df["ticker"] == ticker) & df["date"].isin(anom_dates)
            flags.loc[mask] = True
            log.info("  %s residual anomalies: %d", ticker, mask.sum())
        except Exception as exc:
            log.warning("  %s residual failed: %s", ticker, exc)

    log.info("Residual anomalies total: %d", flags.sum())
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Composite score + severity tagging
# ─────────────────────────────────────────────────────────────────────────────

def tag_severity(score: pd.Series) -> pd.Series:
    bins   = [-1, 0, 1, 2, 3]
    labels = ["normal", "watch", "moderate", "severe"]
    return pd.cut(score, bins=bins, labels=labels).astype(str)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_anomaly_detection() -> pd.DataFrame:
    log.info("=== ANOMALY DETECTION START ===")
    df = pd.read_parquet(PROC_DIR / "cy_features.parquet")

    if_flag, if_score    = isolation_forest_anomalies(df)
    dbscan_flag          = dbscan_anomalies(df)
    resid_flag           = residual_anomalies(df)

    df["if_anomaly"]       = if_flag
    df["if_score"]         = if_score
    df["dbscan_anomaly"]   = dbscan_flag
    df["residual_anomaly"] = resid_flag

    df["anomaly_votes"] = (
        df["if_anomaly"].astype(int)
      + df["dbscan_anomaly"].astype(int)
      + df["residual_anomaly"].astype(int)
    )
    df["is_anomaly"]       = df["anomaly_votes"] >= 2
    df["anomaly_severity"] = tag_severity(df["anomaly_votes"])

    out = PROC_DIR / "cy_anomalies.parquet"
    df.to_parquet(out, index=False)

    summary = df.groupby("ticker")["is_anomaly"].sum().sort_values(ascending=False)
    log.info("Top anomaly counts per ticker:\n%s", summary.head(10).to_string())
    log.info("Total anomalies: %d / %d rows → saved %s",
             df["is_anomaly"].sum(), len(df), out)
    log.info("=== ANOMALY DETECTION COMPLETE ===")
    return df


if __name__ == "__main__":
    run_anomaly_detection()
