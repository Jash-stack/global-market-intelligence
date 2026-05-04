"""
Evaluation — RMSE / MAE / MAPE for forecasts, SHAP feature importance,
residual plots, and summary report.  Saves metrics.json + plots to models/.
"""

import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error, \
                           mean_absolute_percentage_error

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[1]
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
PLOT_DIR  = ROOT / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Forecast metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_forecasts() -> dict:
    metrics = {}

    xgb_path   = PROC_DIR / "xgb_forecasts.parquet"
    arima_path = PROC_DIR / "arima_forecasts.parquet"
    actual_path = PROC_DIR / "cy_clean.parquet"

    if not actual_path.exists():
        log.warning("cy_clean.parquet not found — skipping forecast metrics.")
        return metrics

    actual = pd.read_parquet(actual_path)[["ticker", "date", "close"]]
    actual["date"] = pd.to_datetime(actual["date"])

    for name, path in [("xgb", xgb_path), ("arima", arima_path)]:
        if not path.exists():
            continue
        fc = pd.read_parquet(path)
        fc["date"] = pd.to_datetime(fc["date"])
        merged = fc.merge(actual, on=["ticker", "date"], how="inner")
        if merged.empty:
            log.warning("%s: no overlap with actuals (likely future dates)", name)
            metrics[f"{name}_rows_forecast"] = len(fc)
            continue
        rmse = np.sqrt(mean_squared_error(merged["close"], merged["forecast"]))
        mae  = mean_absolute_error(merged["close"], merged["forecast"])
        mape = mean_absolute_percentage_error(
                   merged["close"].replace(0, np.nan).dropna(),
                   merged.loc[merged["close"] != 0, "forecast"]) * 100
        metrics[name] = {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape_pct": round(mape, 4)}
        log.info("%s — RMSE: %.2f | MAE: %.2f | MAPE: %.2f%%", name.upper(), rmse, mae, mape)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SHAP feature importance
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(df: pd.DataFrame) -> None:
    model_path = MODEL_DIR / "xgb_forecaster.joblib"
    feat_path  = MODEL_DIR / "feature_names.json"
    if not model_path.exists() or not feat_path.exists():
        log.warning("XGBoost model not found — skipping SHAP.")
        return
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — skipping SHAP plots.")
        return

    model = joblib.load(model_path)
    feats = json.load(open(feat_path))
    cols  = [c for c in feats if c in df.columns]
    X     = df[cols].fillna(0).sample(min(2000, len(df)), random_state=42)

    try:
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X)
    except (ValueError, TypeError) as exc:
        log.warning("SHAP TreeExplainer failed (%s) — skipping SHAP plot.", exc)
        return

    # bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:20]
    ax.barh(np.array(cols)[idx][::-1], mean_abs[idx][::-1], color="#0066cc")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("XGBoost Feature Importance (SHAP)")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "shap_importance.png", dpi=120)
    plt.close(fig)
    log.info("SHAP plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Residual plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_residuals(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # return distribution
    ax1 = fig.add_subplot(gs[0])
    returns = df["daily_return"].dropna()
    ax1.hist(returns, bins=100, color="#0066cc", alpha=0.7, edgecolor="none")
    ax1.set_title("Daily Return Distribution (All Tickers)")
    ax1.set_xlabel("Daily Return")
    ax1.set_ylabel("Frequency")

    # Q-Q
    from scipy import stats
    ax2 = fig.add_subplot(gs[1])
    (osm, osr), (slope, intercept, r) = stats.probplot(returns.dropna(), dist="norm")
    ax2.scatter(osm, osr, s=2, alpha=0.3, color="#0066cc")
    ax2.plot(osm, slope * np.array(osm) + intercept, "r--", lw=1.5)
    ax2.set_title(f"Q-Q Plot  (R²={r**2:.4f})")
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "residual_plots.png", dpi=120)
    plt.close(fig)
    log.info("Residual plots saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly summary plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomaly_timeline(df: pd.DataFrame) -> None:
    if "is_anomaly" not in df.columns:
        return
    tickers = df["ticker"].unique()[:6]
    fig, axes = plt.subplots(len(tickers), 1, figsize=(14, 3 * len(tickers)), sharex=True)
    for ax, ticker in zip(axes, tickers):
        sub = df[df["ticker"] == ticker].sort_values("date")
        ax.plot(sub["date"], sub["close"], lw=0.8, color="#0066cc", label="Close")
        anom = sub[sub["is_anomaly"]]
        ax.scatter(anom["date"], anom["close"], color="red", s=15, zorder=5, label="Anomaly")
        ax.set_ylabel(ticker, rotation=0, labelpad=40)
        ax.legend(fontsize=7, loc="upper left")
    plt.suptitle("Anomaly Detection — Price Timeline", y=1.01, fontsize=13)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "anomaly_timeline.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Anomaly timeline plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation() -> None:
    log.info("=== EVALUATION START ===")
    df = pd.read_parquet(PROC_DIR / "cy_anomalies.parquet")

    forecast_metrics = evaluate_forecasts()
    compute_shap(df)
    plot_residuals(df)
    plot_anomaly_timeline(df)

    # load existing metrics and merge
    metrics_path = MODEL_DIR / "metrics.json"
    existing = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)
    existing.update(forecast_metrics)
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)

    log.info("Metrics: %s", json.dumps(existing, indent=2))
    log.info("=== EVALUATION COMPLETE ===")


if __name__ == "__main__":
    run_evaluation()
