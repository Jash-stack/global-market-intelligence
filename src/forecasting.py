"""
Forecasting — ARIMA (per-ticker) + XGBoost (global model) for 90-day ahead
price forecasts.  Saves xgb_forecasts.parquet, arima_forecasts.parquet,
and models to models/.
"""

import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import timedelta

from sklearn.preprocessing      import StandardScaler
from sklearn.model_selection    import TimeSeriesSplit

import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[1]
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FORECAST_DAYS = 90
TOP_TICKERS   = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "JNJ", "XOM", "WMT", "CAT", "META"]

XGB_FEATURES = [
    "return_lag_1d", "return_lag_2d", "return_lag_3d", "return_lag_5d",
    "roll_mean_5d",  "roll_mean_20d", "roll_std_5d",   "roll_std_20d",
    "rsi_14",        "macd_hist",     "bb_pct",        "bb_width",
    "atr",           "rel_volume",    "momentum_1m",   "momentum_3m",
    "vol_20d",       "day_of_week",   "month",
]


# ─────────────────────────────────────────────────────────────────────────────
# ARIMA
# ─────────────────────────────────────────────────────────────────────────────

def _fit_arima(series: pd.Series):
    """Auto-selects order via AIC search; falls back to ARIMA(2,1,2)."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        best_aic, best_order, best_model = np.inf, (2, 1, 2), None
        for p in range(4):
            for d in [1]:
                for q in range(4):
                    try:
                        m = ARIMA(series, order=(p, d, q)).fit()
                        if m.aic < best_aic:
                            best_aic, best_order, best_model = m.aic, (p, d, q), m
                    except Exception:
                        pass
        return best_model, best_order
    except ImportError:
        log.error("statsmodels not installed.")
        return None, None


def run_arima_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Fitting ARIMA for %d tickers …", len(TOP_TICKERS))
    rows = []
    for ticker in TOP_TICKERS:
        grp = df[df["ticker"] == ticker].sort_values("date").dropna(subset=["close"])
        if len(grp) < 100:
            continue
        series = grp.set_index("date")["close"].asfreq("B").ffill()
        model, order = _fit_arima(series)
        if model is None:
            continue

        # save model
        joblib.dump(model, MODEL_DIR / f"arima_{ticker}.pkl")

        # forecast
        fc = model.forecast(steps=FORECAST_DAYS)
        last_date = series.index[-1]
        future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=FORECAST_DAYS)
        conf = model.get_forecast(steps=FORECAST_DAYS).conf_int(alpha=0.10)

        for i, (fd, fv) in enumerate(zip(future_dates, fc)):
            rows.append({
                "ticker":     ticker,
                "date":       fd,
                "model":      "ARIMA",
                "arima_order": str(order),
                "forecast":   round(fv, 2),
                "lower_90":   round(conf.iloc[i, 0], 2),
                "upper_90":   round(conf.iloc[i, 1], 2),
            })
        log.info("  %s ARIMA%s done", ticker, order)

    arima_df = pd.DataFrame(rows)
    arima_df.to_parquet(PROC_DIR / "arima_forecasts.parquet", index=False)
    log.info("ARIMA forecasts saved: %d rows", len(arima_df))
    return arima_df


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_xgb_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feats = [c for c in XGB_FEATURES if c in df.columns]
    target = "daily_return"
    sub = df[feats + [target, "ticker", "date"]].dropna()
    X = sub[feats]
    y = sub[target]
    return X, y, feats


def train_xgb(df: pd.DataFrame) -> xgb.XGBRegressor:
    log.info("Training XGBoost …")
    X, y, feats = _prepare_xgb_data(df)

    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
            objective="reg:squarederror", tree_method="hist",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                  verbose=False)
        preds  = model.predict(X.iloc[val_idx])
        actual = y.iloc[val_idx].values
        # directional accuracy: did we predict the right sign?
        dir_acc = np.mean(np.sign(preds) == np.sign(actual)) * 100
        # MAE in basis points (1 bp = 0.01%)
        mae_bp  = np.mean(np.abs(preds - actual)) * 10_000
        fold_metrics.append({"dir_acc": dir_acc, "mae_bp": mae_bp})
        log.info("  Fold %d  Dir.Acc=%.1f%%  MAE=%.2f bp", fold + 1, dir_acc, mae_bp)

    # final model on full data
    final = xgb.XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        objective="reg:squarederror", tree_method="hist",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    final.fit(X, y)

    avg_dir_acc = round(np.mean([m["dir_acc"] for m in fold_metrics]), 2)
    avg_mae_bp  = round(np.mean([m["mae_bp"]  for m in fold_metrics]), 2)

    joblib.dump(final, MODEL_DIR / "xgb_forecaster.joblib")
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(feats, f)
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({
            "xgb_cv_directional_accuracy_pct": avg_dir_acc,
            "xgb_cv_mae_basis_points":         avg_mae_bp,
            "xgb_cv_folds":                    5,
        }, f)

    log.info("XGBoost CV  Dir.Acc=%.1f%%  MAE=%.2f bp", avg_dir_acc, avg_mae_bp)
    return final, feats


def run_xgb_forecasts(df: pd.DataFrame, model: xgb.XGBRegressor, feats: list[str]) -> pd.DataFrame:
    """Iterative 90-step ahead forecast using last known feature row per ticker."""
    log.info("Generating XGBoost 90-day forecasts …")
    rows = []
    for ticker in TOP_TICKERS:
        grp = df[df["ticker"] == ticker].sort_values("date").dropna(subset=feats + ["close"])
        if grp.empty:
            continue
        last_row    = grp[feats].iloc[[-1]].copy()
        last_price  = grp["close"].iloc[-1]
        last_date   = grp["date"].iloc[-1]
        future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=FORECAST_DAYS)

        price = last_price
        for fd in future_dates:
            ret_pred = float(model.predict(last_row)[0])
            # clip to ±0.3% per day — limits 90-day range to ≈±31% (realistic)
            ret_pred = np.clip(ret_pred, -0.003, 0.003)
            price    = price * (1 + ret_pred)
            rows.append({
                "ticker":   ticker,
                "date":     fd,
                "model":    "XGBoost",
                "forecast": round(price, 2),
            })
            # shift lag features forward by 1 step
            for lag in [5, 3, 2, 1]:
                src = f"return_lag_{lag-1}d" if lag > 1 else "return_lag_1d"
                dst = f"return_lag_{lag}d"
                if src in last_row.columns and dst in last_row.columns:
                    last_row[dst] = last_row[src]
            if "return_lag_1d" in last_row.columns:
                last_row["return_lag_1d"] = ret_pred

    xgb_df = pd.DataFrame(rows)
    xgb_df.to_parquet(PROC_DIR / "xgb_forecasts.parquet", index=False)
    log.info("XGBoost forecasts saved: %d rows", len(xgb_df))
    return xgb_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_forecasting() -> None:
    log.info("=== FORECASTING START ===")
    df = pd.read_parquet(PROC_DIR / "cy_features.parquet")

    run_arima_forecasts(df)

    xgb_model, feats = train_xgb(df)
    run_xgb_forecasts(df, xgb_model, feats)

    log.info("=== FORECASTING COMPLETE ===")


if __name__ == "__main__":
    run_forecasting()
