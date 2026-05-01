# Global Market Intelligence

> Multi-sector equity analytics platform — data ingestion, cleaning, EDA,
> forecasting (ARIMA + XGBoost), anomaly detection, and an MNC-level
> Streamlit dashboard.

**Data source:** [Yahoo Finance via RapidAPI](https://rapidapi.com/apidojo/api/yh-finance)  
**Universe:** 36 blue-chip stocks across 6 sectors (Technology, Finance, Healthcare, Energy, Consumer, Industrials)

---

## Project Structure

```
global-market-intelligence/
│
├── data/
│   ├── raw/
│   │   ├── ohlcv_raw.parquet          ← 5-year daily OHLCV (36 tickers)
│   │   └── fundamentals_raw.parquet   ← P/E, market cap, beta, EPS
│   └── processed/
│       ├── sector_daily.parquet       ← sector-aggregated daily stats
│       ├── annual_returns.parquet     ← per-ticker annual return table
│       ├── cy_clean.parquet           ← cleaned OHLCV + derived cols
│       ├── cy_outliers.parquet        ← with IQR/Z-score/IF/domain flags
│       ├── cy_features.parquet        ← full ML feature matrix
│       ├── cy_anomalies.parquet       ← IF + DBSCAN + residual anomalies
│       ├── xgb_forecasts.parquet      ← XGBoost 90-day price forecasts
│       └── arima_forecasts.parquet    ← ARIMA 90-day price forecasts
│
├── notebooks/
│   ├── 01_eda.ipynb                   ← Exploratory data analysis
│   └── 02_modeling.ipynb              ← Forecasting & anomaly detection
│
├── src/
│   ├── ingestion.py          ← RapidAPI fetch + synthetic fallback
│   ├── cleaning.py           ← Normalisation, interpolation, deduplication
│   ├── outlier_detection.py  ← IQR, Z-score, IsolationForest, domain rules
│   ├── feature_engineering.py ← Lag, rolling, CAGR, HHI, RSI, MACD, BB, ATR
│   ├── forecasting.py        ← ARIMA + XGBoost training & 90-day forecasts
│   ├── anomaly_detection.py  ← IsolationForest + DBSCAN + residual-based
│   ├── evaluation.py         ← RMSE/MAE/MAPE, SHAP plots, residual diagnostics
│   └── pipeline.py           ← End-to-end orchestrator (CLI)
│
├── app/
│   └── dashboard.py          ← 7-page Streamlit interactive dashboard
│
├── models/
│   ├── xgb_forecaster.joblib
│   ├── anomaly_iforest.joblib
│   ├── arima_<ticker>.pkl
│   ├── feature_names.json
│   ├── metrics.json
│   ├── shap_importance.png
│   ├── residual_plots.png
│   └── anomaly_timeline.png
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/global-market-intelligence.git
cd global-market-intelligence
pip install -r requirements.txt
```

### 2. Configure API key (optional)

```bash
cp .env.example .env
# edit .env and add your RAPIDAPI_KEY
```

> **No API key?** The pipeline runs in **demo mode** using synthetic GBM price
> paths that mimic real market behaviour — all ML steps work identically.

### 3. Run the full pipeline

```bash
# All steps
python src/pipeline.py

# Individual steps
python src/pipeline.py --steps ingest,clean,outlier
python src/pipeline.py --steps features,forecast,anomaly,eval

# Skip a step
python src/pipeline.py --skip eval
```

### 4. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

### 5. Open notebooks

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_modeling.ipynb
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 Executive Overview | KPI cards, sector heatmap, top/bottom movers |
| 📈 Price & Volume | Candlestick + Bollinger Bands, RSI, MACD |
| 🔍 EDA & Statistics | Return distributions, correlation matrix, rolling Sharpe |
| ⚠️ Anomaly Radar | Timeline, severity breakdown, event table |
| 🔮 Forecasting | ARIMA vs XGBoost 90-day forecasts with CI |
| 🏦 Fundamentals | Screener — P/E, beta, market cap bubble chart |
| 📋 Model Report | SHAP importance, residual diagnostics, metrics JSON |

---

## ML Models

| Model | Task | Key Metric |
|-------|------|-----------|
| ARIMA(p,1,q) | Per-ticker price forecast | AIC-selected order |
| XGBoost | Global return forecast (18 features) | CV MAPE |
| IsolationForest | Multivariate anomaly detection | contamination=1.5% |
| DBSCAN | Cluster-based anomaly detection | noise points |

---

## Tech Stack

`Python 3.11` · `pandas` · `numpy` · `scikit-learn` · `xgboost` · `statsmodels` · `shap` · `plotly` · `streamlit` · `pyarrow`

---

## Author

**Jash Shah** — Data Science Portfolio Project  
Data Source: Yahoo Finance via [RapidAPI](https://rapidapi.com/apidojo/api/yh-finance)
