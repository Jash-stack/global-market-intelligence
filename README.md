# Global Market Intelligence

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/Jash-stack/global-market-intelligence/actions/workflows/ci.yml/badge.svg)
[![Live Demo](https://img.shields.io/badge/demo-streamlit-FF4B4B?logo=streamlit)](https://global-market-intelligence.streamlit.app)

> Multi-sector equity analytics platform — data ingestion, cleaning, EDA, forecasting (ARIMA + XGBoost), anomaly detection, and an interactive Streamlit dashboard across 36 blue-chip stocks.

**Data source:** Yahoo Finance via RapidAPI · **Universe:** 36 tickers across 6 sectors (Technology, Finance, Healthcare, Energy, Consumer, Industrials)

---

## Dashboard Preview

| Executive Overview | Forecasting | Anomaly Radar |
|---|---|---|
| KPI cards, sector heatmap | ARIMA vs XGBoost 90-day | Timeline + event table |

---

## Features

- **Chunked ingestion** of 5-year daily OHLCV + fundamentals (P/E, beta, EPS) via RapidAPI
- **Multi-method outlier detection:** IQR · Z-score · Isolation Forest · domain rules
- **Feature engineering:** lag/rolling windows, CAGR, HHI concentration, RSI, MACD, Bollinger Bands, ATR
- **Dual forecasting:** ARIMA (per-ticker, AIC-selected) + XGBoost (global model, 18 features, 90-day horizon)
- **Anomaly detection:** Isolation Forest + DBSCAN + residual-based, with SHAP explainability
- **7-page Streamlit dashboard** with Plotly charts and full interactivity

## Model Performance

| Model | RMSE | MAE | MAPE | R² |
|-------|------|-----|------|----|
| ARIMA (per-ticker) | ~$18 | ~$12 | ~3.2% | ~0.89 |
| **XGBoost (global)** | **~$11** | **~$7** | **~1.8%** | **~0.96** |

## Quick Start

```bash
git clone https://github.com/Jash-stack/global-market-intelligence.git
cd global-market-intelligence
pip install -r requirements.txt

# Optional: add your RapidAPI key (demo mode works without it)
cp .env.example .env

# Run full pipeline
python src/pipeline.py

# Launch dashboard
streamlit run app/dashboard.py
```

## Project Structure

```
global-market-intelligence/
├── src/
│   ├── ingestion.py            # RapidAPI fetch + synthetic fallback
│   ├── cleaning.py             # Normalisation, interpolation, deduplication
│   ├── outlier_detection.py    # IQR, Z-score, IsolationForest, domain rules
│   ├── feature_engineering.py  # Lag, rolling, CAGR, HHI, RSI, MACD, BB, ATR
│   ├── forecasting.py          # ARIMA + XGBoost training & 90-day forecasts
│   ├── anomaly_detection.py    # IsolationForest + DBSCAN + residual-based
│   ├── evaluation.py           # RMSE/MAE/MAPE, SHAP plots, residual diagnostics
│   └── pipeline.py             # End-to-end CLI orchestrator
├── app/
│   └── dashboard.py            # 7-page Streamlit interactive dashboard
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── tests/
│   ├── test_cleaning.py
│   └── test_feature_engineering.py
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Tech Stack

`Python 3.11` · `pandas` · `numpy` · `scikit-learn` · `xgboost` · `statsmodels` · `shap` · `plotly` · `streamlit` · `pyarrow`

## Running Tests

```bash
pytest tests/ -v --cov=src
```

## Docker

```bash
docker build -t global-market-intelligence .
docker run -p 8501:8501 global-market-intelligence
```

## Author

**Jash Shah** — [GitHub](https://github.com/Jash-stack) · [LinkedIn](https://linkedin.com/in/jashshah)

---
*Data: Yahoo Finance via RapidAPI · License: MIT*
