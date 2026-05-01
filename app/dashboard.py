"""
Global Market Intelligence — Streamlit Dashboard
Run:  streamlit run app/dashboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT      = Path(__file__).resolve().parents[1]
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

SECTOR_PALETTE = {
    "Technology":  "#0066cc",
    "Finance":     "#00994d",
    "Healthcare":  "#cc0000",
    "Energy":      "#ff8800",
    "Consumer":    "#9900cc",
    "Industrials": "#666666",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_clean() -> pd.DataFrame:
    p = PROC_DIR / "cy_clean.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=300)
def load_features() -> pd.DataFrame:
    p = PROC_DIR / "cy_features.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=300)
def load_anomalies() -> pd.DataFrame:
    p = PROC_DIR / "cy_anomalies.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=300)
def load_forecasts() -> tuple[pd.DataFrame, pd.DataFrame]:
    xgb_p   = PROC_DIR / "xgb_forecasts.parquet"
    arima_p = PROC_DIR / "arima_forecasts.parquet"
    xgb   = pd.read_parquet(xgb_p)   if xgb_p.exists()   else pd.DataFrame()
    arima = pd.read_parquet(arima_p) if arima_p.exists() else pd.DataFrame()
    for df in [xgb, arima]:
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
    return xgb, arima

@st.cache_data(ttl=300)
def load_fundamentals() -> pd.DataFrame:
    p = ROOT / "data" / "raw" / "fundamentals_raw.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)

@st.cache_data(ttl=300)
def load_metrics() -> dict:
    p = MODEL_DIR / "metrics.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def sidebar(df: pd.DataFrame) -> tuple:
    st.sidebar.image("https://img.icons8.com/fluency/96/stock-market.png", width=64)
    st.sidebar.title("Global Market Intelligence")
    st.sidebar.markdown("*Multi-sector equity analytics platform*")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Executive Overview", "📈 Price & Volume", "🔍 EDA & Statistics",
         "⚠️ Anomaly Radar", "🔮 Forecasting", "🏦 Fundamentals", "📋 Model Report"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    all_sectors = sorted(df["sector"].dropna().unique().tolist()) if not df.empty else []
    sectors = st.sidebar.multiselect("Sectors", all_sectors, default=all_sectors)

    all_tickers = sorted(df[df["sector"].isin(sectors)]["ticker"].unique().tolist()) if not df.empty else []
    tickers = st.sidebar.multiselect("Tickers", all_tickers, default=all_tickers[:10])

    min_date = df["date"].min().date() if not df.empty else None
    max_date = df["date"].max().date() if not df.empty else None
    if min_date and max_date:
        date_range = st.sidebar.date_input("Date range", (min_date, max_date),
                                            min_value=min_date, max_value=max_date)
    else:
        date_range = None

    st.sidebar.divider()
    st.sidebar.caption("Data: Yahoo Finance via RapidAPI\nPipeline: ARIMA + XGBoost + IF + DBSCAN")
    return page, sectors, tickers, date_range


def filter_df(df: pd.DataFrame, sectors, tickers, date_range) -> pd.DataFrame:
    if df.empty:
        return df
    if sectors:
        df = df[df["sector"].isin(sectors)]
    if tickers:
        df = df[df["ticker"].isin(tickers)]
    if date_range and len(date_range) == 2:
        df = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Executive Overview
# ─────────────────────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame, anom_df: pd.DataFrame, metrics: dict) -> None:
    st.title("📊 Executive Overview")

    if df.empty:
        st.warning("No data loaded. Run the pipeline first:  `python src/pipeline.py`")
        return

    latest = df.sort_values("date").groupby("ticker").last().reset_index()

    # ── KPI cards ─────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Tickers Tracked", df["ticker"].nunique())
    with c2:
        avg_ret = (df.groupby("ticker")["daily_return"].mean() * 252 * 100).mean()
        st.metric("Avg Annual Return", f"{avg_ret:.1f}%")
    with c3:
        avg_vol = (df.groupby("ticker")["vol_20d"].mean()).mean() * 100 if "vol_20d" in df.columns else 0
        st.metric("Avg Annualised Vol", f"{avg_vol:.1f}%")
    with c4:
        n_anom = anom_df["is_anomaly"].sum() if not anom_df.empty and "is_anomaly" in anom_df.columns else 0
        st.metric("Anomalies Detected", f"{int(n_anom):,}")
    with c5:
        mape = metrics.get("xgb", {}).get("mape_pct", metrics.get("xgb_cv_mape_mean", "—"))
        st.metric("XGBoost MAPE", f"{mape:.2f}%" if isinstance(mape, float) else mape)

    st.divider()

    col_l, col_r = st.columns([3, 2])

    # ── sector performance heatmap ─────────────────────────────────────────
    with col_l:
        st.subheader("Sector Annual Returns (%)")
        df2 = df.copy()
        df2["year"] = df2["date"].dt.year
        sector_yr = (
            df2.groupby(["sector", "year"])["daily_return"]
            .mean()
            .mul(252 * 100)
            .reset_index()
            .rename(columns={"daily_return": "annual_return"})
        )
        pivot = sector_yr.pivot(index="sector", columns="year", values="annual_return").round(1)
        fig = px.imshow(pivot, color_continuous_scale="RdYlGn", aspect="auto",
                        labels={"color": "Return (%)"}, text_auto=True)
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── sector donut ──────────────────────────────────────────────────────
    with col_r:
        st.subheader("Sector Composition")
        sec_cnt = df.groupby("sector")["ticker"].nunique().reset_index()
        fig = px.pie(sec_cnt, names="sector", values="ticker", hole=0.5,
                     color="sector", color_discrete_map=SECTOR_PALETTE)
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── top movers ────────────────────────────────────────────────────────
    st.subheader("Top & Bottom 5 Tickers (YTD Return)")
    ytd_ret = (
        df.groupby("ticker")["daily_return"].mean()
        .mul(252 * 100).round(2).reset_index()
        .rename(columns={"daily_return": "ann_return_pct"})
        .merge(df[["ticker", "sector"]].drop_duplicates(), on="ticker")
        .sort_values("ann_return_pct", ascending=False)
    )
    cl, cr = st.columns(2)
    with cl:
        st.markdown("**Top 5 Gainers**")
        st.dataframe(ytd_ret.head(5).style.format({"ann_return_pct": "{:.2f}%"}),
                     use_container_width=True, hide_index=True)
    with cr:
        st.markdown("**Top 5 Losers**")
        st.dataframe(ytd_ret.tail(5).style.format({"ann_return_pct": "{:.2f}%"}),
                     use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Price & Volume
# ─────────────────────────────────────────────────────────────────────────────

def page_price_volume(df: pd.DataFrame) -> None:
    st.title("📈 Price & Volume Analysis")
    if df.empty:
        st.warning("No data. Run pipeline.")
        return

    ticker_sel = st.selectbox("Select Ticker", sorted(df["ticker"].unique()))
    sub = df[df["ticker"] == ticker_sel].sort_values("date")

    # candlestick + volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=sub["date"], open=sub["open"], high=sub["high"],
        low=sub["low"], close=sub["close"], name="OHLC",
        increasing_line_color="#00994d", decreasing_line_color="#cc0000",
    ), row=1, col=1)

    if "bb_upper" in sub.columns:
        for col, dash, color, label in [
            ("bb_upper", "dash", "rgba(0,102,204,0.4)", "BB Upper"),
            ("bb_lower", "dash", "rgba(0,102,204,0.4)", "BB Lower"),
            ("bb_middle","dot",  "rgba(0,102,204,0.7)", "BB Middle"),
        ]:
            fig.add_trace(go.Scatter(x=sub["date"], y=sub[col], line=dict(dash=dash, color=color),
                                     name=label, showlegend=True), row=1, col=1)

    colors = ["#00994d" if r >= 0 else "#cc0000"
              for r in sub["daily_return"].fillna(0)]
    fig.add_trace(go.Bar(x=sub["date"], y=sub["volume"], name="Volume",
                         marker_color=colors, opacity=0.7), row=2, col=1)

    fig.update_layout(height=600, xaxis_rangeslider_visible=False,
                      title=f"{ticker_sel} — Price & Volume",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # RSI + MACD
    if "rsi_14" in sub.columns and "macd_line" in sub.columns:
        st.subheader("Technical Indicators")
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=["RSI (14)", "MACD"])
        fig2.add_trace(go.Scatter(x=sub["date"], y=sub["rsi_14"],
                                  line=dict(color="#ff8800")), row=1, col=1)
        fig2.add_hline(y=70, line_dash="dash", line_color="red",   row=1, col=1)
        fig2.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        fig2.add_trace(go.Scatter(x=sub["date"], y=sub["macd_line"],
                                  name="MACD",   line=dict(color="#0066cc")), row=2, col=1)
        fig2.add_trace(go.Scatter(x=sub["date"], y=sub["macd_signal"],
                                  name="Signal", line=dict(color="#ff8800")), row=2, col=1)
        fig2.add_trace(go.Bar(x=sub["date"], y=sub["macd_hist"],
                              name="Histogram", marker_color="#9900cc", opacity=0.5), row=2, col=1)
        fig2.update_layout(height=420, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — EDA & Statistics
# ─────────────────────────────────────────────────────────────────────────────

def page_eda(df: pd.DataFrame) -> None:
    st.title("🔍 EDA & Statistics")
    if df.empty:
        st.warning("No data. Run pipeline.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Return Distribution", "Correlation Matrix",
                                       "Volatility Surface", "Rolling Stats"])

    with tab1:
        st.subheader("Daily Return Distributions by Sector")
        fig = px.violin(df.dropna(subset=["daily_return"]), x="sector", y="daily_return",
                        color="sector", box=True, points=False,
                        color_discrete_map=SECTOR_PALETTE,
                        labels={"daily_return": "Daily Return", "sector": "Sector"})
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # descriptive stats
        stats = (
            df.groupby("sector")["daily_return"]
            .agg(["mean", "std", "skew", "kurtosis", "min", "max"])
            .mul({"mean": 252*100, "std": np.sqrt(252)*100,
                  "skew": 1, "kurtosis": 1, "min": 100, "max": 100})
            .round(3)
            .rename(columns={"mean": "Ann.Return%", "std": "Ann.Vol%",
                              "skew": "Skewness", "kurtosis": "Excess Kurt",
                              "min": "Min Ret%", "max": "Max Ret%"})
        )
        st.dataframe(stats.style.background_gradient(cmap="RdYlGn", axis=0),
                     use_container_width=True)

    with tab2:
        st.subheader("Ticker Return Correlation Heatmap")
        pivot = df.pivot_table(index="date", columns="ticker", values="daily_return")
        corr  = pivot.corr().round(2)
        fig   = px.imshow(corr, color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, aspect="auto", text_auto=False)
        fig.update_layout(height=550, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Annualised Volatility Over Time (by Sector)")
        if "vol_20d" in df.columns:
            monthly = (
                df.copy()
                .assign(month=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
                .groupby(["month", "sector"])["vol_20d"].mean().mul(100).reset_index()
            )
            fig = px.area(monthly, x="month", y="vol_20d", color="sector",
                          color_discrete_map=SECTOR_PALETTE,
                          labels={"vol_20d": "Annualised Vol (%)", "month": "Month"})
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("20-Day Rolling Sharpe Ratio")
        rf = 0.05 / 252
        df2 = df.sort_values(["ticker", "date"]).copy()
        df2["excess"] = df2["daily_return"] - rf
        df2["roll_sharpe"] = df2.groupby("ticker")["excess"].transform(
            lambda x: x.rolling(20, min_periods=10).mean() /
                      (x.rolling(20, min_periods=10).std() + 1e-9) * np.sqrt(252)
        )
        tickers_sel = st.multiselect("Tickers", df["ticker"].unique().tolist(),
                                     default=list(df["ticker"].unique()[:6]))
        sub = df2[df2["ticker"].isin(tickers_sel)]
        fig = px.line(sub, x="date", y="roll_sharpe", color="ticker",
                      labels={"roll_sharpe": "Rolling Sharpe", "date": "Date"})
        fig.add_hline(y=1, line_dash="dash", line_color="grey")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Anomaly Radar
# ─────────────────────────────────────────────────────────────────────────────

def page_anomaly(anom_df: pd.DataFrame) -> None:
    st.title("⚠️ Anomaly Radar")
    if anom_df.empty or "is_anomaly" not in anom_df.columns:
        st.warning("No anomaly data. Run pipeline.")
        return

    anom = anom_df[anom_df["is_anomaly"]].copy()
    st.metric("Total Anomalies", f"{len(anom):,}", delta=f"{len(anom)/len(anom_df)*100:.2f}% of all observations")

    col1, col2 = st.columns(2)
    with col1:
        cnt = anom.groupby("ticker").size().reset_index(name="count").sort_values("count", ascending=False)
        fig = px.bar(cnt.head(15), x="ticker", y="count", color="count",
                     color_continuous_scale="Reds", title="Anomaly Count by Ticker")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "anomaly_severity" in anom.columns:
            sev = anom["anomaly_severity"].value_counts().reset_index()
            sev.columns = ["severity", "count"]
            fig = px.pie(sev, names="severity", values="count",
                         color="severity",
                         color_discrete_map={"watch": "#ffcc00", "moderate": "#ff8800",
                                             "severe": "#cc0000", "normal": "#00994d"},
                         title="Anomaly Severity Breakdown", hole=0.4)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    # timeline
    st.subheader("Anomaly Timeline")
    ticker_sel = st.selectbox("Ticker", sorted(anom_df["ticker"].unique()))
    sub     = anom_df[anom_df["ticker"] == ticker_sel].sort_values("date")
    sub_a   = sub[sub["is_anomaly"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["close"], mode="lines",
                             line=dict(color="#0066cc", width=1), name="Close"))
    fig.add_trace(go.Scatter(x=sub_a["date"], y=sub_a["close"], mode="markers",
                             marker=dict(color="red", size=7, symbol="x"),
                             name="Anomaly"))
    fig.update_layout(height=380, title=f"{ticker_sel} — Anomaly Timeline")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomaly Events Table")
    cols_show = ["date", "ticker", "sector", "close", "daily_return",
                 "anomaly_votes", "anomaly_severity",
                 "if_anomaly", "dbscan_anomaly", "residual_anomaly"]
    cols_show = [c for c in cols_show if c in anom.columns]
    st.dataframe(anom[cols_show].sort_values("date", ascending=False).head(200),
                 use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 5 — Forecasting
# ─────────────────────────────────────────────────────────────────────────────

def page_forecasting(df: pd.DataFrame, xgb_fc: pd.DataFrame, arima_fc: pd.DataFrame) -> None:
    st.title("🔮 Price Forecasting")
    all_tickers = sorted(set(
        (xgb_fc["ticker"].unique().tolist() if not xgb_fc.empty else []) +
        (arima_fc["ticker"].unique().tolist() if not arima_fc.empty else [])
    ))
    if not all_tickers:
        st.warning("No forecast data. Run pipeline.")
        return

    ticker_sel = st.selectbox("Select Ticker", all_tickers)
    hist_days  = st.slider("Historical context (days)", 60, 365, 180)

    fig = go.Figure()

    # historical
    if not df.empty:
        hist = df[df["ticker"] == ticker_sel].sort_values("date").tail(hist_days)
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["close"],
                                 mode="lines", line=dict(color="#0066cc", width=1.5),
                                 name="Historical"))

    # XGBoost forecast
    if not xgb_fc.empty:
        fc_x = xgb_fc[xgb_fc["ticker"] == ticker_sel].sort_values("date")
        fig.add_trace(go.Scatter(x=fc_x["date"], y=fc_x["forecast"],
                                 mode="lines", line=dict(color="#00994d", width=2, dash="dash"),
                                 name="XGBoost Forecast"))

    # ARIMA forecast + CI
    if not arima_fc.empty:
        fc_a = arima_fc[arima_fc["ticker"] == ticker_sel].sort_values("date")
        fig.add_trace(go.Scatter(x=fc_a["date"], y=fc_a["forecast"],
                                 mode="lines", line=dict(color="#ff8800", width=2, dash="dot"),
                                 name="ARIMA Forecast"))
        if "upper_90" in fc_a.columns:
            fig.add_traces([
                go.Scatter(x=pd.concat([fc_a["date"], fc_a["date"][::-1]]),
                           y=pd.concat([fc_a["upper_90"], fc_a["lower_90"][::-1]]),
                           fill="toself", fillcolor="rgba(255,136,0,0.15)",
                           line=dict(color="rgba(255,136,0,0)"),
                           name="90% CI ARIMA"),
            ])

    fig.update_layout(height=480, title=f"{ticker_sel} — 90-Day Price Forecast",
                      xaxis_title="Date", yaxis_title="Price (USD)",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # model comparison table
    col1, col2 = st.columns(2)
    if not xgb_fc.empty:
        with col1:
            st.subheader("XGBoost Forecast (next 30 days)")
            fc_x30 = xgb_fc[xgb_fc["ticker"] == ticker_sel].head(30)[["date","forecast"]]
            st.dataframe(fc_x30.style.format({"forecast": "${:.2f}"}),
                         use_container_width=True, hide_index=True)
    if not arima_fc.empty:
        with col2:
            st.subheader("ARIMA Forecast (next 30 days)")
            fc_a30 = arima_fc[arima_fc["ticker"] == ticker_sel].head(30)[
                ["date","forecast"] + (["lower_90","upper_90"]
                 if "lower_90" in arima_fc.columns else [])
            ]
            st.dataframe(fc_a30.style.format(
                {c: "${:.2f}" for c in fc_a30.columns if c != "date"}),
                use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 6 — Fundamentals
# ─────────────────────────────────────────────────────────────────────────────

def page_fundamentals(funds: pd.DataFrame) -> None:
    st.title("🏦 Fundamentals Screener")
    if funds.empty:
        st.warning("No fundamentals data. Run pipeline.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        pe_max = st.slider("Max P/E Ratio", 0, 100, 50)
    with col2:
        beta_max = st.slider("Max Beta", 0.0, 3.0, 2.0)
    with col3:
        mcap_min = st.slider("Min Market Cap ($Bn)", 0, 500, 0)

    filtered = funds[
        (funds["pe_ratio"]     <= pe_max) &
        (funds["beta"]         <= beta_max) &
        (funds["market_cap_bn"] >= mcap_min)
    ].copy() if all(c in funds.columns for c in ["pe_ratio", "beta", "market_cap_bn"]) else funds.copy()

    st.subheader(f"Screener Results: {len(filtered)} tickers")

    # bubble chart: market cap vs P/E
    if all(c in filtered.columns for c in ["market_cap_bn", "pe_ratio", "beta"]):
        fig = px.scatter(
            filtered.dropna(subset=["market_cap_bn", "pe_ratio"]),
            x="pe_ratio", y="market_cap_bn", size="market_cap_bn",
            color="sector", hover_data=["ticker", "beta", "dividend_yield"],
            labels={"pe_ratio": "P/E Ratio", "market_cap_bn": "Market Cap ($Bn)"},
            color_discrete_map=SECTOR_PALETTE,
            title="Market Cap vs P/E (bubble size = market cap)",
        )
        fig.update_layout(height=440)
        st.plotly_chart(fig, use_container_width=True)

    show_cols = [c for c in ["ticker","sector","name","market_cap_bn","pe_ratio","forward_pe",
                              "eps_ttm","beta","dividend_yield","52w_high","52w_low"]
                 if c in filtered.columns]
    st.dataframe(
        filtered[show_cols].sort_values("market_cap_bn", ascending=False)
        .style.format({c: "{:.2f}" for c in show_cols if c not in ["ticker","sector","name"]}),
        use_container_width=True, hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page 7 — Model Report
# ─────────────────────────────────────────────────────────────────────────────

def page_model_report(metrics: dict) -> None:
    st.title("📋 Model Performance Report")

    if not metrics:
        st.warning("No metrics found. Run pipeline.")
        return

    # metrics cards
    cols = st.columns(4)
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}_{kk}"] = vv
        else:
            flat[k] = v

    for i, (k, v) in enumerate(flat.items()):
        with cols[i % 4]:
            st.metric(k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else v)

    st.divider()

    # SHAP plot
    shap_path = MODEL_DIR / "shap_importance.png"
    if shap_path.exists():
        st.subheader("XGBoost SHAP Feature Importance")
        st.image(str(shap_path), use_column_width=True)

    # residual plots
    resid_path = MODEL_DIR / "residual_plots.png"
    if resid_path.exists():
        st.subheader("Return Distribution & Q-Q Plot")
        st.image(str(resid_path), use_column_width=True)

    # anomaly timeline
    anom_path = MODEL_DIR / "anomaly_timeline.png"
    if anom_path.exists():
        st.subheader("Anomaly Detection Timeline")
        st.image(str(anom_path), use_column_width=True)

    st.subheader("Raw Metrics JSON")
    st.json(metrics)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    df      = load_clean()
    anom_df = load_anomalies()
    feat_df = load_features()
    xgb_fc, arima_fc = load_forecasts()
    funds   = load_fundamentals()
    metrics = load_metrics()

    page, sectors, tickers, date_range = sidebar(df)

    df_f      = filter_df(df,      sectors, tickers, date_range)
    anom_df_f = filter_df(anom_df, sectors, tickers, date_range)
    feat_df_f = filter_df(feat_df, sectors, tickers, date_range)

    if   page == "📊 Executive Overview":  page_overview(df_f, anom_df_f, metrics)
    elif page == "📈 Price & Volume":       page_price_volume(df_f)
    elif page == "🔍 EDA & Statistics":     page_eda(feat_df_f)
    elif page == "⚠️ Anomaly Radar":        page_anomaly(anom_df_f)
    elif page == "🔮 Forecasting":          page_forecasting(df_f, xgb_fc, arima_fc)
    elif page == "🏦 Fundamentals":         page_fundamentals(funds)
    elif page == "📋 Model Report":         page_model_report(metrics)


main()
