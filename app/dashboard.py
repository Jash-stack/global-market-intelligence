"""
Global Market Intelligence — Pitch-Ready Streamlit Dashboard
Run:  streamlit run app/dashboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats as scipy_stats
from scipy.optimize import minimize

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
    "Technology":  "#3b82f6",
    "Finance":     "#10b981",
    "Healthcare":  "#f43f5e",
    "Energy":      "#f59e0b",
    "Consumer":    "#a855f7",
    "Industrials": "#64748b",
}

# ── custom CSS ────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* background */
    .stApp { background: #0a0e1a; color: #e2e8f0; }
    .main .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1600px; }

    /* sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1224 0%, #111827 100%);
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.05em; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(59,130,246,0.15); }
    .kpi-card::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    .kpi-label { font-size: 0.7rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
    .kpi-value { font-size: 1.9rem; font-weight: 700; color: #f1f5f9; line-height: 1; }
    .kpi-delta-pos { font-size: 0.78rem; font-weight: 600; color: #10b981; margin-top: 0.3rem; }
    .kpi-delta-neg { font-size: 0.78rem; font-weight: 600; color: #f43f5e; margin-top: 0.3rem; }
    .kpi-delta-neu { font-size: 0.78rem; font-weight: 600; color: #64748b; margin-top: 0.3rem; }
    .kpi-icon { position: absolute; top: 1rem; right: 1rem; font-size: 1.6rem; opacity: 0.2; }

    /* section headers */
    .section-header {
        font-size: 1.05rem; font-weight: 700; color: #e2e8f0;
        border-left: 3px solid #3b82f6; padding-left: 0.75rem;
        margin: 1.5rem 0 0.75rem 0;
    }

    /* page title */
    .page-title {
        font-size: 1.6rem; font-weight: 800; color: #f8fafc;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .page-subtitle { font-size: 0.82rem; color: #475569; margin-bottom: 1.5rem; }

    /* table styling */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    [data-testid="stDataFrame"] { background: #1e293b; border-radius: 12px; }

    /* badge */
    .badge {
        display: inline-block; padding: 0.2rem 0.6rem;
        border-radius: 9999px; font-size: 0.7rem; font-weight: 600;
    }
    .badge-green { background: rgba(16,185,129,0.15); color: #10b981; }
    .badge-red   { background: rgba(244,63,94,0.15);  color: #f43f5e; }
    .badge-blue  { background: rgba(59,130,246,0.15); color: #3b82f6; }
    .badge-amber { background: rgba(245,158,11,0.15); color: #f59e0b; }

    /* divider */
    hr { border-color: #1e293b; margin: 1.25rem 0; }

    /* tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #111827; border-radius: 10px; padding: 4px; gap: 2px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 8px; color: #64748b;
        font-size: 0.8rem; font-weight: 600; padding: 0.4rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
        color: white !important;
    }

    /* metric override */
    [data-testid="metric-container"] {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 12px; padding: 0.75rem 1rem;
    }
    [data-testid="stMetricValue"] { color: #f1f5f9; }
    [data-testid="stMetricLabel"] { color: #64748b; font-size: 0.72rem; text-transform: uppercase; }

    /* slider */
    .stSlider [data-baseweb="slider"] { color: #3b82f6; }

    /* selectbox, multiselect */
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: #1e293b; border: 1px solid #334155; color: #e2e8f0; border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Plot theme helper
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0f172a",
    font=dict(family="Inter", color="#94a3b8", size=11),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155", zeroline=False),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#94a3b8")),
    margin=dict(l=10, r=10, t=30, b=10),
    hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font_color="#e2e8f0", font_size=12),
)

def apply_theme(fig, height=400, title="", show_legend=True):
    layout = dict(**PLOTLY_LAYOUT, height=height, showlegend=show_legend)
    if title:
        layout["title"] = dict(text=title, font=dict(size=13, color="#e2e8f0"), x=0.01)
    fig.update_layout(**layout)
    return fig


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
def load_forecasts():
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
# Risk / portfolio helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_metrics(returns: pd.Series, rf: float = 0.05/252) -> dict:
    r = returns.dropna()
    ann_ret  = r.mean() * 252
    ann_vol  = r.std() * np.sqrt(252)
    sharpe   = (r.mean() - rf) / (r.std() + 1e-9) * np.sqrt(252)
    downside = r[r < rf].std() * np.sqrt(252) + 1e-9
    sortino  = (r.mean() - rf) / downside
    cum      = (1 + r).cumprod()
    roll_max = cum.cummax()
    dd       = (cum - roll_max) / (roll_max + 1e-9)
    max_dd   = dd.min()
    var_95   = float(np.percentile(r, 5))
    cvar_95  = float(r[r <= var_95].mean())
    calmar   = ann_ret / (abs(max_dd) + 1e-9)
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                sortino=sortino, max_dd=max_dd, var_95=var_95,
                cvar_95=cvar_95, calmar=calmar)

def max_drawdown_series(returns: pd.Series) -> pd.Series:
    cum = (1 + returns.fillna(0)).cumprod()
    roll_max = cum.cummax()
    return (cum - roll_max) / (roll_max + 1e-9)

def portfolio_stats(weights, mean_ret, cov):
    p_ret = np.dot(weights, mean_ret) * 252
    p_vol = np.sqrt(weights @ cov @ weights) * np.sqrt(252)
    return p_ret, p_vol

def efficient_frontier_points(mean_ret, cov, n=60):
    n_assets = len(mean_ret)
    results = []
    target_rets = np.linspace(mean_ret.min()*252, mean_ret.max()*252, n)
    for tr in target_rets:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, tr=tr: portfolio_stats(w, mean_ret, cov)[0] - tr},
        )
        bounds = tuple((0, 1) for _ in range(n_assets))
        w0 = np.ones(n_assets) / n_assets
        res = minimize(lambda w: portfolio_stats(w, mean_ret, cov)[1],
                       w0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-9, "maxiter": 500})
        if res.success:
            p_r, p_v = portfolio_stats(res.x, mean_ret, cov)
            results.append({"ret": p_r, "vol": p_v,
                            "sharpe": (p_r - 0.05) / (p_v + 1e-9)})
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def sidebar(df: pd.DataFrame):
    st.sidebar.markdown("""
    <div style='padding: 1rem 0.5rem 0.5rem 0.5rem;'>
        <div style='font-size: 1.1rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;'>
            📈 Global Market<br>Intelligence
        </div>
        <div style='font-size: 0.7rem; color: #475569; margin-top: 0.25rem;'>
            Multi-sector equity analytics platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Executive Overview",
         "📈 Price & Volume",
         "🔍 EDA & Statistics",
         "📐 Risk & Portfolio",
         "⚠️ Anomaly Radar",
         "🔮 Forecasting",
         "🏦 Fundamentals",
         "📋 Model Report"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()

    all_sectors = sorted(df["sector"].dropna().unique().tolist()) if not df.empty else []
    sectors = st.sidebar.multiselect("Sectors", all_sectors, default=all_sectors)

    all_tickers = sorted(df[df["sector"].isin(sectors)]["ticker"].unique().tolist()) if not df.empty and sectors else []
    tickers = st.sidebar.multiselect("Tickers", all_tickers, default=all_tickers)

    min_date = df["date"].min().date() if not df.empty else None
    max_date = df["date"].max().date() if not df.empty else None
    if min_date and max_date:
        date_range = st.sidebar.date_input("Date range", (min_date, max_date),
                                            min_value=min_date, max_value=max_date)
    else:
        date_range = None

    st.sidebar.divider()
    st.sidebar.markdown("""
    <div style='font-size: 0.65rem; color: #334155; line-height: 1.6;'>
        <b style='color:#475569'>Data:</b> Yahoo Finance (synthetic demo)<br>
        <b style='color:#475569'>Models:</b> ARIMA · XGBoost · IsolationForest · DBSCAN<br>
        <b style='color:#475569'>Universe:</b> 36 tickers · 6 sectors · 5 years
    </div>
    """, unsafe_allow_html=True)

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
# KPI card helper
# ─────────────────────────────────────────────────────────────────────────────

def kpi(label, value, delta=None, icon="", delta_suffix=""):
    if delta is None:
        delta_html = f'<div class="kpi-delta-neu">—</div>'
    elif isinstance(delta, (int, float)):
        cls = "kpi-delta-pos" if delta >= 0 else "kpi-delta-neg"
        sign = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="{cls}">{sign} {abs(delta):.2f}{delta_suffix}</div>'
    else:
        delta_html = f'<div class="kpi-delta-neu">{delta}</div>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Executive Overview
# ─────────────────────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame, anom_df: pd.DataFrame, metrics: dict) -> None:
    page_header("Executive Overview", "Real-time market intelligence across 6 sectors · 36 tickers · 5-year dataset")

    if df.empty:
        st.warning("No data loaded. Run the pipeline first: `python -m src.pipeline`")
        return

    # ── KPI row ────────────────────────────────────────────────────────────
    cols = st.columns(5, gap="small")
    ann_rets = df.groupby("ticker")["daily_return"].mean() * 252 * 100
    avg_ret  = ann_rets.mean()
    avg_vol  = (df.groupby("ticker")["vol_20d"].mean()).mean() * 100 if "vol_20d" in df.columns else 0
    n_anom   = int(anom_df["is_anomaly"].sum()) if not anom_df.empty and "is_anomaly" in anom_df.columns else 0
    sharpe_x = avg_ret / (avg_vol + 1e-9)

    with cols[0]: kpi("Tickers Tracked", df["ticker"].nunique(), icon="🏢",
                      delta=f"{df['sector'].nunique()} sectors", delta_suffix="")
    with cols[1]: kpi("Avg Annual Return", f"{avg_ret:.1f}%", delta=avg_ret, icon="📈", delta_suffix="%")
    with cols[2]: kpi("Avg Annualised Vol", f"{avg_vol:.1f}%", delta=None, icon="📊")
    with cols[3]: kpi("Portfolio Sharpe", f"{sharpe_x:.2f}", delta=sharpe_x - 1.0, icon="⚡", delta_suffix="")
    xgb_mape = metrics.get("xgb_cv_mape_mean")
    with cols[4]: kpi("Anomalies Detected", f"{n_anom:,}",
                      delta=f"XGB MAPE: {xgb_mape:.0f}%" if isinstance(xgb_mape, float) else f"{n_anom/len(anom_df)*100:.2f}% rate" if not anom_df.empty else None, icon="⚠️")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── row 1: sector heatmap + donut ──────────────────────────────────────
    col_l, col_r = st.columns([3, 2], gap="medium")

    with col_l:
        section("Sector Annual Returns (%)")
        df2 = df.copy()
        df2["year"] = df2["date"].dt.year
        sector_yr = (
            df2.groupby(["sector", "year"])["daily_return"]
            .mean().mul(252 * 100).reset_index()
            .rename(columns={"daily_return": "annual_return"})
        )
        pivot = sector_yr.pivot(index="sector", columns="year", values="annual_return").round(1)
        fig = px.imshow(pivot, color_continuous_scale="RdYlGn", aspect="auto",
                        labels={"color": "Return (%)"}, text_auto=True,
                        zmin=-30, zmax=60)
        fig.update_traces(textfont_size=11)
        fig.update_coloraxes(colorbar=dict(thickness=10, len=0.8, tickfont=dict(size=9, color="#94a3b8")))
        apply_theme(fig, height=280)
        fig.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section("Sector Composition")
        sec_cnt = df.groupby("sector")["ticker"].nunique().reset_index()
        fig = px.pie(sec_cnt, names="sector", values="ticker", hole=0.62,
                     color="sector", color_discrete_map=SECTOR_PALETTE)
        fig.update_traces(textinfo="label+percent", textfont_size=10,
                          marker=dict(line=dict(color="#0a0e1a", width=2)))
        apply_theme(fig, height=280, show_legend=False)
        fig.update_layout(annotations=[dict(text=f"<b>{sec_cnt['ticker'].sum()}</b><br><span style='font-size:10px'>tickers</span>",
                                             x=0.5, y=0.5, font_size=16, font_color="#f1f5f9",
                                             showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    # ── row 2: top/bottom movers + market breadth ─────────────────────────
    section("Performance Leaderboard")
    ytd_ret = (
        df.groupby("ticker")["daily_return"].mean().mul(252*100).round(2)
        .reset_index().rename(columns={"daily_return": "ann_return_pct"})
        .merge(df[["ticker","sector"]].drop_duplicates(), on="ticker")
        .sort_values("ann_return_pct", ascending=False)
    )

    cl, cr = st.columns(2, gap="medium")
    with cl:
        top5 = ytd_ret.head(5).copy()
        fig = go.Figure(go.Bar(
            x=top5["ann_return_pct"], y=top5["ticker"], orientation="h",
            marker=dict(
                color=top5["ann_return_pct"],
                colorscale=[[0, "#0f4c2a"], [1, "#10b981"]],
                line=dict(width=0)),
            text=[f"{v:.1f}%" for v in top5["ann_return_pct"]],
            textposition="outside", textfont=dict(color="#10b981", size=11),
        ))
        apply_theme(fig, height=200, title="Top 5 Performers")
        fig.update_layout(xaxis_title="", yaxis_title="",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        bot5 = ytd_ret.tail(5).copy().sort_values("ann_return_pct")
        fig = go.Figure(go.Bar(
            x=bot5["ann_return_pct"], y=bot5["ticker"], orientation="h",
            marker=dict(
                color=bot5["ann_return_pct"],
                colorscale=[[0, "#f43f5e"], [1, "#4c0519"]],
                line=dict(width=0)),
            text=[f"{v:.1f}%" for v in bot5["ann_return_pct"]],
            textposition="outside", textfont=dict(color="#f43f5e", size=11),
        ))
        apply_theme(fig, height=200, title="Bottom 5 Performers")
        fig.update_layout(xaxis_title="", yaxis_title="",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # ── row 3: cumulative returns by sector ────────────────────────────────
    section("Cumulative Return by Sector")
    df_sorted = df.sort_values(["sector","date"])
    sector_ret = df_sorted.groupby(["date","sector"])["daily_return"].mean().reset_index()
    sector_ret["cum_ret"] = sector_ret.groupby("sector")["daily_return"].transform(
        lambda x: (1+x).cumprod() - 1) * 100

    fig = px.line(sector_ret, x="date", y="cum_ret", color="sector",
                  color_discrete_map=SECTOR_PALETTE,
                  labels={"cum_ret": "Cumulative Return (%)", "date": ""})
    fig.update_traces(line_width=1.8)
    apply_theme(fig, height=280)
    fig.add_hline(y=0, line_dash="dot", line_color="#334155", line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── row 4: market breadth + rolling sharpe heatmap ─────────────────────
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        section("Market Breadth — % Tickers Above 20-day MA")
        if "roll_mean_20d" in df.columns:
            breadth = (
                df.assign(above_ma=df["close"] > df["roll_mean_20d"])
                .groupby("date")["above_ma"].mean().mul(100)
                .reset_index().rename(columns={"above_ma": "pct"})
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=breadth["date"], y=breadth["pct"], mode="lines",
                line=dict(color="#3b82f6", width=1.5),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.08)", name="Breadth"))
            fig.add_hline(y=50, line_dash="dash", line_color="#475569", line_width=1)
            fig.add_hline(y=80, line_dash="dot",  line_color="#10b981", line_width=1,
                          annotation_text="Overbought", annotation_font_color="#10b981")
            fig.add_hline(y=20, line_dash="dot",  line_color="#f43f5e", line_width=1,
                          annotation_text="Oversold", annotation_font_color="#f43f5e")
            apply_theme(fig, height=240)
            fig.update_layout(yaxis_title="% Above 20d MA", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("Monthly Return Heatmap (All Sectors Avg)")
        df_m = df.copy()
        df_m["year"]  = df_m["date"].dt.year
        df_m["month"] = df_m["date"].dt.month
        monthly_ret = (
            df_m.groupby(["year","month"])["daily_return"].mean()
            .mul(21*100).reset_index()
            .rename(columns={"daily_return": "monthly_ret"})
        )
        pivot_m = monthly_ret.pivot(index="year", columns="month", values="monthly_ret").round(2)
        pivot_m.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot_m.columns)]
        fig = px.imshow(pivot_m, color_continuous_scale="RdYlGn", aspect="auto",
                        text_auto=True, zmin=-8, zmax=8)
        fig.update_traces(textfont_size=9)
        apply_theme(fig, height=240)
        fig.update_layout(xaxis_title="", yaxis_title="",
                          coloraxis_colorbar=dict(thickness=8, len=0.7, tickfont=dict(size=8)))
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Price & Volume
# ─────────────────────────────────────────────────────────────────────────────

def page_price_volume(df: pd.DataFrame) -> None:
    page_header("Price & Volume Analysis", "OHLCV candlestick · technical indicators · multi-ticker comparison")
    if df.empty:
        st.warning("No data. Run pipeline."); return

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
    with col_ctrl1:
        ticker_sel = st.selectbox("Ticker", sorted(df["ticker"].unique()), key="pv_ticker")
    with col_ctrl2:
        lookback = st.selectbox("Lookback", ["3M","6M","1Y","2Y","All"], index=2, key="pv_lb")
    with col_ctrl3:
        show_bb = st.toggle("Bollinger Bands", value=True)

    sub = df[df["ticker"] == ticker_sel].sort_values("date")
    lookback_days = {"3M":63,"6M":126,"1Y":252,"2Y":504,"All":len(sub)}[lookback]
    sub = sub.tail(lookback_days)

    # ── candlestick + volume ───────────────────────────────────────────────
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.58, 0.22, 0.20], vertical_spacing=0.02)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=sub["date"], open=sub["open"], high=sub["high"],
        low=sub["low"], close=sub["close"], name="OHLC",
        increasing=dict(line_color="#10b981", fillcolor="#10b981"),
        decreasing=dict(line_color="#f43f5e", fillcolor="#f43f5e"),
    ), row=1, col=1)

    # Bollinger Bands
    if show_bb and "bb_upper" in sub.columns:
        for col_n, dash, color, label in [
            ("bb_upper", "dash", "rgba(59,130,246,0.5)", "BB Upper"),
            ("bb_lower", "dash", "rgba(59,130,246,0.5)", "BB Lower"),
            ("bb_middle","dot",  "rgba(59,130,246,0.8)", "BB Middle"),
        ]:
            fig.add_trace(go.Scatter(x=sub["date"], y=sub[col_n],
                                     line=dict(dash=dash, color=color, width=1),
                                     name=label, showlegend=True), row=1, col=1)
        # fill between bands
        fig.add_trace(go.Scatter(
            x=pd.concat([sub["date"], sub["date"][::-1]]),
            y=pd.concat([sub["bb_upper"], sub["bb_lower"][::-1]]),
            fill="toself", fillcolor="rgba(59,130,246,0.04)",
            line=dict(color="rgba(0,0,0,0)"), name="BB Fill", showlegend=False,
        ), row=1, col=1)

    # Volume bars
    colors_vol = ["#10b981" if r >= 0 else "#f43f5e"
                  for r in sub["daily_return"].fillna(0)]
    fig.add_trace(go.Bar(x=sub["date"], y=sub["volume"], name="Volume",
                         marker_color=colors_vol, opacity=0.7, showlegend=False), row=2, col=1)

    # Moving averages
    if "roll_mean_20d" in sub.columns:
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["roll_mean_20d"],
                                 line=dict(color="#f59e0b", width=1, dash="dot"),
                                 name="MA20", showlegend=True), row=1, col=1)
    if "roll_mean_60d" in sub.columns:
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["roll_mean_60d"],
                                 line=dict(color="#a855f7", width=1, dash="dot"),
                                 name="MA60", showlegend=True), row=1, col=1)

    # ATR row
    if "atr" in sub.columns:
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["atr"], mode="lines",
                                 line=dict(color="#f59e0b", width=1),
                                 name="ATR", fill="tozeroy",
                                 fillcolor="rgba(245,158,11,0.08)"), row=3, col=1)

    apply_theme(fig, height=560)
    fig.update_layout(xaxis_rangeslider_visible=False,
                      title=dict(text=f"<b>{ticker_sel}</b> — OHLCV Chart",
                                 font=dict(size=14, color="#e2e8f0"), x=0.01))
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1, title_font_size=10)
    fig.update_yaxes(title_text="Volume",      row=2, col=1, title_font_size=10)
    fig.update_yaxes(title_text="ATR",         row=3, col=1, title_font_size=10)
    st.plotly_chart(fig, use_container_width=True)

    # ── technical indicators: RSI + MACD ──────────────────────────────────
    if "rsi_14" in sub.columns and "macd_line" in sub.columns:
        section("Technical Indicators")
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=["RSI (14)", "MACD (12/26/9)"],
                             vertical_spacing=0.06)
        # RSI
        fig2.add_trace(go.Scatter(x=sub["date"], y=sub["rsi_14"],
                                  line=dict(color="#f59e0b", width=1.5), name="RSI"), row=1, col=1)
        fig2.add_hrect(y0=70, y1=100, fillcolor="rgba(244,63,94,0.07)", line_width=0, row=1, col=1)
        fig2.add_hrect(y0=0, y1=30, fillcolor="rgba(16,185,129,0.07)", line_width=0, row=1, col=1)
        fig2.add_hline(y=70, line_dash="dash", line_color="#f43f5e", line_width=1, row=1, col=1)
        fig2.add_hline(y=30, line_dash="dash", line_color="#10b981", line_width=1, row=1, col=1)
        fig2.add_hline(y=50, line_dash="dot",  line_color="#334155", line_width=1, row=1, col=1)
        # MACD
        hist_colors = ["#10b981" if v >= 0 else "#f43f5e" for v in sub["macd_hist"].fillna(0)]
        fig2.add_trace(go.Bar(x=sub["date"], y=sub["macd_hist"], name="Histogram",
                              marker_color=hist_colors, opacity=0.6), row=2, col=1)
        fig2.add_trace(go.Scatter(x=sub["date"], y=sub["macd_line"],
                                  line=dict(color="#3b82f6", width=1.5), name="MACD"), row=2, col=1)
        fig2.add_trace(go.Scatter(x=sub["date"], y=sub["macd_signal"],
                                  line=dict(color="#f43f5e", width=1.5, dash="dash"), name="Signal"), row=2, col=1)
        fig2.add_hline(y=0, line_color="#334155", line_width=1, row=2, col=1)
        apply_theme(fig2, height=360)
        st.plotly_chart(fig2, use_container_width=True)

    # ── multi-ticker comparison ────────────────────────────────────────────
    section("Normalised Price Comparison (Base = 100)")
    compare_tickers = st.multiselect("Compare Tickers", sorted(df["ticker"].unique()),
                                     default=[ticker_sel] + sorted(df["ticker"].unique())[:3],
                                     key="pv_compare")
    if compare_tickers:
        fig3 = go.Figure()
        colors_seq = ["#3b82f6","#10b981","#f43f5e","#f59e0b","#a855f7","#64748b","#06b6d4","#ec4899"]
        for i, t in enumerate(compare_tickers):
            sub_t = df[df["ticker"] == t].sort_values("date").tail(lookback_days)
            if sub_t.empty: continue
            norm = sub_t["close"] / sub_t["close"].iloc[0] * 100
            fig3.add_trace(go.Scatter(x=sub_t["date"], y=norm,
                                      mode="lines", name=t,
                                      line=dict(color=colors_seq[i % len(colors_seq)], width=1.8)))
        fig3.add_hline(y=100, line_dash="dot", line_color="#334155", line_width=1)
        apply_theme(fig3, height=300)
        fig3.update_layout(yaxis_title="Indexed Price (Base=100)", xaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — EDA & Statistics
# ─────────────────────────────────────────────────────────────────────────────

def page_eda(df: pd.DataFrame) -> None:
    page_header("EDA & Statistics", "Return distributions · correlation analysis · volatility & risk decomposition")
    if df.empty:
        st.warning("No data. Run pipeline."); return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Return Distributions", "🔗 Correlations", "🌋 Volatility Surface",
        "📉 Rolling Metrics", "📋 Descriptive Stats"])

    with tab1:
        col_l, col_r = st.columns([3, 2], gap="medium")
        with col_l:
            section("Daily Return Violin by Sector")
            fig = px.violin(df.dropna(subset=["daily_return"]), x="sector", y="daily_return",
                            color="sector", box=True, points=False,
                            color_discrete_map=SECTOR_PALETTE,
                            labels={"daily_return": "Daily Return", "sector": ""})
            fig.update_traces(meanline_visible=True)
            apply_theme(fig, height=360, show_legend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            section("Return Distribution & Fat Tails")
            ticker_sel = st.selectbox("Ticker", sorted(df["ticker"].unique()), key="eda_ticker")
            r = df[df["ticker"]==ticker_sel]["daily_return"].dropna()
            x_range = np.linspace(r.min(), r.max(), 200)
            mu, sigma = r.mean(), r.std()
            normal_pdf = scipy_stats.norm.pdf(x_range, mu, sigma)

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=r, nbinsx=80, name="Returns",
                                       histnorm="probability density",
                                       marker=dict(color="#3b82f6", opacity=0.7, line_width=0)))
            fig.add_trace(go.Scatter(x=x_range, y=normal_pdf, mode="lines",
                                     line=dict(color="#f43f5e", width=2, dash="dash"),
                                     name="Normal Fit"))
            apply_theme(fig, height=300, show_legend=True)
            kurt = float(r.kurtosis())
            skew = float(r.skew())
            fig.add_annotation(text=f"Kurt: {kurt:.2f}  |  Skew: {skew:.2f}",
                                x=0.98, y=0.95, xref="paper", yref="paper",
                                showarrow=False, font=dict(size=10, color="#94a3b8"),
                                align="right")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        section("Cross-Ticker Return Correlation Heatmap")
        pivot = df.pivot_table(index="date", columns="ticker", values="daily_return")
        corr  = pivot.corr().round(2)

        # cluster by sector
        sector_order = df[["ticker","sector"]].drop_duplicates().set_index("ticker")["sector"]
        ordered_tickers = corr.index[
            np.argsort([list(SECTOR_PALETTE.keys()).index(sector_order.get(t, "Industrials"))
                        for t in corr.index])
        ]
        corr = corr.loc[ordered_tickers, ordered_tickers]

        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        aspect="auto", text_auto=False)
        fig.update_traces(hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>ρ = %{z:.2f}<extra></extra>")
        apply_theme(fig, height=520)
        fig.update_layout(coloraxis_colorbar=dict(thickness=10, len=0.7,
                                                   title=dict(text="ρ", font=dict(size=10))))
        st.plotly_chart(fig, use_container_width=True)

        # correlation distribution
        section("Pairwise Correlation Distribution")
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        pair_corrs = corr.values[mask]
        fig2 = go.Figure(go.Histogram(x=pair_corrs, nbinsx=40,
                                       marker=dict(color="#3b82f6", opacity=0.8, line_width=0),
                                       histnorm="probability density"))
        fig2.add_vline(x=np.mean(pair_corrs), line_dash="dash",
                       line_color="#f59e0b", annotation_text=f"Mean: {np.mean(pair_corrs):.2f}")
        apply_theme(fig2, height=220)
        fig2.update_layout(xaxis_title="Correlation Coefficient", yaxis_title="Density")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        section("Annualised Volatility Surface (Sector × Month)")
        if "vol_20d" in df.columns:
            df_v = df.copy()
            df_v["month"] = df_v["date"].dt.to_period("M").dt.to_timestamp()
            monthly_vol = (
                df_v.groupby(["month","sector"])["vol_20d"]
                .mean().mul(100).reset_index()
            )
            fig = px.area(monthly_vol, x="month", y="vol_20d", color="sector",
                          color_discrete_map=SECTOR_PALETTE,
                          labels={"vol_20d": "Annualised Vol (%)", "month": ""},
                          line_group="sector")
            fig.update_traces(line_width=1.5)
            apply_theme(fig, height=340)
            st.plotly_chart(fig, use_container_width=True)

        section("Volatility Regime Scatter (Ann. Return vs Volatility)")
        tickers_u = df["ticker"].unique()
        scatter_data = []
        for t in tickers_u:
            sub = df[df["ticker"]==t]["daily_return"].dropna()
            scatter_data.append({
                "ticker": t,
                "sector": df[df["ticker"]==t]["sector"].iloc[0],
                "ann_ret": sub.mean()*252*100,
                "ann_vol": sub.std()*np.sqrt(252)*100,
                "sharpe":  (sub.mean()-0.05/252)/(sub.std()+1e-9)*np.sqrt(252),
            })
        sdf = pd.DataFrame(scatter_data)
        sdf["sharpe_size"] = sdf["sharpe"].abs().clip(lower=0.05)
        sdf["sharpe_label"] = sdf["sharpe"].round(2).astype(str)
        fig3 = px.scatter(sdf, x="ann_vol", y="ann_ret", color="sector", text="ticker",
                          size="sharpe_size", size_max=20,
                          color_discrete_map=SECTOR_PALETTE,
                          labels={"ann_vol":"Annual Volatility (%)", "ann_ret":"Annual Return (%)"},
                          hover_data={"sharpe_size": False, "sharpe_label": True})
        fig3.update_traces(textposition="top center", textfont_size=8)
        fig3.add_hline(y=0, line_dash="dot", line_color="#334155")
        apply_theme(fig3, height=380)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        section("20-Day Rolling Sharpe Ratio")
        rf = 0.05 / 252
        df2 = df.sort_values(["ticker","date"]).copy()
        df2["excess"] = df2["daily_return"] - rf
        df2["roll_sharpe"] = df2.groupby("ticker")["excess"].transform(
            lambda x: x.rolling(20, min_periods=10).mean() /
                      (x.rolling(20, min_periods=10).std() + 1e-9) * np.sqrt(252))
        tickers_sel = st.multiselect("Tickers", df["ticker"].unique().tolist(),
                                     default=list(df["ticker"].unique()[:6]), key="eda_sharpe")
        sub = df2[df2["ticker"].isin(tickers_sel)]
        fig = px.line(sub, x="date", y="roll_sharpe", color="ticker",
                      labels={"roll_sharpe": "Rolling Sharpe", "date": ""})
        fig.update_traces(line_width=1.5)
        fig.add_hline(y=1, line_dash="dash", line_color="#f59e0b",
                      annotation_text="Sharpe = 1", annotation_font_color="#f59e0b")
        fig.add_hline(y=0, line_dash="dot", line_color="#334155")
        apply_theme(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)

        section("Drawdown Chart")
        tickers_dd = st.multiselect("Tickers", df["ticker"].unique().tolist(),
                                    default=list(df["ticker"].unique()[:4]), key="eda_dd")
        fig4 = go.Figure()
        colors_dd = ["#3b82f6","#10b981","#f43f5e","#f59e0b","#a855f7"]
        for i, t in enumerate(tickers_dd):
            sub_t = df[df["ticker"]==t].sort_values("date")
            dd = max_drawdown_series(sub_t["daily_return"]) * 100
            fig4.add_trace(go.Scatter(x=sub_t["date"], y=dd, mode="lines", name=t,
                                      line=dict(color=colors_dd[i % len(colors_dd)], width=1.5),
                                      fill="tozeroy",
                                      fillcolor=f"rgba({int(colors_dd[i%len(colors_dd)][1:3],16)},"
                                                f"{int(colors_dd[i%len(colors_dd)][3:5],16)},"
                                                f"{int(colors_dd[i%len(colors_dd)][5:7],16)},0.06)"))
        apply_theme(fig4, height=280)
        fig4.update_layout(yaxis_title="Drawdown (%)", xaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)

    with tab5:
        section("Sector-Level Descriptive Statistics")
        grp = df.groupby("sector")["daily_return"]
        stats = pd.DataFrame({
            "Ann. Return %":   (grp.mean() * 252 * 100).round(2),
            "Ann. Vol %":      (grp.std() * np.sqrt(252) * 100).round(2),
            "Skewness":        grp.apply(lambda x: float(x.skew())).round(3),
            "Excess Kurtosis": grp.apply(lambda x: float(x.kurtosis())).round(3),
            "Min Daily %":     (grp.min() * 100).round(3),
            "Max Daily %":     (grp.max() * 100).round(3),
            "Observations":    grp.count().astype(int),
        })
        st.dataframe(stats.style
                     .background_gradient(cmap="RdYlGn", subset=["Ann. Return %"])
                     .background_gradient(cmap="YlOrRd_r", subset=["Ann. Vol %"])
                     .format({"Ann. Return %": "{:.2f}","Ann. Vol %": "{:.2f}",
                              "Skewness": "{:.3f}","Excess Kurtosis": "{:.3f}"}),
                     use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Risk & Portfolio
# ─────────────────────────────────────────────────────────────────────────────

def page_risk_portfolio(df: pd.DataFrame) -> None:
    page_header("Risk & Portfolio Analytics",
                "VaR · CVaR · max drawdown · portfolio optimization · efficient frontier")
    if df.empty:
        st.warning("No data. Run pipeline."); return

    tab1, tab2, tab3 = st.tabs(["📉 Risk Metrics", "🎯 Portfolio Optimizer", "🏔️ Efficient Frontier"])

    with tab1:
        section("Per-Ticker Risk Scorecard")
        all_tickers = sorted(df["ticker"].unique())
        risk_rows = []
        for t in all_tickers:
            r = df[df["ticker"]==t]["daily_return"].dropna()
            if len(r) < 20: continue
            m = compute_risk_metrics(r)
            risk_rows.append({
                "Ticker": t,
                "Sector": df[df["ticker"]==t]["sector"].iloc[0],
                "Ann. Return %":  round(m["ann_ret"]*100, 2),
                "Ann. Vol %":     round(m["ann_vol"]*100, 2),
                "Sharpe":         round(m["sharpe"], 3),
                "Sortino":        round(m["sortino"], 3),
                "Max Drawdown %": round(m["max_dd"]*100, 2),
                "VaR 95% (daily)":  f"{m['var_95']*100:.3f}%",
                "CVaR 95% (daily)": f"{m['cvar_95']*100:.3f}%",
                "Calmar":           round(m["calmar"], 3),
            })
        risk_df = pd.DataFrame(risk_rows)
        st.dataframe(
            risk_df.sort_values("Sharpe", ascending=False)
            .style.background_gradient(cmap="RdYlGn", subset=["Ann. Return %","Sharpe"])
                  .background_gradient(cmap="YlOrRd_r", subset=["Ann. Vol %","Max Drawdown %"]),
            use_container_width=True, hide_index=True
        )

        section("VaR & CVaR Deep Dive")
        var_ticker = st.selectbox("Select Ticker", all_tickers, key="risk_var")
        r_var = df[df["ticker"]==var_ticker]["daily_return"].dropna()
        var95  = np.percentile(r_var, 5)
        cvar95 = r_var[r_var <= var95].mean()
        var99  = np.percentile(r_var, 1)
        cvar99 = r_var[r_var <= var99].mean()

        cv1, cv2, cv3, cv4 = st.columns(4, gap="small")
        with cv1: kpi("VaR 95%",  f"{var95*100:.3f}%",  delta=None, icon="📉")
        with cv2: kpi("CVaR 95%", f"{cvar95*100:.3f}%", delta=None, icon="🔥")
        with cv3: kpi("VaR 99%",  f"{var99*100:.3f}%",  delta=None, icon="📉")
        with cv4: kpi("CVaR 99%", f"{cvar99*100:.3f}%", delta=None, icon="🔥")

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=r_var*100, nbinsx=80, name="Returns",
                                    histnorm="probability density",
                                    marker=dict(color="#3b82f6", opacity=0.7, line_width=0)))
        # shade tail
        tail_x = r_var[r_var <= var95] * 100
        fig.add_trace(go.Histogram(x=tail_x, nbinsx=30, name="VaR 95% Tail",
                                    histnorm="probability density",
                                    marker=dict(color="#f43f5e", opacity=0.6, line_width=0)))
        fig.add_vline(x=var95*100, line_dash="dash", line_color="#f43f5e",
                      annotation_text=f"VaR 95% = {var95*100:.2f}%",
                      annotation_font_color="#f43f5e")
        fig.add_vline(x=cvar95*100, line_dash="dot", line_color="#f59e0b",
                      annotation_text=f"CVaR = {cvar95*100:.2f}%",
                      annotation_font_color="#f59e0b")
        apply_theme(fig, height=320)
        fig.update_layout(xaxis_title="Daily Return (%)", yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)

        section("Rolling Max Drawdown by Ticker")
        dd_tickers = st.multiselect("Tickers", all_tickers,
                                    default=all_tickers[:5], key="risk_dd")
        fig2 = go.Figure()
        clrs = ["#3b82f6","#10b981","#f43f5e","#f59e0b","#a855f7","#06b6d4"]
        for i, t in enumerate(dd_tickers):
            sub_t = df[df["ticker"]==t].sort_values("date")
            dd = max_drawdown_series(sub_t["daily_return"]) * 100
            fig2.add_trace(go.Scatter(x=sub_t["date"], y=dd, mode="lines", name=t,
                                      line=dict(color=clrs[i%len(clrs)], width=1.6)))
        apply_theme(fig2, height=300)
        fig2.update_layout(yaxis_title="Drawdown (%)", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        section("Portfolio Weight Optimizer")
        col_a, col_b = st.columns([1, 2])
        with col_a:
            port_tickers = st.multiselect("Select Tickers (4–12)",
                                          sorted(df["ticker"].unique()),
                                          default=sorted(df["ticker"].unique())[:8],
                                          key="port_tickers")
            opt_goal = st.radio("Optimisation Goal",
                                ["Max Sharpe", "Min Volatility", "Equal Weight"],
                                index=0, key="port_goal")
            rf_rate  = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.5) / 100

        if len(port_tickers) < 2:
            st.info("Select at least 2 tickers."); return

        ret_pivot = df[df["ticker"].isin(port_tickers)].pivot_table(
            index="date", columns="ticker", values="daily_return")
        ret_pivot = ret_pivot.dropna()
        mean_ret = ret_pivot.mean()
        cov      = ret_pivot.cov()
        n        = len(port_tickers)
        w0       = np.ones(n) / n
        bounds   = tuple((0.01, 0.50) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        if opt_goal == "Max Sharpe":
            def neg_sharpe(w):
                p_r, p_v = portfolio_stats(w, mean_ret.values, cov.values)
                return -((p_r - rf_rate) / (p_v + 1e-9))
            res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                           constraints=constraints, options={"ftol":1e-9,"maxiter":500})
            opt_w = res.x if res.success else w0
        elif opt_goal == "Min Volatility":
            def port_vol(w):
                return portfolio_stats(w, mean_ret.values, cov.values)[1]
            res = minimize(port_vol, w0, method="SLSQP", bounds=bounds,
                           constraints=constraints, options={"ftol":1e-9,"maxiter":500})
            opt_w = res.x if res.success else w0
        else:
            opt_w = w0

        p_ret, p_vol = portfolio_stats(opt_w, mean_ret.values, cov.values)
        p_sharpe = (p_ret - rf_rate) / (p_vol + 1e-9)

        with col_b:
            kc1, kc2, kc3 = st.columns(3)
            with kc1: kpi("Portfolio Return",   f"{p_ret*100:.2f}%",  delta=p_ret*100, icon="📈", delta_suffix="%")
            with kc2: kpi("Portfolio Volatility",f"{p_vol*100:.2f}%", delta=None, icon="📊")
            with kc3: kpi("Portfolio Sharpe",    f"{p_sharpe:.3f}",   delta=p_sharpe-1, icon="⚡")

        weight_df = pd.DataFrame({"Ticker": port_tickers,
                                   "Weight %": (opt_w * 100).round(2)}).sort_values("Weight %", ascending=False)

        col_c, col_d = st.columns([1,2])
        with col_c:
            st.dataframe(weight_df.style.bar(subset=["Weight %"], color="#3b82f6"),
                         use_container_width=True, hide_index=True)
        with col_d:
            fig = px.bar(weight_df, x="Ticker", y="Weight %",
                         color="Weight %", color_continuous_scale="Blues",
                         labels={"Weight %": "Allocation (%)"})
            fig.update_traces(marker_line_width=0)
            apply_theme(fig, height=260, title=f"{opt_goal} — Optimal Weights")
            fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Weight (%)")
            st.plotly_chart(fig, use_container_width=True)

        # portfolio cumulative return
        section("Optimised Portfolio — Cumulative Return vs Equal Weight")
        port_ret_ts   = (ret_pivot[port_tickers] * opt_w).sum(axis=1)
        equal_ret_ts  = ret_pivot[port_tickers].mean(axis=1)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=port_ret_ts.index,
                                   y=((1+port_ret_ts).cumprod()-1)*100,
                                   mode="lines", name=f"{opt_goal}",
                                   line=dict(color="#3b82f6", width=2)))
        fig2.add_trace(go.Scatter(x=equal_ret_ts.index,
                                   y=((1+equal_ret_ts).cumprod()-1)*100,
                                   mode="lines", name="Equal Weight",
                                   line=dict(color="#64748b", width=1.5, dash="dash")))
        fig2.add_hline(y=0, line_dash="dot", line_color="#334155")
        apply_theme(fig2, height=280)
        fig2.update_layout(yaxis_title="Cumulative Return (%)", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        section("Efficient Frontier")
        ef_tickers = st.multiselect("Tickers for Frontier",
                                    sorted(df["ticker"].unique()),
                                    default=sorted(df["ticker"].unique())[:10],
                                    key="ef_tickers")
        if len(ef_tickers) < 3:
            st.info("Select at least 3 tickers."); return

        ret_p = df[df["ticker"].isin(ef_tickers)].pivot_table(
            index="date", columns="ticker", values="daily_return").dropna()
        mr   = ret_p.mean()
        cv   = ret_p.cov()

        with st.spinner("Computing efficient frontier…"):
            ef_df = efficient_frontier_points(mr.values, cv.values, n=50)

        if ef_df.empty:
            st.warning("Optimisation did not converge."); return

        # individual tickers
        ind = []
        for t in ef_tickers:
            r = df[df["ticker"]==t]["daily_return"].dropna()
            ind.append({"ticker":t, "ret":r.mean()*252*100, "vol":r.std()*np.sqrt(252)*100,
                        "sharpe":(r.mean()-0.05/252)/(r.std()+1e-9)*np.sqrt(252)})
        ind_df = pd.DataFrame(ind)

        fig = go.Figure()
        # scatter individual tickers
        fig.add_trace(go.Scatter(
            x=ind_df["vol"], y=ind_df["ret"],
            mode="markers+text", text=ind_df["ticker"],
            textposition="top center", textfont=dict(size=8, color="#94a3b8"),
            marker=dict(size=8, color=ind_df["sharpe"], colorscale="RdYlGn",
                        showscale=True, colorbar=dict(title="Sharpe",thickness=8,len=0.5),
                        line=dict(color="#0a0e1a", width=1)),
            name="Tickers",
        ))
        # frontier line
        ef_sorted = ef_df.sort_values("vol")
        fig.add_trace(go.Scatter(
            x=ef_sorted["vol"]*100, y=ef_sorted["ret"]*100,
            mode="lines", line=dict(color="#3b82f6", width=2.5),
            name="Efficient Frontier",
        ))
        # max sharpe point
        max_s_row = ef_df.loc[ef_df["sharpe"].idxmax()]
        fig.add_trace(go.Scatter(
            x=[max_s_row["vol"]*100], y=[max_s_row["ret"]*100],
            mode="markers", marker=dict(size=14, color="#f59e0b", symbol="star",
                                        line=dict(color="#0a0e1a",width=1)),
            name="Max Sharpe",
        ))
        apply_theme(fig, height=460, title="Efficient Frontier (Markowitz)")
        fig.update_layout(xaxis_title="Annualised Volatility (%)",
                          yaxis_title="Annualised Return (%)")
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 5 — Anomaly Radar
# ─────────────────────────────────────────────────────────────────────────────

def page_anomaly(anom_df: pd.DataFrame) -> None:
    page_header("Anomaly Radar", "IsolationForest · DBSCAN · ARIMA residual · ensemble voting")
    if anom_df.empty or "is_anomaly" not in anom_df.columns:
        st.warning("No anomaly data. Run pipeline."); return

    anom = anom_df[anom_df["is_anomaly"]].copy()
    total = len(anom_df)

    # ── KPI row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1: kpi("Total Anomalies", f"{len(anom):,}", delta=len(anom)/total*100, icon="⚠️", delta_suffix="% of data")
    with c2:
        if_c = int(anom_df.get("if_anomaly", pd.Series(dtype=bool)).sum()) if "if_anomaly" in anom_df.columns else 0
        kpi("IsolationForest", f"{if_c:,}", delta=None, icon="🌲")
    with c3:
        db_c = int(anom_df.get("dbscan_anomaly", pd.Series(dtype=bool)).sum()) if "dbscan_anomaly" in anom_df.columns else 0
        kpi("DBSCAN Noise Pts", f"{db_c:,}", delta=None, icon="🔵")
    with c4:
        r_c = int(anom_df.get("residual_anomaly", pd.Series(dtype=bool)).sum()) if "residual_anomaly" in anom_df.columns else 0
        kpi("ARIMA Residual", f"{r_c:,}", delta=None, icon="📈")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── top bar + severity donut ───────────────────────────────────────────
    col1, col2 = st.columns([3,2], gap="medium")
    with col1:
        section("Anomaly Count by Ticker")
        cnt = anom.groupby("ticker").size().reset_index(name="count").sort_values("count", ascending=False)
        fig = go.Figure(go.Bar(
            x=cnt["count"], y=cnt["ticker"], orientation="h",
            marker=dict(color=cnt["count"], colorscale="YlOrRd",
                        showscale=False, line=dict(width=0)),
            text=cnt["count"], textposition="outside",
            textfont=dict(color="#94a3b8", size=10),
            showlegend=False,
        ))
        apply_theme(fig, height=350, show_legend=False)
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Count", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Severity Breakdown")
        if "anomaly_severity" in anom.columns:
            sev = anom["anomaly_severity"].value_counts().reset_index()
            sev.columns = ["severity","count"]
            sev_colors = {"watch":"#f59e0b","moderate":"#f97316","severe":"#f43f5e","normal":"#10b981"}
            fig = px.pie(sev, names="severity", values="count", hole=0.58,
                         color="severity", color_discrete_map=sev_colors)
            fig.update_traces(textinfo="label+percent", textfont_size=10,
                              marker=dict(line=dict(color="#0a0e1a",width=2)))
            apply_theme(fig, height=280, show_legend=False)
            st.plotly_chart(fig, use_container_width=True)

        section("Method Agreement Heatmap")
        if all(c in anom_df.columns for c in ["if_anomaly","dbscan_anomaly","residual_anomaly"]):
            vote_counts = anom_df.groupby("ticker")[["if_anomaly","dbscan_anomaly","residual_anomaly"]].sum().astype(int)
            vote_counts.columns = ["IForest","DBSCAN","Residual"]
            fig2 = px.imshow(vote_counts.T, color_continuous_scale="YlOrRd",
                             aspect="auto", text_auto=True)
            fig2.update_traces(textfont_size=8)
            apply_theme(fig2, height=200)
            fig2.update_layout(xaxis_title="", yaxis_title="",
                               coloraxis_colorbar=dict(thickness=8,len=0.5))
            st.plotly_chart(fig2, use_container_width=True)

    # ── anomaly timeline ───────────────────────────────────────────────────
    section("Anomaly Timeline")
    ticker_sel = st.selectbox("Ticker", sorted(anom_df["ticker"].unique()), key="anom_ticker")
    sub_a_full  = anom_df[anom_df["ticker"]==ticker_sel].sort_values("date")
    sub_anom    = sub_a_full[sub_a_full["is_anomaly"]]

    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.7, 0.3], vertical_spacing=0.03)
    # price line
    fig3.add_trace(go.Scatter(x=sub_a_full["date"], y=sub_a_full["close"],
                               mode="lines", line=dict(color="#3b82f6",width=1.5), name="Close"), row=1, col=1)
    # anomaly markers sized by severity
    sev_map = {"normal":4,"watch":8,"moderate":12,"severe":16}
    if "anomaly_severity" in sub_anom.columns:
        sizes = sub_anom["anomaly_severity"].map(sev_map).fillna(8).astype(int).tolist()
    else:
        sizes = [10]*len(sub_anom)
    fig3.add_trace(go.Scatter(x=sub_anom["date"], y=sub_anom["close"],
                               mode="markers",
                               marker=dict(color="#f43f5e", size=sizes, symbol="x",
                                           line=dict(color="#f43f5e",width=1.5)),
                               name="Anomaly"), row=1, col=1)
    # anomaly score
    if "if_score" in sub_a_full.columns:
        fig3.add_trace(go.Scatter(x=sub_a_full["date"], y=sub_a_full["if_score"],
                                   mode="lines", line=dict(color="#f59e0b",width=1),
                                   fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                                   name="IF Score"), row=2, col=1)
    apply_theme(fig3, height=420)
    fig3.update_layout(title=dict(text=f"<b>{ticker_sel}</b> — Anomaly Detection Timeline",
                                   font=dict(size=13,color="#e2e8f0"), x=0.01))
    st.plotly_chart(fig3, use_container_width=True)

    # ── anomaly calendar heatmap ───────────────────────────────────────────
    section("Anomaly Density Calendar (All Tickers)")
    anom_cal = anom.copy()
    anom_cal["week"]    = anom_cal["date"].dt.isocalendar().week.astype(int)
    anom_cal["weekday"] = anom_cal["date"].dt.dayofweek
    anom_cal["year"]    = anom_cal["date"].dt.year
    heat = anom_cal.groupby(["year","week","weekday"]).size().reset_index(name="count")
    if not heat.empty:
        year_sel = st.selectbox("Year", sorted(heat["year"].unique(), reverse=True), key="anom_year")
        h_sub = heat[heat["year"]==year_sel]
        pivot_cal = h_sub.pivot_table(index="weekday", columns="week", values="count", fill_value=0)
        pivot_cal.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(pivot_cal)]
        fig4 = px.imshow(pivot_cal, color_continuous_scale="YlOrRd", aspect="auto",
                         labels={"x":"Week of Year","y":"Day","color":"Anomalies"})
        apply_theme(fig4, height=200)
        fig4.update_layout(xaxis_title="Week of Year", yaxis_title="",
                            coloraxis_colorbar=dict(thickness=8,len=0.7))
        st.plotly_chart(fig4, use_container_width=True)

    # ── events table ────────────────────────────────────────────────────────
    section("Recent Anomaly Events")
    cols_show = ["date","ticker","sector","close","daily_return",
                 "anomaly_votes","anomaly_severity","if_anomaly","dbscan_anomaly","residual_anomaly"]
    cols_show = [c for c in cols_show if c in anom.columns]
    st.dataframe(anom[cols_show].sort_values("date", ascending=False).head(200),
                 use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 6 — Forecasting
# ─────────────────────────────────────────────────────────────────────────────

def page_forecasting(df: pd.DataFrame, xgb_fc: pd.DataFrame, arima_fc: pd.DataFrame) -> None:
    page_header("Price Forecasting", "XGBoost iterative forecast · ARIMA with 90% CI · 90-day horizon")

    all_tickers = sorted(set(
        (xgb_fc["ticker"].unique().tolist() if not xgb_fc.empty else []) +
        (arima_fc["ticker"].unique().tolist() if not arima_fc.empty else [])
    ))
    if not all_tickers:
        st.warning("No forecast data. Run pipeline."); return

    col_a, col_b = st.columns([2,1])
    with col_a:
        ticker_sel = st.selectbox("Select Ticker", all_tickers, key="fc_ticker")
    with col_b:
        hist_days  = st.slider("Historical context (days)", 60, 365, 180, key="fc_hist")

    # ── main forecast chart ────────────────────────────────────────────────
    fig = go.Figure()

    # historical price
    if not df.empty:
        hist = df[df["ticker"]==ticker_sel].sort_values("date").tail(hist_days)
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["close"],
                                 mode="lines", line=dict(color="#64748b",width=1.5), name="Historical"))

    # XGBoost forecast
    if not xgb_fc.empty:
        fc_x = xgb_fc[xgb_fc["ticker"]==ticker_sel].sort_values("date")
        fig.add_trace(go.Scatter(x=fc_x["date"], y=fc_x["forecast"],
                                 mode="lines",
                                 line=dict(color="#10b981",width=2.5,dash="dash"),
                                 name="XGBoost Forecast"))

    # ARIMA forecast + CI
    if not arima_fc.empty:
        fc_a = arima_fc[arima_fc["ticker"]==ticker_sel].sort_values("date")
        if "upper_90" in fc_a.columns:
            fig.add_traces([
                go.Scatter(
                    x=pd.concat([fc_a["date"], fc_a["date"][::-1]]),
                    y=pd.concat([fc_a["upper_90"], fc_a["lower_90"][::-1]]),
                    fill="toself", fillcolor="rgba(245,158,11,0.12)",
                    line=dict(color="rgba(0,0,0,0)"), name="ARIMA 90% CI"),
            ])
        fig.add_trace(go.Scatter(x=fc_a["date"], y=fc_a["forecast"],
                                 mode="lines",
                                 line=dict(color="#f59e0b",width=2.5,dash="dot"),
                                 name="ARIMA Forecast"))

    # vertical split line
    if not df.empty:
        last_hist = df[df["ticker"]==ticker_sel]["date"].max()
        x_str = str(last_hist.date())
        fig.add_shape(type="line", x0=x_str, x1=x_str, y0=0, y1=1, yref="paper",
                      line=dict(dash="dash", color="#334155", width=1))
        fig.add_annotation(x=x_str, y=0.98, yref="paper", text="Forecast start",
                           showarrow=False, font=dict(size=9, color="#475569"),
                           xanchor="left", bgcolor="rgba(0,0,0,0)")

    apply_theme(fig, height=460)
    fig.update_layout(
        title=dict(text=f"<b>{ticker_sel}</b> — 90-Day Price Forecast",
                   font=dict(size=14,color="#e2e8f0"), x=0.01),
        xaxis_title="", yaxis_title="Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── side-by-side tables ────────────────────────────────────────────────
    c1, c2 = st.columns(2, gap="medium")
    if not xgb_fc.empty:
        with c1:
            section("XGBoost — Next 30 Days")
            fc_x30 = xgb_fc[xgb_fc["ticker"]==ticker_sel].head(30)[["date","forecast"]].copy()
            fc_x30["date"] = pd.to_datetime(fc_x30["date"]).dt.date
            fc_x30["chg_%"] = fc_x30["forecast"].pct_change()*100
            st.dataframe(fc_x30.style.format({"forecast":"${:.2f}","chg_%":"{:+.2f}%"})
                         .background_gradient(cmap="RdYlGn", subset=["chg_%"]),
                         use_container_width=True, hide_index=True)

    if not arima_fc.empty:
        with c2:
            section("ARIMA — Next 30 Days (with CI)")
            show_cols = ["date","forecast"] + (["lower_90","upper_90"] if "lower_90" in arima_fc.columns else [])
            fc_a30 = arima_fc[arima_fc["ticker"]==ticker_sel].head(30)[show_cols].copy()
            fc_a30["date"] = pd.to_datetime(fc_a30["date"]).dt.date
            fmt = {c:"${:.2f}" for c in fc_a30.columns if c!="date"}
            st.dataframe(fc_a30.style.format(fmt), use_container_width=True, hide_index=True)

    # ── model comparison across tickers ────────────────────────────────────
    section("Forecast Divergence — XGBoost vs ARIMA (Day 90)")
    if not xgb_fc.empty and not arima_fc.empty:
        xgb_last  = xgb_fc.groupby("ticker")["forecast"].last().reset_index().rename(columns={"forecast":"xgb_90d"})
        arima_last = arima_fc.groupby("ticker")["forecast"].last().reset_index().rename(columns={"forecast":"arima_90d"})
        merged_fc  = xgb_last.merge(arima_last, on="ticker")
        merged_fc["divergence_%"] = ((merged_fc["xgb_90d"] - merged_fc["arima_90d"]) /
                                      merged_fc["arima_90d"] * 100).round(2)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=merged_fc["ticker"], y=merged_fc["xgb_90d"],
                               name="XGBoost Day-90", marker_color="#10b981", opacity=0.8))
        fig2.add_trace(go.Bar(x=merged_fc["ticker"], y=merged_fc["arima_90d"],
                               name="ARIMA Day-90",   marker_color="#f59e0b", opacity=0.8))
        apply_theme(fig2, height=300)
        fig2.update_layout(barmode="group", xaxis_title="", yaxis_title="Forecast Price ($)")
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 7 — Fundamentals
# ─────────────────────────────────────────────────────────────────────────────

def page_fundamentals(funds: pd.DataFrame, df: pd.DataFrame) -> None:
    page_header("Fundamentals Screener", "P/E · EPS · beta · market cap · dividend yield · 52-week range")
    if funds.empty:
        st.warning("No fundamentals data. Run pipeline."); return

    # ── filters ────────────────────────────────────────────────────────────
    with st.expander("🔧 Screener Filters", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1: pe_max    = st.slider("Max P/E",         0, 100, 60)
        with fc2: beta_max  = st.slider("Max Beta",     0.0, 3.0, 2.5, 0.1)
        with fc3: mcap_min  = st.slider("Min MCap ($Bn)", 0, 500,   0)
        with fc4: div_min   = st.slider("Min Div Yield %", 0.0, 10.0, 0.0, 0.5)

    if all(c in funds.columns for c in ["pe_ratio","beta","market_cap_bn"]):
        filtered = funds[
            (funds["pe_ratio"].fillna(999) <= pe_max) &
            (funds["beta"].fillna(999) <= beta_max) &
            (funds["market_cap_bn"].fillna(0) >= mcap_min)
        ].copy()
    else:
        filtered = funds.copy()

    if "dividend_yield" in filtered.columns and div_min > 0:
        filtered = filtered[filtered["dividend_yield"].fillna(0) >= div_min/100]

    # KPIs
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1: kpi("Tickers Passing Screen", str(len(filtered)), icon="✅")
    with c2:
        avg_pe = filtered["pe_ratio"].mean() if "pe_ratio" in filtered.columns else 0
        kpi("Avg P/E", f"{avg_pe:.1f}x", delta=None, icon="💹")
    with c3:
        avg_beta = filtered["beta"].mean() if "beta" in filtered.columns else 0
        kpi("Avg Beta", f"{avg_beta:.2f}", delta=None, icon="📐")
    with c4:
        total_mcap = filtered["market_cap_bn"].sum() if "market_cap_bn" in filtered.columns else 0
        kpi("Total MCap ($Bn)", f"${total_mcap:,.0f}", delta=None, icon="🏦")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── bubble chart ───────────────────────────────────────────────────────
    section("Market Cap vs P/E Ratio (bubble = market cap, colour = sector)")
    if all(c in filtered.columns for c in ["market_cap_bn","pe_ratio"]):
        bub = filtered.dropna(subset=["market_cap_bn","pe_ratio"]).copy()
        bub = bub[bub["market_cap_bn"] > 0]
        hover_cols = [c for c in ["ticker","beta","dividend_yield","eps_ttm"] if c in bub.columns]
        fig = px.scatter(bub, x="pe_ratio", y="market_cap_bn",
                         size="market_cap_bn", size_max=40,
                         color="sector", color_discrete_map=SECTOR_PALETTE,
                         text="ticker", hover_data=hover_cols,
                         labels={"pe_ratio":"P/E Ratio","market_cap_bn":"Market Cap ($Bn)"})
        fig.update_traces(textposition="top center", textfont_size=8,
                          marker=dict(line=dict(color="#0a0e1a",width=1)))
        apply_theme(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── 52-week range chart ────────────────────────────────────────────────
    if all(c in filtered.columns for c in ["52w_low","52w_high"]):
        section("52-Week Price Range")
        rng = filtered.dropna(subset=["52w_low","52w_high"]).copy()

        # get latest close price from clean data
        if not df.empty:
            latest = df.sort_values("date").groupby("ticker")["close"].last().reset_index()
            rng = rng.merge(latest, on="ticker", how="left")
        else:
            rng["close"] = np.nan

        rng["pct_of_range"] = ((rng.get("close", rng["52w_high"]) - rng["52w_low"]) /
                                (rng["52w_high"] - rng["52w_low"] + 1e-6) * 100).round(1)
        rng = rng.sort_values("pct_of_range")

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=rng["pct_of_range"], y=rng["ticker"], orientation="h",
            marker=dict(color=rng["pct_of_range"], colorscale="RdYlGn",
                        cmin=0, cmax=100, line_width=0),
            text=[f"{v:.0f}%" for v in rng["pct_of_range"]],
            textposition="outside", textfont=dict(size=9, color="#94a3b8"),
        ))
        fig2.add_vline(x=50, line_dash="dash", line_color="#334155")
        apply_theme(fig2, height=max(300, len(rng)*18))
        fig2.update_layout(xaxis_title="Position in 52-Week Range (%)", yaxis_title="",
                            xaxis_range=[0,120])
        st.plotly_chart(fig2, use_container_width=True)

    # ── data table ─────────────────────────────────────────────────────────
    section("Screener Results")
    show_cols = [c for c in ["ticker","sector","name","market_cap_bn","pe_ratio","forward_pe",
                              "eps_ttm","beta","dividend_yield","52w_high","52w_low"]
                 if c in filtered.columns]
    st.dataframe(
        filtered[show_cols].sort_values("market_cap_bn", ascending=False)
        .style.format({c:"{:.2f}" for c in show_cols if c not in ["ticker","sector","name"]}),
        use_container_width=True, hide_index=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page 8 — Model Report
# ─────────────────────────────────────────────────────────────────────────────

def page_model_report(metrics: dict) -> None:
    page_header("Model Performance Report", "XGBoost · ARIMA · IsolationForest · evaluation metrics")

    # flatten nested metric dicts (e.g. {"xgb": {"rmse": ...}})
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}_{kk}"] = vv
        else:
            flat[k] = v

    METRIC_META = {
        "xgb_cv_directional_accuracy_pct": ("Dir. Accuracy",  "{:.2f}%",   "🎯", "XGBoost"),
        "xgb_cv_mae_basis_points":         ("MAE",            "{:.1f} bp", "📏", "XGBoost"),
        "xgb_cv_folds":                    ("CV Folds",       "{:d}",      "🔁", "XGBoost"),
        "xgb_rows_forecast":               ("Forecast Rows",  "{:,.0f}",   "📊", "XGBoost"),
        "arima_rows_forecast":             ("Forecast Rows",  "{:,.0f}",   "📊", "ARIMA"),
    }

    # ── KPI cards ──────────────────────────────────────────────────────────
    section("Model Metrics")
    kpi_keys = [k for k in flat if k in METRIC_META]
    if kpi_keys:
        cols = st.columns(len(kpi_keys), gap="small")
        for i, k in enumerate(kpi_keys):
            label, fmt, icon, mdl = METRIC_META[k]
            v = flat[k]
            try:
                val = fmt.format(int(v) if fmt == "{:d}" else v)
            except (ValueError, TypeError):
                val = str(v)
            with cols[i]:
                kpi(f"{mdl} {label}", val, delta=None, icon=icon)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── pipeline architecture ──────────────────────────────────────────────
    section("Pipeline Architecture")
    col_a, col_b, col_c, col_d = st.columns(4, gap="medium")
    cards = [
        ("1 · Ingestion", col_a,
         "Yahoo Finance (yfinance)<br>36 tickers · 6 sectors · 5y OHLCV<br>Fundamentals: P/E, beta, MCap, EPS<br>Synthetic GBM fallback if offline"),
        ("2 · Feature Engineering", col_b,
         "73 features per row<br>Lag returns (1–21d) · Rolling stats<br>RSI(14) · MACD · Bollinger · ATR<br>Momentum (1–12m) · CAGR · HHI"),
        ("3 · Models", col_c,
         "XGBoost: n=600, depth=6, lr=0.04<br>ARIMA: auto-AIC per ticker<br>IsolationForest: 300 trees<br>DBSCAN + residual anomaly ensemble"),
        ("4 · Evaluation", col_d,
         "5-fold time-series CV<br>Directional accuracy + MAE (bp)<br>SHAP feature importance<br>VaR · CVaR · Sharpe · Calmar"),
    ]
    for title, col, body in cards:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label" style="color:#60a5fa;">{title}</div>
                <div style='font-size:0.8rem; color:#94a3b8; line-height:1.8; margin-top:0.6rem;'>
                    {body.replace(chr(10), "<br>")}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── feature importance (full-width interactive chart) ──────────────────
    feat_path  = MODEL_DIR / "feature_names.json"
    model_path = MODEL_DIR / "xgb_forecaster.joblib"
    if feat_path.exists() and model_path.exists():
        try:
            xgb_model = joblib.load(model_path)
            feats = json.load(open(feat_path))
            imp   = xgb_model.feature_importances_
            fi_df = pd.DataFrame({"Feature": feats[:len(imp)], "Importance": imp})
            fi_df = fi_df.sort_values("Importance", ascending=False).head(15)
            fi_df["Pct"] = (fi_df["Importance"] / fi_df["Importance"].sum() * 100).round(1)

            section("XGBoost Feature Importance (Gain)")
            fig = go.Figure(go.Bar(
                x=fi_df["Importance"].values[::-1],
                y=fi_df["Feature"].values[::-1],
                orientation="h",
                text=[f"{p:.1f}%" for p in fi_df["Pct"].values[::-1]],
                textposition="outside",
                textfont=dict(size=10, color="#94a3b8"),
                marker=dict(
                    color=fi_df["Importance"].values[::-1],
                    colorscale=[[0,"#1e3a5f"],[0.5,"#2563eb"],[1,"#60a5fa"]],
                    line_width=0,
                ),
                hovertemplate="<b>%{y}</b><br>Gain: %{x:.5f}<extra></extra>",
            ))
            apply_theme(fig, height=440)
            fig.update_layout(
                xaxis_title="Feature Importance (Gain)",
                yaxis_title="",
                margin=dict(l=160, r=80, t=30, b=40),
                xaxis=dict(showgrid=True, gridcolor="#1e293b"),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    # ── static analysis plots ──────────────────────────────────────────────
    shap_path  = MODEL_DIR / "shap_importance.png"
    resid_path = MODEL_DIR / "residual_plots.png"
    anom_path  = MODEL_DIR / "anomaly_timeline.png"

    if shap_path.exists() or resid_path.exists():
        col_p, col_q = st.columns(2, gap="medium")
        with col_p:
            if shap_path.exists():
                section("SHAP Feature Importance")
                st.image(str(shap_path), use_column_width=True)
        with col_q:
            if resid_path.exists():
                section("Return Distribution & Q-Q Plot")
                st.image(str(resid_path), use_column_width=True)

    if anom_path.exists():
        section("Anomaly Detection Timeline")
        st.image(str(anom_path), use_column_width=True)

    # ── model evaluation summary table ────────────────────────────────────
    section("Model Evaluation Summary")
    METRIC_DISPLAY = {
        "xgb_cv_directional_accuracy_pct": ("XGBoost", "Directional Accuracy", "{:.2f}%",   "% of days where predicted return sign matched actual"),
        "xgb_cv_mae_basis_points":         ("XGBoost", "MAE (basis points)",   "{:.1f} bp", "Mean absolute error in return predictions (1 bp = 0.01%)"),
        "xgb_cv_folds":                    ("XGBoost", "CV Folds",             "{:d}",      "Number of time-series cross-validation folds"),
        "xgb_rows_forecast":               ("XGBoost", "Forecast Rows",        "{:,.0f}",   "Total 90-day forecast rows generated"),
        "arima_rows_forecast":             ("ARIMA",   "Forecast Rows",        "{:,.0f}",   "Total 90-day forecast rows generated"),
        "xgb_rmse":                        ("XGBoost", "RMSE",                 "{:.4f}",    "Root mean squared error vs actuals"),
        "xgb_mae":                         ("XGBoost", "MAE",                  "{:.4f}",    "Mean absolute error vs actuals"),
        "xgb_mape_pct":                    ("XGBoost", "MAPE",                 "{:.2f}%",   "Mean absolute percentage error"),
        "arima_rmse":                      ("ARIMA",   "RMSE",                 "{:.4f}",    "Root mean squared error vs actuals"),
        "arima_mae":                       ("ARIMA",   "MAE",                  "{:.4f}",    "Mean absolute error vs actuals"),
        "arima_mape_pct":                  ("ARIMA",   "MAPE",                 "{:.2f}%",   "Mean absolute percentage error"),
    }
    rows = []
    for k, v in flat.items():
        if isinstance(v, dict):
            continue
        mdl, metric, fmt, desc = METRIC_DISPLAY.get(k, ("—", k.replace("_", " ").title(), "{:.4g}", ""))
        try:
            value_str = fmt.format(int(v) if fmt == "{:d}" else v)
        except (ValueError, TypeError):
            value_str = str(v)
        rows.append({"Model": mdl, "Metric": metric, "Value": value_str, "Description": desc})
    if rows:
        st.dataframe(
            pd.DataFrame(rows).style
              .apply(lambda col: [
                  "background-color:#1e3a5f; color:#60a5fa; font-weight:600" if v == "XGBoost"
                  else "background-color:#1a2e1a; color:#4ade80; font-weight:600" if v == "ARIMA"
                  else "" for v in col
              ], subset=["Model"]),
            use_container_width=True, hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_css()

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
    elif page == "📈 Price & Volume":       page_price_volume(feat_df_f if not feat_df_f.empty else df_f)
    elif page == "🔍 EDA & Statistics":     page_eda(feat_df_f)
    elif page == "📐 Risk & Portfolio":     page_risk_portfolio(df_f)
    elif page == "⚠️ Anomaly Radar":        page_anomaly(anom_df_f)
    elif page == "🔮 Forecasting":          page_forecasting(df_f, xgb_fc, arima_fc)
    elif page == "🏦 Fundamentals":         page_fundamentals(funds, df_f)
    elif page == "📋 Model Report":         page_model_report(metrics)


main()
