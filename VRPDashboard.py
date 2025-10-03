# app.py
# -------------------------------------------------------------------
# Streamlit Dashboard: S&P 500 Volatility Risk Premium (VRP)
# Robust to missing OHLC (Yahoo hiccups), with ^GSPC/SPY toggle + fallback
# Updated with enhancements: VRP Percentile & Signals, Multi-Asset Comparison,
# Strategy Backtest, Macro Correlations, and Actionable Trader Insights
# -------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
import plotly.express as px

st.set_page_config(page_title="S&P 500 VRP Dashboard", layout="wide")

# --------------------------- Utilities ---------------------------
REQUIRED_OHLC = ["Open", "High", "Low", "Close"]

def ensure_datetime_single_index(df: pd.DataFrame) -> pd.DataFrame:
    """Force a single DatetimeIndex named 'Date' on a DataFrame."""
    out = df.copy()
    if out.empty:
        return out
    if isinstance(out.index, pd.MultiIndex):
        lvl0 = pd.to_datetime(out.index.get_level_values(0), errors="coerce")
        out.index = lvl0
    else:
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "Date"
    return out.sort_index()

def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten any MultiIndex columns from yfinance into a single row of names."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        new_cols = []
        for c in out.columns:
            if isinstance(c, tuple):
                # Prefer the OHLC field name if present (first level usually)
                new_cols.append(c[0] if isinstance(c[0], str) else "_".join([str(x) for x in c if x]))
            else:
                new_cols.append(c)
        out.columns = new_cols
    return out

def ensure_single_close(df: pd.DataFrame, close_label: str = "Close", new_name: str | None = None) -> pd.DataFrame:
    """
    Keep only one 'Close'-like column and optionally rename it.
    Falls back to 'Adj Close' if 'Close' absent; otherwise first numeric column.
    """
    out = flatten_yf_columns(df)
    if out.empty:
        return out
    col = None
    if close_label in out.columns:
        col = close_label
    elif "Adj Close" in out.columns:
        col = "Adj Close"
    else:
        numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        if numeric_cols:
            col = numeric_cols[0]
        else:
            return pd.DataFrame(index=out.index)  # empty -> caller handles
    out = out[[col]].copy()
    if new_name:
        out = out.rename(columns={col: new_name})
    return out

def has_ohlc(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in REQUIRED_OHLC)

def yang_zhang_rv(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Annualized Yang–Zhang realized volatility (%) using OHLC."""
    missing = [c for c in REQUIRED_OHLC if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    k = 0.34
    o = np.log(df["Open"])
    h = np.log(df["High"])
    l = np.log(df["Low"])
    c = np.log(df["Close"])

    r_oc = o - c.shift(1)           # overnight
    r_co = c - o                    # intraday

    sigma2_o  = r_oc.rolling(window).var(ddof=0)
    sigma2_c  = r_co.rolling(window).var(ddof=0)
    sigma2_rs = ((h - o) * (h - c) + (l - o) * (l - c)).rolling(window).mean()

    yz_var = sigma2_o + k * sigma2_c + (1 - k) * sigma2_rs
    rv_pct = np.sqrt(yz_var) * np.sqrt(252) * 100.0
    rv_pct.name = "RV_YZ_AnnPct"
    return rv_pct

@st.cache_data(show_spinner=False)
def fetch_ohlc(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLC for a ticker. Returns what it can; caller checks has_ohlc(...)."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        return df
    df = flatten_yf_columns(df)
    # Keep OHLC if present (order preserved)
    keep = [c for c in REQUIRED_OHLC if c in df.columns]
    df = df[keep + [c for c in df.columns if c not in keep]]
    return ensure_datetime_single_index(df)

@st.cache_data(show_spinner=False)
def load_data(primary: str, start: str, end: str):
    """
    Try primary (^GSPC or SPY) for OHLC; if missing, fallback to the other.
    Always load VIX. Return (prices_df, vix_df, used_ticker, fell_back_bool).
    """
    # Try primary
    prices = fetch_ohlc(primary, start, end)
    used = primary
    fell_back = False

    # Fallback logic if OHLC missing or empty
    if prices.empty or not has_ohlc(prices):
        alt = "SPY" if primary == "^GSPC" else "^GSPC"
        alt_prices = fetch_ohlc(alt, start, end)
        if not alt_prices.empty and has_ohlc(alt_prices):
            prices = alt_prices
            used = alt
            fell_back = True

    # VIX (implied volatility)
    vix_raw = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False)
    vix_raw = ensure_datetime_single_index(flatten_yf_columns(vix_raw))
    vix = ensure_single_close(vix_raw, close_label="Close", new_name="VIX")
    vix = ensure_datetime_single_index(vix)

    return prices, vix, used, fell_back

def compute_vrp_for_ticker(ticker: str, vix_proxy: str, start: str, end: str, window: int):
    prices = fetch_ohlc(ticker, start, end)
    prices = ensure_datetime_single_index(prices)
    if prices.empty or not has_ohlc(prices):
        return pd.DataFrame()  # Skip if no data

    vix_raw = yf.download(vix_proxy, start=start, end=end, auto_adjust=False, progress=False)
    if vix_raw is None or len(vix_raw) == 0:
        return pd.DataFrame()
    vix_raw = ensure_datetime_single_index(flatten_yf_columns(vix_raw))
    vix_df = ensure_single_close(vix_raw, close_label="Close", new_name="IV_Proxy")
    vix_df = ensure_datetime_single_index(vix_df)
    if vix_df.empty:
        return pd.DataFrame()

    # Join on a single DatetimeIndex
    raw = prices.join(vix_df, how="inner")
    if raw.empty:
        return pd.DataFrame()

    rv = yang_zhang_rv(raw[REQUIRED_OHLC], window)
    aligned = pd.concat([raw["IV_Proxy"].rename("IV_AnnPct"), rv], axis=1).dropna()
    aligned["VRP_PctPts"] = aligned["IV_AnnPct"] - aligned["RV_YZ_AnnPct"]
    return aligned

def format_pct(x: float) -> str:
    return f"{x:.2f}%"

# --------------------------- Sidebar ---------------------------
st.sidebar.header("Settings")
default_start = date(2004, 1, 1)  # post VIX methodology change
default_end = date.today()

ticker_choice = st.sidebar.selectbox(
    "Price source",
    options=["^GSPC (Index)", "SPY (ETF)"],
    index=0,
    help="If ^GSPC is unavailable for your date range, the app will auto-fallback to SPY."
)
primary_ticker = "^GSPC" if ticker_choice.startswith("^GSPC") else "SPY"

date_range = st.sidebar.date_input(
    "Historical range",
    value=(default_start, default_end),
    min_value=date(1990, 1, 1),
    max_value=default_end
)
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = default_start, date_range

window = st.sidebar.slider("Realized Vol window (trading days)", min_value=5, max_value=63, value=21, step=1)
agg = st.sidebar.selectbox("Resample (optional)", ["None (daily)", "Month-end"], index=0)
st.sidebar.caption("Tip: Month-end resampling smooths the series and aligns to ~30-day VIX horizon.")

threshold = st.sidebar.slider("VRP Signal Threshold (pp)", 0.0, 10.0, 3.0, 0.5)

# --------------------------- Data & RV ---------------------------
st.title("S&P 500 Volatility Risk Premium (VRP)")

with st.spinner("Loading data…"):
    prices, vix, used_ticker, fell_back = load_data(primary_ticker, str(start_date), str(end_date))

if prices.empty:
    st.error("Could not download S&P 500 prices with OHLC for the selected range (both ^GSPC and SPY failed). "
             "Please widen the date range or try again later.")
    st.stop()

if fell_back:
    st.info(f"Primary ticker {primary_ticker} lacked OHLC for part/all of the range. "
            f"Fell back to **{used_ticker}** for prices.")

# Join prices & VIX on common dates
raw = prices.join(vix, how="inner")
if raw.empty or "VIX" not in raw.columns:
    st.error("VIX could not be aligned with price data for the selected range. Try widening the dates.")
    st.stop()

# Compute Yang–Zhang RV (guaranteed OHLC)
try:
    rv_series = yang_zhang_rv(raw[REQUIRED_OHLC], window=window)
except Exception as e:
    st.exception(e)
    st.stop()

rv_df = pd.DataFrame({"RV_YZ_AnnPct": rv_series.reindex(raw.index)})
df = pd.concat([raw, rv_df], axis=1)
df = df.dropna(subset=["RV_YZ_AnnPct"])

# Optional resampling
if agg == "Month-end":
    iv = df["VIX"].resample("M").last()
    rv_series = df["RV_YZ_AnnPct"].resample("M").last()
else:
    iv = df["VIX"]
    rv_series = df["RV_YZ_AnnPct"]

aligned = pd.concat(
    [iv.rename("IV_VIX_AnnPct"), rv_series.rename("RV_YZ_AnnPct")],
    axis=1
).dropna()

aligned["VRP_PctPts"] = aligned["IV_VIX_AnnPct"] - aligned["RV_YZ_AnnPct"]

# --------------------------- Enhancement 1: VRP Percentile & Signals ---------------------------
aligned["VRP_Percentile"] = aligned["VRP_PctPts"].rank(pct=True) * 100

# Historical conditional returns (forward-ish monthly using rolling sum of daily logs if daily freq)
df["LogRet"] = np.log(df["Close"] / df["Close"].shift(1))  # Daily log returns
if agg == "Month-end":
    monthly_rets = df["LogRet"].resample("M").sum() * 100  # % monthly
else:
    monthly_rets = df["LogRet"].rolling(21).sum() * 100  # Approx monthly

aligned = aligned.join(monthly_rets.rename("Fwd_Monthly_Ret"), how="inner")

high_vrp = aligned[aligned["VRP_Percentile"] > 75]
low_vrp = aligned[aligned["VRP_Percentile"] < 25]
high_mean_ret = high_vrp["Fwd_Monthly_Ret"].mean() if not high_vrp.empty else np.nan
low_mean_ret = low_vrp["Fwd_Monthly_Ret"].mean() if not low_vrp.empty else np.nan
hit_rate_high = (high_vrp["Fwd_Monthly_Ret"] > 0).mean() * 100 if not high_vrp.empty else np.nan

# --------------------------- Enhancement 2: Multi-Asset VRP Comparison ---------------------------
assets = {
    "S&P 500": {"ticker": used_ticker, "vix": "^VIX"},
    "Nasdaq": {"ticker": "QQQ", "vix": "^VXN"},  # Nasdaq VIX
    "Bonds": {"ticker": "TLT", "vix": "^MOVE"}   # Bond vol index (scale differs; used for relative comp)
}
vrp_dict = {}
for name, info in assets.items():
    try:
        vrp_dict[name] = compute_vrp_for_ticker(info["ticker"], info["vix"], str(start_date), str(end_date), window)
    except Exception:
        vrp_dict[name] = pd.DataFrame()

# --------------------------- Enhancement 3: Simple VRP-Based Strategy Backtest ---------------------------
aligned["Signal"] = np.where(aligned["VRP_PctPts"] > threshold, 1, 0)  # 1 = exposure (e.g., short vol / long equity)
df["Strat_Ret"] = df["LogRet"] * aligned["Signal"].reindex(df.index).ffill().fillna(0)  # Align signals
strat_cumret = (np.exp(df["Strat_Ret"].cumsum()) - 1) * 100  # Cumulative % return
bh_cumret = (np.exp(df["LogRet"].cumsum()) - 1) * 100

sharpe_strat = (df["Strat_Ret"].mean() / df["Strat_Ret"].std()) * np.sqrt(252) if df["Strat_Ret"].std() not in (0, np.nan) else 0
# Max drawdown (percentage terms)
if not strat_cumret.empty:
    running_max = (1 + strat_cumret/100).cummax()
    dd_series = ((1 + strat_cumret/100) / running_max - 1) * 100
    max_dd_strat = dd_series.min()
else:
    max_dd_strat = 0

# --------------------------- Enhancement 4: VRP Correlation with Economic Indicators ---------------------------
try:
    yield_10y_raw = yf.download("^TNX", str(start_date), str(end_date), auto_adjust=False, progress=False)
    yield_2y_raw = yf.download("^IRX", str(start_date), str(end_date), auto_adjust=False, progress=False)
    yield_10y = ensure_single_close(ensure_datetime_single_index(flatten_yf_columns(yield_10y_raw)), new_name="10Y_Yield")
    yield_2y = ensure_single_close(ensure_datetime_single_index(flatten_yf_columns(yield_2y_raw)), new_name="2Y_Yield")
    yields = yield_10y.join(yield_2y, how="inner").dropna()
    yields["Curve_Spread"] = yields["10Y_Yield"] - yields["2Y_Yield"]
    aligned = aligned.join(yields["Curve_Spread"], how="inner")
    corr_vrp_curve = aligned["VRP_PctPts"].corr(aligned["Curve_Spread"]) if "Curve_Spread" in aligned else np.nan
except Exception:
    corr_vrp_curve = np.nan

# --------------------------- KPIs ---------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Latest VIX (IV)", format_pct(aligned["IV_VIX_AnnPct"].iloc[-1]))
with col2:
    st.metric("Latest Realized Vol (Yang–Zhang)", format_pct(aligned["RV_YZ_AnnPct"].iloc[-1]))
with col3:
    st.metric("Latest VRP (IV − RV)", f"{aligned['VRP_PctPts'].iloc[-1]:.2f} pp")

# --------------------------- Plot 1: IV vs RV ---------------------------
st.subheader(f"Implied vs Realized Volatility — Prices from {used_ticker}")
df_plot1 = aligned.reset_index().rename(columns={aligned.index.name or "index": "Date"})
fig1 = px.line(
    df_plot1,
    x="Date",
    y=["IV_VIX_AnnPct", "RV_YZ_AnnPct"],
    labels={"value": "Annualized Volatility (%)", "variable": "Series"},
)
fig1.update_layout(legend_title_text="", height=420, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig1, use_container_width=True)

# --------------------------- Plot 2: VRP ---------------------------
st.subheader("Volatility Risk Premium (VRP) = VIX − Realized Vol (Yang–Zhang)")
df_plot2 = aligned.reset_index().rename(columns={aligned.index.name or "index": "Date"})
fig2 = px.line(
    df_plot2,
    x="Date",
    y="VRP_PctPts",
    labels={"VRP_PctPts": "Percentage points"},
)
fig2.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig2, use_container_width=True)

# --------------------------- Enhancement 1 Display: VRP Percentile & Signals ---------------------------
st.subheader("VRP Percentile & Signals")
col4, col5 = st.columns(2)
with col4:
    st.metric("Current VRP Percentile", f"{aligned['VRP_Percentile'].iloc[-1]:.1f}%")
with col5:
    perc = aligned["VRP_Percentile"].iloc[-1]
    signal = "High (Sell Volatility)" if perc > 75 else \
             "Low (Buy Volatility)" if perc < 25 else "Neutral"
    st.metric("Signal", signal)

st.write(
    "Actionable Insight: In high VRP regimes (>75th percentile), sell volatility (e.g., short VIX futures or sell straddles) "
    f"as implied vol is overpriced relative to realized—historical avg forward monthly S&P return: "
    f"{(high_mean_ret if pd.notna(high_mean_ret) else 0):.2f}% "
    f"(hit rate: {(hit_rate_high if pd.notna(hit_rate_high) else 0):.0f}%). "
    "In low VRP (<25th percentile), buy volatility (e.g., long VIX calls or protective puts) to hedge against potential vol spikes—"
    f"avg forward return: {(low_mean_ret if pd.notna(low_mean_ret) else 0):.2f}%."
)

fig_hist = px.histogram(aligned.reset_index(), x="VRP_PctPts", nbins=50, title="VRP Distribution")
fig_hist.add_vline(x=aligned["VRP_PctPts"].iloc[-1], line_dash="dash", annotation_text="Current")
st.plotly_chart(fig_hist, use_container_width=True)

# --------------------------- Enhancement 2 Display: Cross-Asset VRP Comparison ---------------------------
st.subheader("Cross-Asset VRP Comparison")
latest_vrps = {name: df2["VRP_PctPts"].iloc[-1] for name, df2 in vrp_dict.items() if isinstance(df2, pd.DataFrame) and not df2.empty}

if latest_vrps:
    vrp_df = pd.DataFrame.from_dict(latest_vrps, orient="index", columns=["Latest VRP (pp)"])
    fig_heat = px.imshow(vrp_df.T, color_continuous_scale="RdYlGn", title="Relative VRP Heatmap", aspect="auto")
    st.plotly_chart(fig_heat, use_container_width=True)
    st.write(
        "Actionable Insight: Compare VRPs across assets—sell volatility in the highest VRP asset "
        "(e.g., if Nasdaq VRP > S&P by 2pp, rotate to short VXN for better premium capture); "
        "buy vol in low VRP assets for diversification."
    )
else:
    st.info("Could not compute cross-asset VRPs (one or more proxies returned no data).")

# --------------------------- Enhancement 3 Display: VRP Strategy Backtest ---------------------------
st.subheader("VRP Strategy Backtest")
backtest_df = pd.DataFrame({"Strategy": strat_cumret, "Buy-Hold": bh_cumret}).reset_index().rename(columns={"index": "Date"})
fig_bt = px.line(backtest_df, x="Date", y=["Strategy", "Buy-Hold"], title="Cumulative Returns (%)")
st.plotly_chart(fig_bt, use_container_width=True)
col6, col7 = st.columns(2)
with col6:
    st.metric("Strategy Sharpe", f"{sharpe_strat:.2f}")
with col7:
    st.metric("Max Drawdown", f"{max_dd_strat:.1f}%")
st.write(
    f"Actionable Insight: At {threshold}pp VRP threshold, the strategy sells volatility (goes long equity/short vol) "
    "when VRP is high, capturing premium with reduced exposure during low VRP periods. "
    "Traders can implement via VIX futures/ETFs: Sell when signal=1 for expected edge; hold cash or buy vol protection otherwise."
)

# --------------------------- Enhancement 4 Display: VRP Macro Correlations ---------------------------
st.subheader("VRP Macro Correlations")
if "Curve_Spread" in aligned:
    fig_scatter = px.scatter(aligned.reset_index(), x="Curve_Spread", y="VRP_PctPts",
                             trendline="ols", title="VRP vs. Yield Curve Spread")
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.write(
        f"Correlation: {corr_vrp_curve:.2f}. Actionable Insight: When yield curve inverts (negative spread) and VRP spikes (>5pp), "
        "buy volatility aggressively as it signals recessionary vol risks; in steep curves with high VRP, sell vol for carry "
        "but monitor for flattening."
    )
else:
    st.info("Yield curve data unavailable for the selected range.")

# --------------------------- Data & Download ---------------------------
with st.expander("Show data"):
    st.dataframe(
        aligned.style.format({"IV_VIX_AnnPct": "{:.2f}", "RV_YZ_AnnPct": "{:.2f}", "VRP_PctPts": "{:.2f}"}),
        use_container_width=True
    )

csv = aligned.to_csv(index=True).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="spx_vrp_series.csv", mime="text/csv")

st.caption(
    "Notes: VIX is a ~30-day implied volatility proxy. Realized vol uses the Yang–Zhang estimator with a rolling window "
    f"of {window} trading days and is annualized by √252. VRP is in percentage points."
)

# --------------------------- Debug expander ---------------------------
with st.expander("Debug info"):
    st.write("Used ticker for OHLC:", used_ticker, " | Fell back:", fell_back)
    st.write("raw columns:", list(raw.columns))
    st.write("First rows:", raw.head())
