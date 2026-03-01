"""Streamlit dashboard for Nifty RF backtest and risk analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from backtest import run_backtest
from config import DATA_FILE, DEFAULT_THRESHOLD


def rolling_sharpe(returns: pd.Series, window: int = 20) -> pd.Series:
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std().replace(0, np.nan)
    return (mean / std) * np.sqrt(252)


def risk_metrics(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    if rets.empty:
        return {"sharpe": 0.0, "sortino": 0.0, "cagr": 0.0, "volatility": 0.0}

    downside = rets[rets < 0]
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    sortino = rets.mean() / downside.std() * np.sqrt(252) if downside.std() > 0 else 0.0
    periods = len(rets)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / periods) - 1 if periods > 0 else 0.0
    vol = rets.std() * np.sqrt(252)
    return {"sharpe": float(sharpe), "sortino": float(sortino), "cagr": float(cagr), "volatility": float(vol)}


st.set_page_config(page_title="Nifty RF Dashboard", layout="wide")
st.title("Nifty Random Forest Strategy Dashboard")

with st.sidebar:
    st.header("Controls")
    threshold = st.slider("ML Probability Threshold", 0.50, 0.90, float(DEFAULT_THRESHOLD), 0.01)
    lookback_days = st.selectbox("Backtest Window (days)", [30, 45, 60], index=1)
    capital = st.number_input("Starting Capital", min_value=100000, value=1_000_000, step=50000)

raw = pd.read_csv(DATA_FILE)
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
max_ts = raw["timestamp"].max()
cutoff = max_ts - pd.Timedelta(days=lookback_days)
subset = raw.loc[raw["timestamp"] >= cutoff].copy()

result = run_backtest(subset, threshold=threshold, starting_capital=float(capital))
equity_df = result.equity_curve.copy()

if equity_df.empty:
    st.warning("No equity data generated. Check your input data/model.")
    st.stop()

equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
equity_df["drawdown"] = equity_df["equity"] / equity_df["equity"].cummax() - 1
returns = equity_df["equity"].pct_change().fillna(0)
equity_df["rolling_perf"] = returns.rolling(24).sum()
equity_df["rolling_sharpe"] = rolling_sharpe(returns, window=24)

m1, m2, m3, m4 = st.columns(4)
metrics = risk_metrics(equity_df["equity"])
m1.metric("Total Return", f"{result.summary['total_return'] * 100:.2f}%")
m2.metric("Max Drawdown", f"{result.summary['max_drawdown'] * 100:.2f}%")
m3.metric("Sharpe", f"{metrics['sharpe']:.2f}")
m4.metric("Sortino", f"{metrics['sortino']:.2f}")

st.subheader("Equity Curve")
st.line_chart(equity_df.set_index("timestamp")["equity"])

st.subheader("Drawdown Curve")
st.area_chart(equity_df.set_index("timestamp")["drawdown"])

st.subheader("Rolling Performance")
st.line_chart(equity_df.set_index("timestamp")[["rolling_perf", "rolling_sharpe"]])

st.subheader("Trades")
if result.trades.empty:
    st.info("No trades for current configuration.")
else:
    st.dataframe(result.trades.tail(100), use_container_width=True)
