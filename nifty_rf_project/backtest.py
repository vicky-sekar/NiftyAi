"""Backtest engine for Nifty Random Forest strategy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from config import (
    BROKERAGE_BPS,
    DATA_FILE,
    DEFAULT_THRESHOLD,
    FEATURE_COLUMNS,
    MODEL_PATH,
    STOP_LOSS_PCT,
    TARGET_PCT,
    TRIGGER_PCT,
)
from train_model import build_features


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: dict


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return float(dd.min())


def run_backtest(df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD, starting_capital: float = 1_000_000) -> BacktestResult:
    model = joblib.load(MODEL_PATH)
    data = build_features(df)

    probs = model.predict_proba(data[FEATURE_COLUMNS])[:, 1]
    data["ml_prob"] = probs

    data["session"] = data["timestamp"].dt.date
    prev_close = data.groupby("session")["close"].transform("last").shift(1)
    data["prev_close"] = prev_close

    trades = []
    capital = starting_capital
    equity_rows = []

    in_position = False
    entry_price = 0.0
    entry_time = None

    for row in data.itertuples(index=False):
        if pd.isna(row.prev_close):
            equity_rows.append((row.timestamp, capital))
            continue

        trigger_hit = (row.close / row.prev_close - 1) <= TRIGGER_PCT
        signal = trigger_hit and row.close > row.vwap and row.ml_prob >= threshold

        if not in_position and signal:
            in_position = True
            entry_price = row.close
            entry_time = row.timestamp

        if in_position:
            ret = row.close / entry_price - 1
            exit_reason = None
            if ret >= TARGET_PCT:
                exit_reason = "target"
            elif ret <= -STOP_LOSS_PCT:
                exit_reason = "stop_loss"
            elif row.timestamp.time().strftime("%H:%M") >= "15:25":
                exit_reason = "eod"

            if exit_reason:
                gross_pnl = capital * ret
                cost = capital * (BROKERAGE_BPS / 10000)
                net_pnl = gross_pnl - cost
                capital += net_pnl
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": row.timestamp,
                        "entry": entry_price,
                        "exit": row.close,
                        "return": ret,
                        "pnl": net_pnl,
                        "reason": exit_reason,
                        "ml_prob": row.ml_prob,
                    }
                )
                in_position = False

        equity_rows.append((row.timestamp, capital))

    trades_df = pd.DataFrame(trades)
    equity_curve = pd.DataFrame(equity_rows, columns=["timestamp", "equity"])
    if equity_curve.empty:
        equity_curve = pd.DataFrame({"timestamp": [], "equity": []})

    summary = {
        "total_trades": int(len(trades_df)),
        "win_rate": float((trades_df["pnl"] > 0).mean()) if not trades_df.empty else 0.0,
        "total_return": float((capital / starting_capital) - 1),
        "max_drawdown": max_drawdown(equity_curve["equity"]) if not equity_curve.empty else 0.0,
        "avg_trade_return": float(trades_df["return"].mean()) if not trades_df.empty else 0.0,
        "final_equity": float(capital),
    }

    return BacktestResult(trades=trades_df, equity_curve=equity_curve, summary=summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Nifty RF strategy")
    parser.add_argument("--data", type=str, default=str(DATA_FILE))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--capital", type=float, default=1_000_000)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    result = run_backtest(df, threshold=args.threshold, starting_capital=args.capital)

    print("Backtest Summary")
    for k, v in result.summary.items():
        print(f"{k}: {v}")

    if not result.trades.empty:
        print("\nRecent Trades")
        print(result.trades.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
