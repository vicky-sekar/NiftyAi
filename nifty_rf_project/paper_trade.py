"""Paper trading mode for the Nifty RF strategy (no real execution)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import joblib
import pandas as pd

from config import DATA_FILE, DEFAULT_THRESHOLD, FEATURE_COLUMNS, MODEL_PATH, TRIGGER_PCT
from train_model import build_features


@dataclass
class PaperSignal:
    timestamp: pd.Timestamp
    price: float
    vwap: float
    probability: float
    trigger_hit: bool
    action: str


def generate_paper_signal(df: pd.DataFrame, threshold: float) -> PaperSignal:
    model = joblib.load(MODEL_PATH)
    data = build_features(df)

    if data.empty:
        raise ValueError("Insufficient data to generate paper signal.")

    data["session"] = data["timestamp"].dt.date
    data["prev_close"] = data.groupby("session")["close"].transform("last").shift(1)

    latest = data.iloc[-1].copy()
    prob = float(model.predict_proba(latest[FEATURE_COLUMNS].to_frame().T)[0, 1])

    trigger_hit = False
    if pd.notna(latest["prev_close"]):
        trigger_hit = (latest["close"] / latest["prev_close"] - 1) <= TRIGGER_PCT

    conditions_met = trigger_hit and latest["close"] > latest["vwap"] and prob >= threshold
    action = "PAPER_BUY" if conditions_met else "NO_TRADE"

    return PaperSignal(
        timestamp=latest["timestamp"],
        price=float(latest["close"]),
        vwap=float(latest["vwap"]),
        probability=prob,
        trigger_hit=trigger_hit,
        action=action,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Nifty RF in paper trading mode")
    parser.add_argument("--data", type=str, default=str(DATA_FILE))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    signal = generate_paper_signal(df, threshold=args.threshold)

    print("Paper Trade Signal")
    print(f"timestamp: {signal.timestamp}")
    print(f"price: {signal.price:.2f}")
    print(f"vwap: {signal.vwap:.2f}")
    print(f"probability: {signal.probability:.3f}")
    print(f"trigger_hit: {signal.trigger_hit}")
    print(f"action: {signal.action}")


if __name__ == "__main__":
    main()
