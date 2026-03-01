"""Model training script for Nifty 5-minute Random Forest strategy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from config import (
    DATA_FILE,
    FEATURE_COLUMNS,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TRAIN_SPLIT,
    ensure_dirs,
)


@dataclass
class TrainArtifacts:
    model: RandomForestClassifier
    report: str
    roc_auc: float


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    out["session"] = out["timestamp"].dt.date
    out["pv"] = out["close"] * out["volume"]
    out["cum_pv"] = out.groupby("session")["pv"].cumsum()
    out["cum_vol"] = out.groupby("session")["volume"].cumsum().replace(0, np.nan)
    out["vwap"] = out["cum_pv"] / out["cum_vol"]

    out["return_1"] = out["close"].pct_change()
    out["return_3"] = out["close"].pct_change(3)
    out["return_6"] = out["close"].pct_change(6)

    rolling_vol = out["volume"].rolling(20)
    out["volume_z"] = (out["volume"] - rolling_vol.mean()) / rolling_vol.std()
    out["dist_to_vwap"] = (out["close"] - out["vwap"]) / out["vwap"]
    out["rsi_14"] = compute_rsi(out["close"], 14)

    forward_return = out["close"].shift(-3) / out["close"] - 1
    out[TARGET_COLUMN] = (forward_return > 0).astype(int)

    return out.dropna().reset_index(drop=True)


def train(df: pd.DataFrame) -> TrainArtifacts:
    features = build_features(df)
    split_idx = int(len(features) * TRAIN_SPLIT)

    train_df = features.iloc[:split_idx]
    test_df = features.iloc[split_idx:]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    return TrainArtifacts(model=model, report=report, roc_auc=roc_auc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest model on Nifty 5-minute data")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_FILE),
        help="Path to 5-minute Nifty OHLCV CSV with timestamp, open, high, low, close, volume",
    )
    args = parser.parse_args()

    ensure_dirs()
    data = pd.read_csv(args.data)
    artifacts = train(data)

    joblib.dump(artifacts.model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"ROC-AUC: {artifacts.roc_auc:.4f}")
    print("Classification report:")
    print(artifacts.report)


if __name__ == "__main__":
    main()
