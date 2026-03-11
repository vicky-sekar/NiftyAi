"""Machine-learning confidence engine with self-learning workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


FEATURE_COLUMNS = [
    "trend_strength_score",
    "displacement_strength",
    "liquidity_quality",
    "atr_volatility_ratio",
    "session_type",
    "spread_value",
    "distance_to_OB",
    "distance_to_FVG",
    "premium_discount_flag",
]


@dataclass
class ModelMeta:
    version: str
    path: Path


class AIEngine:
    def __init__(self, model_dir: Path, threshold: float = 0.65) -> None:
        self.model_dir = model_dir
        self.threshold = threshold
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            class_weight="balanced_subsample",
        )
        self.model_meta = ModelMeta(version="v1", path=self.model_dir / "smc_rf_v1.joblib")
        self._is_fitted = False

    def load_latest(self) -> None:
        models = sorted(self.model_dir.glob("smc_rf_v*.joblib"))
        if not models:
            return
        latest = models[-1]
        self.model = joblib.load(latest)
        self.model_meta = ModelMeta(version=latest.stem.replace("smc_rf_", ""), path=latest)
        self._is_fitted = True

    def predict_probability(self, features: Dict[str, float]) -> float:
        if not self._is_fitted:
            return 0.5
        x = np.array([[features[col] for col in FEATURE_COLUMNS]])
        return float(self.model.predict_proba(x)[0][1])

    def should_trade(self, probability: float) -> bool:
        return probability > self.threshold

    def fit(self, train_df: pd.DataFrame) -> None:
        if train_df.empty:
            return
        x = train_df[FEATURE_COLUMNS]
        y = train_df["trade_result"].astype(int)
        if y.nunique() < 2:
            return
        self.model.fit(x, y)
        self._is_fitted = True

    def evaluate(self, validation_df: pd.DataFrame) -> Dict[str, float]:
        if not self._is_fitted or validation_df.empty:
            return {"accuracy": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "drawdown": 1.0}

        x_val = validation_df[FEATURE_COLUMNS]
        y_val = validation_df["trade_result"].astype(int)
        pred = self.model.predict(x_val)
        acc = float(accuracy_score(y_val, pred))

        rr = validation_df["rr_achieved"].astype(float)
        gains = rr[rr > 0].sum()
        losses = abs(rr[rr < 0].sum()) + 1e-9
        profit_factor = float(gains / losses)

        equity = rr.cumsum()
        drawdown = float((equity.cummax() - equity).max()) if not equity.empty else 1.0

        return {
            "accuracy": acc,
            "win_rate": float((validation_df["trade_result"] == 1).mean()),
            "profit_factor": profit_factor,
            "drawdown": drawdown,
        }

    def retrain_if_needed(self, all_trades_df: pd.DataFrame, current_metrics: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        if len(all_trades_df) < 200:
            return False, current_metrics

        recent = all_trades_df.tail(500).copy()
        older = all_trades_df.iloc[:-len(recent)] if len(all_trades_df) > len(recent) else pd.DataFrame()
        split = max(1, int(len(recent) * 0.8))
        train_df, val_df = recent.iloc[:split], recent.iloc[split:]

        challenger = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            random_state=44,
            class_weight="balanced",
        )
        x_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["trade_result"].astype(int)
        if y_train.nunique() < 2:
            return False, current_metrics

        challenger.fit(x_train, y_train)

        self.model = challenger
        self._is_fitted = True
        val_metrics = self.evaluate(val_df if not val_df.empty else older)

        improved_pf = val_metrics["profit_factor"] > current_metrics.get("profit_factor", 0)
        lower_dd = val_metrics["drawdown"] < current_metrics.get("drawdown", 999)
        stable_wr = val_metrics["win_rate"] >= current_metrics.get("win_rate", 0)

        if improved_pf and lower_dd and stable_wr:
            next_version_int = int(self.model_meta.version.replace("v", "")) + 1
            self.model_meta = ModelMeta(
                version=f"v{next_version_int}",
                path=self.model_dir / f"smc_rf_v{next_version_int}.joblib",
            )
            joblib.dump(self.model, self.model_meta.path)
            return True, val_metrics

        self.load_latest()
        return False, current_metrics
