"""SMC (Smart Money Concepts) signal engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class StructurePoint:
    index: int
    price: float
    kind: str


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


class SMCEngine:
    def __init__(self, fractal_window: int = 2, eq_tolerance: float = 0.00015) -> None:
        self.fractal_window = fractal_window
        self.eq_tolerance = eq_tolerance
        self.active_fvg: Dict[str, List[Dict[str, float]]] = {}

    def detect_swings(self, df: pd.DataFrame) -> Tuple[List[StructurePoint], List[StructurePoint]]:
        highs: List[StructurePoint] = []
        lows: List[StructurePoint] = []
        w = self.fractal_window

        for i in range(w, len(df) - w):
            h = df.iloc[i]["high"]
            l = df.iloc[i]["low"]
            if h == df.iloc[i - w : i + w + 1]["high"].max():
                highs.append(StructurePoint(i, float(h), "high"))
            if l == df.iloc[i - w : i + w + 1]["low"].min():
                lows.append(StructurePoint(i, float(l), "low"))
        return highs, lows

    def trend_from_swings(self, highs: List[StructurePoint], lows: List[StructurePoint]) -> str:
        if len(highs) < 2 or len(lows) < 2:
            return "neutral"
        bullish = highs[-1].price > highs[-2].price and lows[-1].price > lows[-2].price
        bearish = highs[-1].price < highs[-2].price and lows[-1].price < lows[-2].price
        if bullish:
            return "bullish"
        if bearish:
            return "bearish"
        return "neutral"

    def detect_bos_choch(self, df: pd.DataFrame, swings: List[StructurePoint], side: str) -> Dict[str, Optional[bool]]:
        atr14 = atr(df, 14)
        if len(swings) < 1:
            return {"bos": False, "choch": False}

        last = swings[-1]
        last_close = float(df.iloc[-1]["close"])
        threshold = 0.3 * float(atr14.iloc[-1]) if not np.isnan(atr14.iloc[-1]) else 0.0

        if side == "bullish":
            bos = last_close > last.price + threshold
            choch = last_close < last.price - threshold
        else:
            bos = last_close < last.price - threshold
            choch = last_close > last.price + threshold

        return {"bos": bos, "choch": choch}

    def find_equal_levels(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        highs, lows = self.detect_swings(df)
        eq_highs: List[float] = []
        eq_lows: List[float] = []

        for i in range(1, len(highs)):
            if abs(highs[i].price - highs[i - 1].price) <= self.eq_tolerance:
                eq_highs.append((highs[i].price + highs[i - 1].price) / 2)

        for i in range(1, len(lows)):
            if abs(lows[i].price - lows[i - 1].price) <= self.eq_tolerance:
                eq_lows.append((lows[i].price + lows[i - 1].price) / 2)

        return {"equal_highs": eq_highs, "equal_lows": eq_lows}

    def previous_day_levels(self, df: pd.DataFrame) -> Dict[str, Optional[float]]:
        grouped = df.set_index("time").groupby(pd.Grouper(freq="1D"))
        days = [g for _, g in grouped if len(g) > 0]
        if len(days) < 2:
            return {"prev_day_high": None, "prev_day_low": None}
        prev = days[-2]
        return {"prev_day_high": float(prev["high"].max()), "prev_day_low": float(prev["low"].min())}

    def asian_range(self, df: pd.DataFrame) -> Dict[str, Optional[float]]:
        today = df[df["time"].dt.date == df["time"].iloc[-1].date()]
        asian = today[(today["time"].dt.time >= time(0, 0)) & (today["time"].dt.time <= time(6, 59))]
        if asian.empty:
            return {"asian_high": None, "asian_low": None}
        return {"asian_high": float(asian["high"].max()), "asian_low": float(asian["low"].min())}

    @staticmethod
    def is_sweep(candle: pd.Series, level: float, direction: str) -> bool:
        body = abs(float(candle["close"] - candle["open"]))
        wick_up = float(candle["high"] - max(candle["open"], candle["close"]))
        wick_dn = float(min(candle["open"], candle["close"]) - candle["low"])

        if direction == "below":
            return candle["low"] < level and candle["close"] > level and wick_dn > body
        return candle["high"] > level and candle["close"] < level and wick_up > body

    def displacement(self, df: pd.DataFrame) -> Dict[str, object]:
        a = atr(df, 14)
        last = df.iloc[-1]
        rng = float(last["high"] - last["low"])
        body = abs(float(last["close"] - last["open"]))
        atrv = float(a.iloc[-1]) if not np.isnan(a.iloc[-1]) else 0.0
        is_disp = atrv > 0 and rng > 1.5 * atrv and body > 0.7 * rng
        direction = "bullish" if last["close"] > last["open"] else "bearish"
        return {"is_displacement": is_disp, "direction": direction, "range": rng, "atr": atrv}

    def detect_fvg(self, df: pd.DataFrame, symbol: str) -> List[Dict[str, object]]:
        zones = self.active_fvg.setdefault(symbol, [])
        if len(df) < 3:
            return zones
        c1, _, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        now = df.iloc[-1]["time"]

        if c1["high"] < c3["low"]:
            zones.append({"type": "bullish", "low": float(c1["high"]), "high": float(c3["low"]), "time": now})
        if c1["low"] > c3["high"]:
            zones.append({"type": "bearish", "low": float(c3["high"]), "high": float(c1["low"]), "time": now})

        return zones[-20:]

    def detect_order_block(self, df: pd.DataFrame, displacement_info: Dict[str, object]) -> Optional[Dict[str, float]]:
        if not displacement_info["is_displacement"]:
            return None
        direction = displacement_info["direction"]
        for i in range(len(df) - 2, -1, -1):
            c = df.iloc[i]
            bullish_candle = c["close"] > c["open"]
            bearish_candle = c["close"] < c["open"]
            if direction == "bullish" and bearish_candle:
                return {
                    "type": "bullish",
                    "low": float(c["low"]),
                    "high": float(c["high"]),
                    "mid": float((c["low"] + c["high"]) / 2),
                    "strength": float(displacement_info["range"]),
                }
            if direction == "bearish" and bullish_candle:
                return {
                    "type": "bearish",
                    "low": float(c["low"]),
                    "high": float(c["high"]),
                    "mid": float((c["low"] + c["high"]) / 2),
                    "strength": float(displacement_info["range"]),
                }
        return None

    def in_london_or_ny(self, ts: datetime) -> bool:
        h = ts.hour
        return 7 <= h <= 20
