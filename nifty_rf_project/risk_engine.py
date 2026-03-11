"""Volatility-based risk model and trade management helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskPlan:
    side: str
    entry: float
    sl: float
    tp: float
    atr: float
    lot: float


class RiskEngine:
    def __init__(self, lot_size: float = 0.04, daily_loss_limit_pct: float = 0.03) -> None:
        self.lot_size = lot_size
        self.daily_loss_limit_pct = daily_loss_limit_pct

    def build_plan(
        self,
        side: str,
        entry: float,
        atr: float,
        liquidity_level: float,
        next_liquidity_level: Optional[float],
    ) -> RiskPlan:
        if side == "buy":
            sl = liquidity_level - (1.2 * atr)
            tp_atr = entry + (2.5 * atr)
            tp = min(next_liquidity_level, tp_atr) if next_liquidity_level else tp_atr
        else:
            sl = liquidity_level + (1.2 * atr)
            tp_atr = entry - (2.5 * atr)
            tp = max(next_liquidity_level, tp_atr) if next_liquidity_level else tp_atr

        return RiskPlan(side=side, entry=entry, sl=sl, tp=tp, atr=atr, lot=self.lot_size)

    @staticmethod
    def break_even_trigger(entry: float, current_price: float, atr: float, side: str) -> bool:
        if side == "buy":
            return current_price - entry >= atr
        return entry - current_price >= atr

    @staticmethod
    def trailing_stop(current_price: float, atr: float, side: str) -> float:
        if side == "buy":
            return current_price - atr
        return current_price + atr

    def daily_loss_exceeded(self, day_pnl: float, balance: float) -> bool:
        return abs(day_pnl) >= balance * self.daily_loss_limit_pct and day_pnl < 0
