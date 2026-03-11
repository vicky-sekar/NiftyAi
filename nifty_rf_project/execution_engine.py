"""Order execution and position management for MT5."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import MetaTrader5 as mt5


@dataclass
class TradeRequest:
    symbol: str
    side: str
    lot: float
    entry: float
    sl: float
    tp: float
    comment: str


class ExecutionEngine:
    def __init__(self, max_trades_per_symbol_per_session: int = 2, magic: int = 20260311) -> None:
        self.max_trades = max_trades_per_symbol_per_session
        self.magic = magic
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session_trade_counter: Dict[str, int] = {}

    def _direction(self, side: str) -> int:
        return mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL

    def can_open_trade(self, symbol: str, session_key: str) -> bool:
        key = f"{session_key}:{symbol}"
        return self.session_trade_counter.get(key, 0) < self.max_trades

    def track_trade(self, symbol: str, session_key: str) -> None:
        key = f"{session_key}:{symbol}"
        self.session_trade_counter[key] = self.session_trade_counter.get(key, 0) + 1

    def has_duplicate_entry(self, symbol: str, side: str) -> bool:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return False
        side_type = self._direction(side)
        return any(p.type == side_type for p in positions)

    def send_market_order(self, req: TradeRequest, deviation: int = 20) -> bool:
        tick = mt5.symbol_info_tick(req.symbol)
        if tick is None:
            self.logger.error("Missing tick for %s", req.symbol)
            return False

        price = tick.ask if req.side == "buy" else tick.bid
        order = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": req.symbol,
            "volume": req.lot,
            "type": self._direction(req.side),
            "price": price,
            "sl": req.sl,
            "tp": req.tp,
            "deviation": deviation,
            "magic": self.magic,
            "comment": req.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(order)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error("Order failed for %s: %s", req.symbol, result)
            return False

        self.logger.info("Order placed for %s ticket=%s", req.symbol, result.order)
        return True

    def manage_open_positions(self, atr_by_symbol: Dict[str, float]) -> List[dict]:
        updates = []
        positions = mt5.positions_get()
        if not positions:
            return updates

        for p in positions:
            symbol = p.symbol
            atrv = atr_by_symbol.get(symbol)
            if atrv is None:
                continue

            side = "buy" if p.type == mt5.POSITION_TYPE_BUY else "sell"
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                continue
            current = tick.bid if side == "buy" else tick.ask
            entry = p.price_open
            profit_distance = current - entry if side == "buy" else entry - current

            new_sl = p.sl
            if profit_distance >= atrv:
                new_sl = max(p.sl, entry) if side == "buy" else min(p.sl, entry)

            trail = (current - atrv) if side == "buy" else (current + atrv)
            if side == "buy" and trail > new_sl:
                new_sl = trail
            if side == "sell" and (new_sl == 0 or trail < new_sl):
                new_sl = trail

            if new_sl != p.sl and new_sl > 0:
                mod_req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": p.ticket,
                    "symbol": symbol,
                    "sl": new_sl,
                    "tp": p.tp,
                }
                res = mt5.order_send(mod_req)
                updates.append({"ticket": p.ticket, "symbol": symbol, "retcode": getattr(res, "retcode", None)})
        return updates

    @staticmethod
    def session_key(now: datetime) -> str:
        return f"{now.date()}_{'LDN_NY'}"
