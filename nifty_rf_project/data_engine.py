"""MetaTrader 5 data and connection engine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import MetaTrader5 as mt5
import pandas as pd


@dataclass
class MT5Credentials:
    login: int
    password: str
    server: str
    terminal_path: Optional[str] = None


TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
}


class DataEngine:
    def __init__(self, credentials: MT5Credentials, symbols: List[str]) -> None:
        self.credentials = credentials
        self.symbols = symbols
        self.logger = logging.getLogger(self.__class__.__name__)

    def connect(self) -> bool:
        initialized = mt5.initialize(path=self.credentials.terminal_path)
        if not initialized:
            self.logger.error("MT5 initialize failed: %s", mt5.last_error())
            return False

        logged_in = mt5.login(
            login=self.credentials.login,
            password=self.credentials.password,
            server=self.credentials.server,
        )
        if not logged_in:
            self.logger.error("MT5 login failed: %s", mt5.last_error())
            mt5.shutdown()
            return False

        if not self.ensure_symbols():
            return False

        self.logger.info("Connected and authenticated to MT5")
        return True

    def reconnect(self, max_retries: int = 10, retry_sleep_s: int = 5) -> bool:
        for attempt in range(1, max_retries + 1):
            self.logger.warning("Reconnecting to MT5 (attempt %s/%s)", attempt, max_retries)
            mt5.shutdown()
            if self.connect():
                return True
            time.sleep(retry_sleep_s)
        return False

    def is_connected(self) -> bool:
        return mt5.terminal_info() is not None and mt5.account_info() is not None

    def ensure_symbols(self) -> bool:
        for symbol in self.symbols:
            info = mt5.symbol_info(symbol)
            if info is None:
                self.logger.error("Symbol unavailable: %s", symbol)
                return False
            if not info.visible and not mt5.symbol_select(symbol, True):
                self.logger.error("Failed to enable symbol: %s", symbol)
                return False
        return True

    def get_rates(self, symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
        tf = TIMEFRAME_MAP[timeframe]
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates for {symbol} {timeframe}. Last error: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def get_tick(self, symbol: str) -> Dict[str, float]:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick for {symbol}. Last error: {mt5.last_error()}")
        return {
            "symbol": symbol,
            "time": datetime.fromtimestamp(tick.time),
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": tick.ask - tick.bid,
        }

    def shutdown(self) -> None:
        mt5.shutdown()
