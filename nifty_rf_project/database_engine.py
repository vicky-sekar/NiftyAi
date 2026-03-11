"""SQLite storage for trades, predictions, and candle snapshots."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict

import pandas as pd


class DatabaseEngine:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    entry_time TEXT,
                    entry_price REAL,
                    sl REAL,
                    tp REAL,
                    atr REAL,
                    feature_vector TEXT,
                    ai_probability REAL,
                    trade_result INTEGER,
                    rr_achieved REAL,
                    model_version TEXT,
                    session_key TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    ts TEXT,
                    probability REAL,
                    features TEXT,
                    accepted INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candle_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timeframe TEXT,
                    ts TEXT,
                    o REAL,
                    h REAL,
                    l REAL,
                    c REAL,
                    v REAL
                )
                """
            )

    def insert_trade(self, payload: Dict[str, object]) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    symbol, entry_time, entry_price, sl, tp, atr, feature_vector,
                    ai_probability, trade_result, rr_achieved, model_version, session_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["symbol"],
                    payload["entry_time"],
                    payload["entry_price"],
                    payload["sl"],
                    payload["tp"],
                    payload["atr"],
                    json.dumps(payload["feature_vector"]),
                    payload["ai_probability"],
                    payload.get("trade_result"),
                    payload.get("rr_achieved"),
                    payload["model_version"],
                    payload["session_key"],
                ),
            )

    def insert_prediction(self, symbol: str, ts: str, probability: float, features: Dict[str, float], accepted: bool) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO predictions (symbol, ts, probability, features, accepted) VALUES (?, ?, ?, ?, ?)",
                (symbol, ts, probability, json.dumps(features), int(accepted)),
            )

    def insert_candle_snapshot(self, symbol: str, timeframe: str, candle: pd.Series) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO candle_snapshots (symbol, timeframe, ts, o, h, l, c, v)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    timeframe,
                    str(candle["time"]),
                    float(candle["open"]),
                    float(candle["high"]),
                    float(candle["low"]),
                    float(candle["close"]),
                    float(candle["tick_volume"]),
                ),
            )

    def load_trades_df(self) -> pd.DataFrame:
        with self._conn() as conn:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY id", conn)
        if df.empty:
            return df

        features = df["feature_vector"].apply(json.loads).apply(pd.Series)
        out = pd.concat([df, features], axis=1)
        out["trade_result"] = out["trade_result"].fillna(0).astype(int)
        out["rr_achieved"] = out["rr_achieved"].fillna(0.0).astype(float)
        return out

    def export_summary_csv(self, path: Path) -> None:
        with self._conn() as conn:
            df = pd.read_sql_query("SELECT * FROM trades", conn)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
