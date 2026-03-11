"""Production-grade MT5 SMC trading bot entrypoint."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, Optional

import MetaTrader5 as mt5

from ai_engine import AIEngine, FEATURE_COLUMNS
from data_engine import DataEngine, MT5Credentials
from database_engine import DatabaseEngine
from execution_engine import ExecutionEngine, TradeRequest
from risk_engine import RiskEngine
from smc_engine import SMCEngine, atr


@dataclass
class BotConfig:
    symbols: tuple[str, ...] = ("EURUSD", "USDJPY")
    lot_size: float = 0.04
    poll_seconds: int = 15
    max_spread: float = 0.00035
    rollover_start: dt_time = dt_time(21, 55)
    rollover_end: dt_time = dt_time(22, 10)
    news_block_start: Optional[dt_time] = None
    news_block_end: Optional[dt_time] = None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def in_trading_session(now: datetime) -> bool:
    return 7 <= now.hour <= 20


def in_window(now: datetime, start: Optional[dt_time], end: Optional[dt_time]) -> bool:
    if start is None or end is None:
        return False
    t = now.time()
    return start <= t <= end


def build_feature_vector(
    htf_trend: str,
    displacement_strength: float,
    liquidity_quality: float,
    atr_ratio: float,
    session_type: float,
    spread: float,
    distance_to_ob: float,
    distance_to_fvg: float,
    premium_discount_flag: float,
) -> Dict[str, float]:
    trend_strength_score = 1.0 if htf_trend == "bullish" else (-1.0 if htf_trend == "bearish" else 0.0)
    return {
        "trend_strength_score": trend_strength_score,
        "displacement_strength": displacement_strength,
        "liquidity_quality": liquidity_quality,
        "atr_volatility_ratio": atr_ratio,
        "session_type": session_type,
        "spread_value": spread,
        "distance_to_OB": distance_to_ob,
        "distance_to_FVG": distance_to_fvg,
        "premium_discount_flag": premium_discount_flag,
    }


def strategy_signal(symbol: str, data: DataEngine, smc: SMCEngine) -> Optional[Dict[str, float]]:
    h1 = data.get_rates(symbol, "H1", 400)
    m15 = data.get_rates(symbol, "M15", 500)
    m5 = data.get_rates(symbol, "M5", 500)

    highs_h1, lows_h1 = smc.detect_swings(h1)
    htf_trend = smc.trend_from_swings(highs_h1, lows_h1)

    eq_levels = smc.find_equal_levels(m15)
    highs_m15, lows_m15 = smc.detect_swings(m15)
    side = "bullish" if htf_trend == "bullish" else "bearish"
    structure = smc.detect_bos_choch(m15, highs_m15 if side == "bullish" else lows_m15, side)

    disp = smc.displacement(m5)
    ob = smc.detect_order_block(m5, disp)
    fvgs = smc.detect_fvg(m5, symbol)

    latest = m15.iloc[-1]
    liquidity_sweep = False
    liquidity_level = None
    liq_quality = 0.0

    if htf_trend == "bullish" and eq_levels["equal_lows"]:
        liquidity_level = eq_levels["equal_lows"][-1]
        liquidity_sweep = smc.is_sweep(latest, liquidity_level, "below")
        liq_quality = 1.0 if liquidity_sweep else 0.3
    elif htf_trend == "bearish" and eq_levels["equal_highs"]:
        liquidity_level = eq_levels["equal_highs"][-1]
        liquidity_sweep = smc.is_sweep(latest, liquidity_level, "above")
        liq_quality = 1.0 if liquidity_sweep else 0.3

    if not (liquidity_sweep and structure["bos"] and ob is not None):
        return None

    entry_price = float(m5.iloc[-1]["close"])
    ob_mid = float(ob["mid"])
    fvg_mid = 0.0
    if fvgs:
        last_fvg = fvgs[-1]
        fvg_mid = (last_fvg["low"] + last_fvg["high"]) / 2

    momentum_rejection = abs(float(m5.iloc[-1]["close"] - m5.iloc[-1]["open"])) > 0.5 * (
        float(m5.iloc[-1]["high"] - m5.iloc[-1]["low"])
    )
    if not momentum_rejection:
        return None

    atr_series = atr(m15, 14)
    atr_value = float(atr_series.iloc[-1])
    if atr_value <= 0:
        return None

    return {
        "symbol": symbol,
        "trend": htf_trend,
        "entry_side": "buy" if htf_trend == "bullish" else "sell",
        "entry_price": entry_price,
        "liquidity_level": float(liquidity_level) if liquidity_level else entry_price,
        "next_liquidity": None,
        "atr": atr_value,
        "displacement_strength": float(disp["range"]),
        "liquidity_quality": liq_quality,
        "distance_to_ob": abs(entry_price - ob_mid),
        "distance_to_fvg": abs(entry_price - fvg_mid) if fvg_mid else 0.0,
        "premium_discount_flag": 1.0 if entry_price < ob_mid else 0.0,
    }


def run_bot() -> None:
    setup_logging()
    logger = logging.getLogger("main")
    cfg = BotConfig()

    creds = MT5Credentials(
        login=int(os.getenv("MT5_LOGIN", "0")),
        password=os.getenv("MT5_PASSWORD", ""),
        server=os.getenv("MT5_SERVER", ""),
        terminal_path=os.getenv("MT5_TERMINAL_PATH"),
    )

    data = DataEngine(credentials=creds, symbols=list(cfg.symbols))
    smc = SMCEngine()
    ai = AIEngine(model_dir=Path("models"))
    risk = RiskEngine(lot_size=cfg.lot_size)
    exe = ExecutionEngine(max_trades_per_symbol_per_session=2)
    db = DatabaseEngine(Path("storage/trading_bot.db"))
    ai.load_latest()

    if not data.connect():
        raise RuntimeError("Failed to initialize MT5")

    baseline_metrics = {"profit_factor": 0.0, "drawdown": 999.0, "win_rate": 0.0}

    while True:
        try:
            now = datetime.utcnow()

            if not data.is_connected() and not data.reconnect():
                logger.error("Unable to reconnect MT5; sleeping")
                time.sleep(cfg.poll_seconds)
                continue

            if not in_trading_session(now):
                time.sleep(cfg.poll_seconds)
                continue

            if in_window(now, cfg.rollover_start, cfg.rollover_end) or in_window(now, cfg.news_block_start, cfg.news_block_end):
                time.sleep(cfg.poll_seconds)
                continue

            account = mt5.account_info()
            if account and risk.daily_loss_exceeded(account.profit, account.balance):
                logger.warning("Daily loss limit reached")
                time.sleep(cfg.poll_seconds)
                continue

            atr_map: Dict[str, float] = {}
            for symbol in cfg.symbols:
                tick = data.get_tick(symbol)
                if tick["spread"] > cfg.max_spread:
                    continue

                sig = strategy_signal(symbol, data, smc)
                if sig is None:
                    continue

                features = build_feature_vector(
                    htf_trend=sig["trend"],
                    displacement_strength=sig["displacement_strength"],
                    liquidity_quality=sig["liquidity_quality"],
                    atr_ratio=sig["atr"] / max(sig["entry_price"], 1e-9),
                    session_type=1.0,
                    spread=tick["spread"],
                    distance_to_ob=sig["distance_to_ob"],
                    distance_to_fvg=sig["distance_to_fvg"],
                    premium_discount_flag=sig["premium_discount_flag"],
                )

                for key in FEATURE_COLUMNS:
                    features.setdefault(key, 0.0)

                prob = ai.predict_probability(features)
                accepted = ai.should_trade(prob)
                db.insert_prediction(symbol, str(now), prob, features, accepted)

                if not accepted:
                    continue

                session_key = exe.session_key(now)
                if not exe.can_open_trade(symbol, session_key):
                    continue
                if exe.has_duplicate_entry(symbol, sig["entry_side"]):
                    continue

                plan = risk.build_plan(
                    side=sig["entry_side"],
                    entry=sig["entry_price"],
                    atr=sig["atr"],
                    liquidity_level=sig["liquidity_level"],
                    next_liquidity_level=sig["next_liquidity"],
                )

                ok = exe.send_market_order(
                    TradeRequest(
                        symbol=symbol,
                        side=plan.side,
                        lot=plan.lot,
                        entry=plan.entry,
                        sl=plan.sl,
                        tp=plan.tp,
                        comment=f"SMC_AI_{ai.model_meta.version}",
                    )
                )
                if ok:
                    exe.track_trade(symbol, session_key)
                    db.insert_trade(
                        {
                            "symbol": symbol,
                            "entry_time": str(now),
                            "entry_price": plan.entry,
                            "sl": plan.sl,
                            "tp": plan.tp,
                            "atr": plan.atr,
                            "feature_vector": features,
                            "ai_probability": prob,
                            "trade_result": None,
                            "rr_achieved": None,
                            "model_version": ai.model_meta.version,
                            "session_key": session_key,
                        }
                    )

                atr_map[symbol] = sig["atr"]

            exe.manage_open_positions(atr_map)

            trades_df = db.load_trades_df()
            changed, baseline_metrics = ai.retrain_if_needed(trades_df, baseline_metrics)
            if changed:
                logger.info("Model upgraded to %s", ai.model_meta.version)

            db.export_summary_csv(Path("storage/trade_summary.csv"))
            time.sleep(cfg.poll_seconds)

        except Exception as exc:
            logger.exception("Runtime loop exception: %s", exc)
            time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    run_bot()
