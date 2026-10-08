"""Signal factory pipeline-ready de bbands_breakout.

Derivada de `example_bbands_breakout.py` con cambios:
  - Sin estado global: cada `make_signal_fn(params)` crea su propio closure
    (reentrante; N trials en paralelo no se pisan).
  - Params validados con `BBandsParams` (falla ante clave desconocida/NaN).
  - Señal SOLO sobre barras cerradas; `delay_bars=1` vía `time_generated`
    en la barra siguiente (igual que el ejemplo).
  - `MaxOrdenes=1` y time-stop diario 21:00 (spec congelada).

Modo BACKTEST/LIVE idéntico al ejemplo original.
"""
from __future__ import annotations

from datetime import datetime, time
from decimal import Decimal
from typing import Any, Callable
import logging

import numpy as np

logger = logging.getLogger("pyeventbt")


def make_signal_fn(
    params: dict[str, Any],
    *,
    strategy_id: str = "11313",
    signal_timeframe=None,
) -> Callable:
    """Devuelve `fn(event, modules)` lista para `strategy.custom_signal_engine`."""
    from strategies.bbands_breakout.params import filter_params

    p = filter_params(params) if not hasattr(params, "model_dump") else params
    bb_period = int(p.bb_period)
    bb_std_dev = float(p.bb_std_dev)
    close_h, close_m = int(p.close_hour), int(p.close_minute)
    ord_h, ord_m = int(p.order_placement_hour), int(p.order_placement_minute)

    if signal_timeframe is None:
        from pyeventbt.strategy.core.strategy_timeframes import StrategyTimeframes

        signal_timeframe = StrategyTimeframes.ONE_HOUR

    # Estado diario encapsulado por instancia (no global)
    orders_placed_today: dict[str, bool] = {}
    current_trading_date: dict[str, Any] = {}

    def bbands_breakout(event, modules):  # type: ignore[no-untyped-def]
        from pyeventbt.events.events import OrderType, SignalEvent, SignalType
        from pyeventbt.indicators.indicators import BollingerBands

        symbol = event.symbol
        signal_events = []
        current_time = event.datetime.time()
        current_date = event.datetime.date()

        if current_trading_date.get(symbol) != current_date:
            current_trading_date[symbol] = current_date
            orders_placed_today[symbol] = False

        open_positions = modules.PORTFOLIO.get_number_of_strategy_open_positions_by_symbol(symbol)
        pending_orders = modules.PORTFOLIO.get_number_of_strategy_pending_orders_by_symbol(symbol)

        # Time-stop: cierra todo a las 21:00 (barra cerrada, sin look-ahead)
        if current_time >= time(close_h, close_m):
            if open_positions["TOTAL"] > 0:
                logger.info(f"{event.datetime} - Closing all positions for {symbol}")
                modules.EXECUTION_ENGINE.close_all_strategy_positions()
            if pending_orders["TOTAL"] > 0:
                logger.info(f"{event.datetime} - Cancelling all pending orders for {symbol}")
                modules.EXECUTION_ENGINE.cancel_all_strategy_pending_orders()
            return

        if (
            current_time >= time(ord_h, ord_m)
            and not orders_placed_today.get(symbol, False)
            and pending_orders["TOTAL"] == 0
            and open_positions["TOTAL"] == 0
            and event.timeframe == signal_timeframe
        ):
            bars_needed = bb_period + 10
            indicator_bars = modules.DATA_PROVIDER.get_latest_bars(symbol, signal_timeframe, bars_needed)
            if indicator_bars is None or indicator_bars.height < bars_needed:
                return

            close = indicator_bars.select("close").to_numpy().flatten()
            upper, _middle, lower = BollingerBands.compute(close, bb_period, bb_std_dev)
            current_upper, current_lower = upper[-1], lower[-1]
            if np.isnan(current_upper) or np.isnan(current_lower):
                return

            if modules.TRADING_CONTEXT == "BACKTEST":
                time_generated = event.datetime + signal_timeframe.to_timedelta()
            else:
                time_generated = datetime.now()

            signal_events.append(
                SignalEvent(
                    symbol=symbol,
                    time_generated=time_generated,
                    strategy_id=strategy_id,
                    signal_type=SignalType.BUY,
                    order_type=OrderType.STOP,
                    order_price=Decimal(str(current_upper)),
                    sl=Decimal("0.0"),
                    tp=Decimal("0.0"),
                )
            )
            signal_events.append(
                SignalEvent(
                    symbol=symbol,
                    time_generated=time_generated,
                    strategy_id=strategy_id,
                    signal_type=SignalType.SELL,
                    order_type=OrderType.STOP,
                    order_price=Decimal(str(current_lower)),
                    sl=Decimal("0.0"),
                    tp=Decimal("0.0"),
                )
            )
            orders_placed_today[symbol] = True

        return signal_events

    # Metadatos para el ledger/trazabilidad
    bbands_breakout.pipeline_params = p.model_dump()  # type: ignore[attr-defined]
    bbands_breakout.strategy_family = "bbands_breakout"  # type: ignore[attr-defined]
    return bbands_breakout
