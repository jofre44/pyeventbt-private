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

    # D1 (y superiores): las barras son diarias (00:00) y no existe sesión
    # intradía 08:00/21:00. Adaptación declarada en spec: una oportunidad por
    # barra + time-stop equivalente (flat en cada barra nueva = holding 1 barra).
    from datetime import timedelta as _td

    is_daily_tf = signal_timeframe.to_timedelta() >= _td(days=1)

    # Estado diario encapsulado por instancia (no global)
    orders_placed_today: dict[str, bool] = {}
    current_trading_date: dict[str, Any] = {}

    def _place_breakouts(event, modules, symbol, stf, sid,
                         period: int, std: float, placed: dict):
        """Una oportunidad por barra/día si está flat: STOPs en bandas.
        Devuelve lista de SignalEvent o None. Común a rama intradía y D1."""
        from pyeventbt.events.events import OrderType, SignalEvent, SignalType
        from pyeventbt.indicators.indicators import BollingerBands

        if placed.get(symbol, False):
            return None
        if modules.PORTFOLIO.get_number_of_strategy_open_positions_by_symbol(symbol)["TOTAL"] > 0:
            return None
        if modules.PORTFOLIO.get_number_of_strategy_pending_orders_by_symbol(symbol)["TOTAL"] > 0:
            return None
        if event.timeframe != stf:
            return None
        bars_needed = period + 10
        indicator_bars = modules.DATA_PROVIDER.get_latest_bars(symbol, stf, bars_needed)
        if indicator_bars is None or indicator_bars.height < bars_needed:
            return None
        close = indicator_bars.select("close").to_numpy().flatten()
        upper, _middle, lower = BollingerBands.compute(close, period, std)
        if np.isnan(upper[-1]) or np.isnan(lower[-1]):
            return None
        if modules.TRADING_CONTEXT == "BACKTEST":
            time_generated = event.datetime + stf.to_timedelta()
        else:
            time_generated = datetime.now()
        placed[symbol] = True
        return [
            SignalEvent(symbol=symbol, time_generated=time_generated, strategy_id=sid,
                        signal_type=SignalType.BUY, order_type=OrderType.STOP,
                        order_price=Decimal(str(upper[-1])),
                        sl=Decimal("0.0"), tp=Decimal("0.0")),
            SignalEvent(symbol=symbol, time_generated=time_generated, strategy_id=sid,
                        signal_type=SignalType.SELL, order_type=OrderType.STOP,
                        order_price=Decimal(str(lower[-1])),
                        sl=Decimal("0.0"), tp=Decimal("0.0")),
        ]

    def bbands_breakout(event, modules):  # type: ignore[no-untyped-def]
        symbol = event.symbol
        signal_events = []
        current_time = event.datetime.time()
        current_date = event.datetime.date()

        open_positions = modules.PORTFOLIO.get_number_of_strategy_open_positions_by_symbol(symbol)
        pending_orders = modules.PORTFOLIO.get_number_of_strategy_pending_orders_by_symbol(symbol)

        if is_daily_tf:
            # Rama D1+: sin sesión intradía. Cada barra nueva cierra todo
            # (time-stop de 1 barra) y abre una sola oportunidad si está flat.
            if current_trading_date.get(symbol) != current_date:
                current_trading_date[symbol] = current_date
                orders_placed_today[symbol] = False
                if open_positions["TOTAL"] > 0:
                    logger.debug(f"{event.datetime} - Closing all positions for {symbol}")
                    modules.EXECUTION_ENGINE.close_all_strategy_positions()
                if pending_orders["TOTAL"] > 0:
                    logger.debug(f"{event.datetime} - Cancelling all pending orders for {symbol}")
                    modules.EXECUTION_ENGINE.cancel_all_strategy_pending_orders()
                    return
                return _place_breakouts(
                    event, modules, symbol, signal_timeframe, strategy_id,
                    bb_period, bb_std_dev, orders_placed_today)
            return

        # Rama intradía: reset diario + time-stop 21:00 + ventana 08:00.
        if current_trading_date.get(symbol) != current_date:
            current_trading_date[symbol] = current_date
            orders_placed_today[symbol] = False

        # Time-stop: cierra todo a las 21:00 (barra cerrada, sin look-ahead)
        if current_time >= time(close_h, close_m):
            if open_positions["TOTAL"] > 0:
                logger.debug(f"{event.datetime} - Closing all positions for {symbol}")
                modules.EXECUTION_ENGINE.close_all_strategy_positions()
            if pending_orders["TOTAL"] > 0:
                logger.debug(f"{event.datetime} - Cancelling all pending orders for {symbol}")
                modules.EXECUTION_ENGINE.cancel_all_strategy_pending_orders()
            return

        if current_time >= time(ord_h, ord_m):
            placed = _place_breakouts(
                event, modules, symbol, signal_timeframe, strategy_id,
                bb_period, bb_std_dev, orders_placed_today)
            if placed:
                signal_events.extend(placed)

        return signal_events

    # Metadatos para el ledger/trazabilidad
    bbands_breakout.pipeline_params = p.model_dump()  # type: ignore[attr-defined]
    bbands_breakout.strategy_family = "bbands_breakout"  # type: ignore[attr-defined]
    return bbands_breakout
