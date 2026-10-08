"""Estrategias nulas por construcción (Meta-etapa M): miden el FPR del pipeline.

Tres kinds con la misma frecuencia/holding aproximado que la estrategia real:
  - random_entries: dirección aleatoria diaria + time-stop (mimica estructura).
  - shuffled_signals: mismos días de entrada que el nominal pero permutados.
  - indicator_on_noise: Bollinger sobre closes iid-permutados (indicador ciego).
Fábricas compatibles con el adaptador: fn(params, strategy_id, signal_timeframe).
"""
from __future__ import annotations

from datetime import time
from decimal import Decimal
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class NullParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=0, ge=0)
    kind: str = Field(default="random_entries")
    close_hour: int = 21
    close_minute: int = 0
    order_hour: int = 8
    order_minute: int = 0


NULL_SPACE = {"seed": {"type": "int", "low": 0, "high": 199, "step": 1}}


def _base_state():
    return {"placed": {}, "day": {}}


def make_random_entries(params: dict[str, Any], *, strategy_id: str = "90001",
                        signal_timeframe=None):
    p = NullParams(**params)
    rng = np.random.default_rng(10_000 + p.seed)
    # Secuencia pre-generada por día (determinista; no depende del orden de eventos)
    flips = {}
    state = _base_state()

    def _signal(event, modules):
        from pyeventbt.events.events import OrderType, SignalEvent, SignalType

        sym = event.symbol
        day = event.datetime.date()
        if state["day"].get(sym) != day:
            state["day"][sym] = day
            state["placed"][sym] = False
            flips[day] = rng.random()
        if event.datetime.time() >= time(p.close_hour, p.close_minute):
            if modules.PORTFOLIO.get_number_of_strategy_open_positions_by_symbol(sym)["TOTAL"] > 0:
                modules.EXECUTION_ENGINE.close_all_strategy_positions()
            return
        if (event.datetime.time() >= time(p.order_hour, p.order_minute)
                and not state["placed"].get(sym)
                and event.timeframe == signal_timeframe):
            state["placed"][sym] = True
            side = SignalType.BUY if flips[day] < 0.5 else SignalType.SELL
            px = Decimal(str(modules.DATA_PROVIDER.get_latest_bars(
                sym, signal_timeframe, 1).select("close").to_numpy().flatten()[-1]))
            tgen = event.datetime + signal_timeframe.to_timedelta()
            return [SignalEvent(symbol=sym, time_generated=tgen, strategy_id=strategy_id,
                                signal_type=side, order_type=OrderType.MARKET,
                                order_price=px, sl=Decimal("0.0"), tp=Decimal("0.0"))]

    return _signal


def make_shuffled_signals(params: dict[str, Any], *, strategy_id: str = "90002",
                           signal_timeframe=None, entry_days: list | None = None):
    """Días de entrada fijos (los del nominal) permutados por seed: misma
    frecuencia, timing destruido."""
    p = NullParams(**params)
    rng = np.random.default_rng(20_000 + p.seed)
    days = list(entry_days or [])
    perm = set(rng.choice(days, size=len(days), replace=False).tolist()) if days else set()
    state = _base_state()

    def _signal(event, modules):
        from pyeventbt.events.events import OrderType, SignalEvent, SignalType

        sym = event.symbol
        day = event.datetime.date()
        if state["day"].get(sym) != day:
            state["day"][sym] = day
            state["placed"][sym] = False
        if event.datetime.time() >= time(p.close_hour, p.close_minute):
            if modules.PORTFOLIO.get_number_of_strategy_open_positions_by_symbol(sym)["TOTAL"] > 0:
                modules.EXECUTION_ENGINE.close_all_strategy_positions()
            return
        if (day in perm and event.datetime.time() >= time(p.order_hour, p.order_minute)
                and not state["placed"].get(sym) and event.timeframe == signal_timeframe):
            state["placed"][sym] = True
            px = Decimal(str(modules.DATA_PROVIDER.get_latest_bars(
                sym, signal_timeframe, 1).select("close").to_numpy().flatten()[-1]))
            tgen = event.datetime + signal_timeframe.to_timedelta()
            return [SignalEvent(symbol=sym, time_generated=tgen, strategy_id=strategy_id,
                                signal_type=SignalType.BUY, order_type=OrderType.MARKET,
                                order_price=px, sl=Decimal("0.0"), tp=Decimal("0.0"))]

    _signal.permuted_days = perm  # type: ignore[attr-defined]
    return _signal


def make_indicator_on_noise(params: dict[str, Any], *, strategy_id: str = "90003",
                             signal_timeframe=None):
    """Bollinger sobre serie iid-permutada (misma marginal, sin estructura):
    el indicador no puede extraer nada."""
    p = NullParams(**params)
    rng = np.random.default_rng(30_000 + p.seed)
    state = _base_state()

    def _signal(event, modules):
        from pyeventbt.events.events import OrderType, SignalEvent, SignalType
        from pyeventbt.indicators.indicators import BollingerBands

        sym = event.symbol
        day = event.datetime.date()
        if state["day"].get(sym) != day:
            state["day"][sym] = day
            state["placed"][sym] = False
        if event.datetime.time() >= time(p.close_hour, p.close_minute):
            if modules.PORTFOLIO.get_number_of_strategy_open_positions_by_symbol(sym)["TOTAL"] > 0:
                modules.EXECUTION_ENGINE.close_all_strategy_positions()
            return
        if (event.datetime.time() >= time(p.order_hour, p.order_minute)
                and not state["placed"].get(sym) and event.timeframe == signal_timeframe):
            bars = modules.DATA_PROVIDER.get_latest_bars(sym, signal_timeframe, 30)
            if bars is None or bars.height < 30:
                return
            close = bars.select("close").to_numpy().flatten().astype(float)
            noise = rng.permutation(close)  # destruye orden/estructura
            upper, _, lower = BollingerBands.compute(noise, 20, 2.0)
            if np.isnan(upper[-1]) or np.isnan(lower[-1]):
                return
            last = float(close[-1])
            side = None
            if last > upper[-1]:
                side = SignalType.SELL
            elif last < lower[-1]:
                side = SignalType.BUY
            if side is None:
                return
            state["placed"][sym] = True
            tgen = event.datetime + signal_timeframe.to_timedelta()
            return [SignalEvent(symbol=sym, time_generated=tgen, strategy_id=strategy_id,
                                signal_type=side, order_type=OrderType.MARKET,
                                order_price=Decimal(str(last)), sl=Decimal("0.0"),
                                tp=Decimal("0.0"))]

    return _signal


NULL_FACTORIES = {
    "random_entries": "pyeventbt.calibration.null_strategies:make_random_entries",
    "shuffled_signals": "pyeventbt.calibration.null_strategies:make_shuffled_signals",
    "indicator_on_noise": "pyeventbt.calibration.null_strategies:make_indicator_on_noise",
    "drift_rider": "pyeventbt.calibration.null_strategies:make_drift_rider",
}
NULL_PARAMS_CLASS = "pyeventbt.calibration.null_strategies:NullParams"


def make_drift_rider(params: dict[str, Any], *, strategy_id: str = "90004",
                     signal_timeframe=None):
    """Sonda LONG-only para el power leg: sobre drift plantado alcista tiene
    edge por construcción (captura el drift completo menos costes)."""
    p = NullParams(**params)
    state = _base_state()

    def _signal(event, modules):
        from pyeventbt.events.events import OrderType, SignalEvent, SignalType

        sym = event.symbol
        day = event.datetime.date()
        if state["day"].get(sym) != day:
            state["day"][sym] = day
            state["placed"][sym] = False
        if event.datetime.time() >= time(p.close_hour, p.close_minute):
            if modules.PORTFOLIO.get_number_of_strategy_open_positions_by_symbol(sym)["TOTAL"] > 0:
                modules.EXECUTION_ENGINE.close_all_strategy_positions()
            return
        if (event.datetime.time() >= time(p.order_hour, p.order_minute)
                and not state["placed"].get(sym) and event.timeframe == signal_timeframe):
            state["placed"][sym] = True
            px = Decimal(str(modules.DATA_PROVIDER.get_latest_bars(
                sym, signal_timeframe, 1).select("close").to_numpy().flatten()[-1]))
            tgen = event.datetime + signal_timeframe.to_timedelta()
            return [SignalEvent(symbol=sym, time_generated=tgen, strategy_id=strategy_id,
                                signal_type=SignalType.BUY, order_type=OrderType.MARKET,
                                order_price=px, sl=Decimal("0.0"), tp=Decimal("0.0"))]

    return _signal
