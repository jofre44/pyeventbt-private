"""Adaptador único DMRI↔PyEventBT (Etapa 1, P0).

`run_pyeventbt_once(params, combo, *, costs, window, cash, seed, ...)`
  - valida params contra `params_class` (falla ante clave desconocida/NaN),
  - escribe OHLC real o sintético a temp dir en formato MT5-M1,
  - corre `Strategy + signal_fn + backtest()` idéntico para todos los trials,
  - devuelve frames + `summary()` + retornos diarios.
Modos: real | bars_df (sintético/nulo de Etapa 3).
"""
from __future__ import annotations

import importlib
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

logger = logging.getLogger("pyeventbt.pipeline")

SOURCE_DT_FMT = "%Y.%m.%d %H:%M:%S"


def import_dotted(path: str) -> Any:
    if ":" in path:
        mod, _, attr = path.partition(":")
    else:
        mod, _, attr = path.rpartition(".")
    if not mod or not attr:
        raise ValueError(f"ruta punteada inválida: {path}")
    return getattr(importlib.import_module(mod), attr)


def derive_seed(master_seed: int, trial_number: int) -> int:
    return (int(master_seed) + int(trial_number)) % 2**31


def tf_to_strategy_timeframe(tf: str):  # noqa: ANN202
    from pyeventbt.strategy.core.strategy_timeframes import StrategyTimeframes

    return {"M1": StrategyTimeframes.ONE_MIN, "H1": StrategyTimeframes.ONE_HOUR}.get(tf, StrategyTimeframes.ONE_HOUR)


def normalize_source_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Acepta hora HH:MM o HH:MM:SS, devuelve datetime + OHLCV+spread."""
    df = df.with_columns(
        pl.when(pl.col("time").str.len_chars() == 5)
        .then(pl.col("time") + ":00")
        .otherwise(pl.col("time"))
        .alias("time")
    ).with_columns(
        (pl.col("date") + " " + pl.col("time")).str.strptime(pl.Datetime, SOURCE_DT_FMT).alias("datetime")
    )
    return df


def slice_source_to_tmp(
    symbol: str, source_csv: str | Path, window: tuple[datetime, datetime], tmpdir: str | Path
) -> Path:
    lf = pl.scan_csv(
        str(source_csv), has_header=False,
        new_columns=["date", "time", "open", "high", "low", "close", "tickvol", "volume", "spread"],
    )
    df = normalize_source_frame(lf.collect(streaming=True)).sort("datetime")
    w0, w1 = window
    part = df.filter((pl.col("datetime") >= w0) & (pl.col("datetime") <= w1))
    if part.is_empty():
        raise ValueError(f"ventana {w0}→{w1} sin filas para {symbol} en {source_csv}")
    out = Path(tmpdir) / f"{symbol}.csv"
    (
        part.select([
            pl.col("datetime").dt.strftime("%Y.%m.%d").alias("date"),
            pl.col("datetime").dt.strftime("%H:%M:%S").alias("time"),
            "open", "high", "low", "close", "tickvol", "volume", "spread",
        ]).write_csv(str(out), include_header=False)
    )
    return out


def _needed_aux_symbols(tradeable: list[str], account_currency: str) -> list[str]:
    """Espejo mínimo de CSVDataConnector._create_auxiliary_symbol_list.
    Para XXXYYY: margin=XXX, profit=YYY; cada divisa != cuenta necesita el par
    de conversión (XXXUSD o USDYYY) presente en la lista soportada."""
    all_fx = ["AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF",
              "CADJPY", "CHFJPY", "EURAUD", "EURCAD", "EURCHF", "EURGBP",
              "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPCHF",
              "GBPJPY", "GBPNZD", "GBPUSD", "NZDCAD", "NZDCHF", "NZDJPY",
              "NZDUSD", "USDCAD", "USDCHF", "USDJPY", "USDSEK", "USDNOK",
              "USDMXN", "EURMXN", "GBPMXN"]
    aux: list[str] = []
    for symbol in tradeable:
        base, quote = symbol[:3], symbol[3:]
        for cur in (base, quote):
            if cur != account_currency:
                for fx in all_fx:
                    if fx not in tradeable and account_currency in fx and cur in fx and fx not in aux:
                        aux.append(fx)
                        break
    return aux


def _ensure_aux_csvs(
    tradeable: list[str],
    account_currency: str,
    window: tuple[datetime, datetime],
    tmpdir: str | Path,
    data_dir: str | Path,
    fallback_rates: dict[str, float],
) -> dict[str, str]:
    """Garantiza los CSV auxiliares de conversión de divisa en el temp dir.
    Origen preferido: CSV real en data_dir (slice). Fallback documentado:
    tasa constante de `fallback_rates` (neutra para ranking: Sharpe/PF/maxdd%
    invariantes a la constante; solo la escala absoluta en $ se ve afectada).
    Devuelve {símbolo: origen} para el manifiesto."""
    origins: dict[str, str] = {}
    w0, w1 = window
    for aux in _needed_aux_symbols(tradeable, account_currency):
        src = Path(data_dir) / f"{aux}.csv"
        if src.exists():
            slice_source_to_tmp(aux, src, (w0, w1), tmpdir)
            origins[aux] = "real"
            continue
        if aux not in fallback_rates:
            raise ValueError(
                f"sin CSV real para {aux} ni tasa en universe.aux_fallback_rates: "
                "añade el CSV del broker o la tasa fallback en el maestro"
            )
        rate = float(fallback_rates[aux])
        # Serie diaria constante con margen previo (el índice maestro usa max(first_dates))
        days = []
        d = (w0 - timedelta(days=5)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = (w1 + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        while d <= end:
            days.append(d)
            d += timedelta(days=1)
        df = pl.DataFrame({
            "datetime": days,
            "open": [rate] * len(days), "high": [rate] * len(days),
            "low": [rate] * len(days), "close": [rate] * len(days),
            "tickvol": [1] * len(days), "volume": [1] * len(days), "spread": [20] * len(days),
        })
        write_bars_df_to_tmp(aux, df, tmpdir)
        origins[aux] = f"fallback_constant:{rate}"
    return origins


def write_bars_df_to_tmp(symbol: str, bars: pl.DataFrame, tmpdir: str | Path) -> Path:
    out = Path(tmpdir) / f"{symbol}.csv"
    (
        bars.sort("datetime").select([
            pl.col("datetime").dt.strftime("%Y.%m.%d").alias("date"),
            pl.col("datetime").dt.strftime("%H:%M:%S").alias("time"),
            "open", "high", "low", "close", "tickvol", "volume", "spread",
        ]).write_csv(str(out), include_header=False)
    )
    return out


def run_pyeventbt_once(
    params: dict[str, Any],
    combo: dict[str, Any],
    *,
    costs: dict[str, Any],
    window: tuple[datetime, datetime],
    cash: float,
    seed: int,
    factory_path: str,
    params_class_path: str,
    data_dir: str | Path,
    strategy_id: str = "11313",
    mode: str = "real",
    bars_df: pl.DataFrame | None = None,
    base_spread_points: float = 20.0,
    keep_tmp: str | Path | None = None,
    account_currency: str = "USD",
    aux_fallback_rates: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Frontera única. `combo` = {symbol, timeframe, strategy_family}.
    `window` = (start, end) datetimes naive (hora servidor)."""
    from strategies.bbands_breakout.params import params_hash  # noqa - trazabilidad
    from pyeventbt.analytics.stats import summarize, to_daily_returns
    from pyeventbt.strategy.strategy import Strategy
    from pyeventbt.strategy.core.account_currencies import AccountCurrencies
    from pyeventbt.sizing_engine.core.configurations.sizing_engine_configurations import MinSizingConfig
    from pyeventbt.risk_engine.core.configurations.risk_engine_configurations import PassthroughRiskConfig

    params_cls = import_dotted(params_class_path)
    factory = import_dotted(factory_path)
    # Validación estricta: desconocida/NaN/fuera de rango => ValueError
    validated = params_cls(**params) if isinstance(params, dict) else params_cls(**params.model_dump())
    import math

    for k, v in validated.model_dump().items():
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"param '{k}' no finito")

    symbol, timeframe = combo["symbol"], combo["timeframe"]
    stf = tf_to_strategy_timeframe(timeframe)
    w0, w1 = window

    tmp = Path(keep_tmp) if keep_tmp else Path(tempfile.mkdtemp(prefix="pyeventbt_once_"))
    tmp.mkdir(parents=True, exist_ok=True)
    if mode == "real":
        if bars_df is not None:
            write_bars_df_to_tmp(symbol, bars_df, tmp)
        else:
            slice_source_to_tmp(symbol, Path(data_dir) / f"{symbol}.csv", (w0, w1), tmp)
    elif mode in ("synthetic", "null", "tick_replay"):
        if bars_df is None:
            raise ValueError(f"modo {mode} exige bars_df")
        write_bars_df_to_tmp(symbol, bars_df, tmp)
    else:
        raise ValueError(f"modo desconocido: {mode}")

    aux_origins = _ensure_aux_csvs(
        [symbol], account_currency, (w0, w1), tmp, data_dir, aux_fallback_rates or {}
    )

    spread_override = int(round(float(base_spread_points) * float(costs.get("spread_mult", 1.0))))

    strategy = Strategy()
    signal_fn = factory(validated.model_dump(), strategy_id=strategy_id, signal_timeframe=stf)
    strategy.custom_signal_engine(strategy_id=strategy_id, strategy_timeframes=[stf])(signal_fn)
    strategy.configure_predefined_sizing_engine(MinSizingConfig())
    strategy.configure_predefined_risk_engine(PassthroughRiskConfig())
    results = strategy.backtest(
        strategy_id=strategy_id,
        initial_capital=float(cash),
        symbols_to_trade=[symbol],
        csv_dir=str(tmp),
        backtest_name=f"{combo.get('strategy_family', 'strat')}_{symbol}_{timeframe}",
        start_date=w0,
        end_date=w1,
        export_backtest_csv=False,
        export_backtest_parquet=False,
        backtest_results_dir=str(tmp),
        account_currency=AccountCurrencies.USD,
        symbols_info_override={"*": {"spread": spread_override}},
    )

    pnl = results.pnl
    if not isinstance(pnl, pd.DataFrame):
        pnl = pd.DataFrame(pnl)
    trades = results.trades
    if not isinstance(trades, pd.DataFrame):
        trades = pd.DataFrame(trades)
    summary = summarize(pnl, trades, cash=float(cash))
    daily = to_daily_returns(pnl)
    return {
        "pnl": pnl,
        "trades": trades,
        "summary": summary,
        "daily": daily,
        "params_hash": params_hash(validated.model_dump()),
        "seed": int(seed),
        "mode": mode,
        "tmpdir": str(tmp),
        "aux_origins": aux_origins,
    }
