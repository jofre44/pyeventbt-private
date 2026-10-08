"""Orquestador de validación sintética por representante (Etapa 3).

Flujo: H1 real → fit GJR → validación ±20% → universo sintético →
1 backtest real + N sintéticos (mismos params/costs) → ranks + veredicto.
Incluye re-scoring estresado (spread×2,+1 tick) sin re-simular.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger("pyeventbt.pipeline")

TF_MIN = {"M1": 1, "H1": 60, "H4": 240, "D1": 1440}


def load_real_m1(symbol: str, source_csv: str | Path, window: tuple[datetime, datetime]) -> pl.DataFrame:
    """M1 real en ventana (ordenado, sin nulos). Base del fit y del range-H1."""
    from pyeventbt.optimization.adapter import normalize_source_frame

    lf = pl.scan_csv(
        str(source_csv), has_header=False,
        new_columns=["date", "time", "open", "high", "low", "close", "tickvol", "volume", "spread"],
    )
    df = normalize_source_frame(lf.collect(streaming=True)).sort("datetime")
    w0, w1 = window
    return df.filter((pl.col("datetime") >= w0) & (pl.col("datetime") <= w1)).drop_nulls()


def load_real_h1(symbol: str, source_csv: str | Path, window: tuple[datetime, datetime]) -> pl.DataFrame:
    df = load_real_m1(symbol, source_csv, window)
    agg = [pl.col("open").first().alias("open"), pl.col("high").max().alias("high"),
           pl.col("low").min().alias("low"), pl.col("close").last().alias("close"),
           pl.col("tickvol").sum().alias("tickvol"), pl.col("volume").sum().alias("volume"),
           pl.col("spread").mean().alias("spread")]
    return (df.group_by_dynamic("datetime", every="1h", closed="left").agg(agg)
              .sort("datetime").drop_nulls())


def _h1_range_mean(bars: pl.DataFrame) -> float:
    h1 = (bars.sort("datetime").group_by_dynamic("datetime", every="1h", closed="left")
          .agg([pl.col("high").max().alias("h"), pl.col("low").min().alias("l"),
                pl.col("close").last().alias("c")]).drop_nulls())
    r = ((h1["h"] - h1["l"]) / h1["c"].clip(lower_bound=1e-9)).to_numpy()
    return float(np.mean(r))


def run_consistency(
    params: dict[str, Any],
    combo: dict[str, Any],
    master: dict[str, Any],
    window: tuple[datetime, datetime],
    *,
    n_sims: int = 20,
    seed_base: int = 1500,
    cluster_id: int = 0,
    metrics: tuple[str, ...] = ("sharpe_net", "pf", "n_trades"),
    stress_spread_mult: float = 2.0,
) -> dict[str, Any]:
    """Devuelve {ranks, votes/verdict, synth_ok, validation, null_note}."""
    from pyeventbt.analytics.stats import summarize
    from pyeventbt.optimization.adapter import run_pyeventbt_once
    from pyeventbt.robustness import scoring as _sc
    from pyeventbt.robustness.synthetic import SyntheticMarketGenerator, _describe_returns, validate_synthetic

    symbol, timeframe = combo["symbol"], combo["timeframe"]
    # Simulación a M1 (mismo feed que el real): el motor re-muestrea a TF señal.
    real_m1 = load_real_m1(symbol, Path(master["universe"]["data_dir"]) / f"{symbol}.csv", window)
    closes = real_m1["close"].to_numpy()
    gen = SyntheticMarketGenerator(seed=seed_base).fit(
        closes, real_m1["high"].to_numpy(), real_m1["low"].to_numpy(),
        spread_mean=float(real_m1["spread"].mean()),
        spreads=real_m1["spread"].to_numpy(),
    )
    run_kw = {
        "costs": master["costs"], "cash": master["strategy"]["cash"],
        "factory_path": master["strategy"]["factory"],
        "params_class_path": master["strategy"]["params_class"],
        "data_dir": master["universe"]["data_dir"],
        "strategy_id": master["strategy"].get("strategy_id", "11313"),
        "account_currency": master["strategy"].get("account_currency", "USD"),
        "aux_fallback_rates": master["universe"].get("aux_fallback_rates", {}),
        "base_spread_points": master["costs"].get("base_spread_points", 20.0),
    }
    real = run_pyeventbt_once(params, combo, window=window, seed=seed_base, mode="real", **run_kw)
    real_sum = real["summary"]

    synth_sums = []
    synth_stats_acc = []
    start_dt = real_m1["datetime"][0]
    n_bars = len(real_m1)
    for i in range(n_sims):
        seed = seed_base * 1000 + cluster_id * 100 + i
        bars = gen.generate_bars(n_bars, start_dt, 1, seed,
                                 start_close=float(closes[0]), spread_mult=1.0)
        if i == 0:
            sc = bars["close"].to_numpy()
            synth_stats_acc.append(_describe_returns(
                np.diff(np.log(sc)) * 100.0, sc,
                bars["high"].to_numpy(), bars["low"].to_numpy()))
        r = run_pyeventbt_once(params, combo, window=(start_dt, real_m1["datetime"][-1]),
                               seed=seed, mode="synthetic", bars_df=bars, **run_kw)
        synth_sums.append(r["summary"])

    validation = validate_synthetic(gen.stats_real, synth_stats_acc[0], tol=0.20)
    # Rango medio en H1 (comparable con el real agregado): sustituye el de M1
    h1_real_range = _h1_range_mean(real_m1)
    h1_synth_range = _h1_range_mean(bars)
    validation["mean_range_h1"] = {
        "real": h1_real_range, "synth": h1_synth_range,
        "rel_dev": abs(h1_synth_range - h1_real_range) / max(h1_real_range, 1e-9),
        "pass": bool(abs(h1_synth_range - h1_real_range) / max(h1_real_range, 1e-9) <= 0.20),
    }
    validation["overall"] = bool(validation.get("overall")
                                 and validation["mean_range_h1"]["pass"])
    cfg = master["validation"]["consistency"]
    ranks = {}
    for m in metrics:
        ranks[m] = _sc.score_percentile_rank(real_sum[m], [s[m] for s in synth_sums])
    cons = _sc.consistency_verdict(ranks, quorum=0.5,
                                   low=cfg["percentile_low"], high=cfg["percentile_high"])
    # Re-scoring estresado sin re-simular: spread×2 sobre el real
    stressed_kw = dict(run_kw)
    stressed_kw["costs"] = dict(run_kw["costs"], spread_mult=run_kw["costs"].get("spread_mult", 1.0) * stress_spread_mult)
    stressed = run_pyeventbt_once(params, combo, window=window, seed=seed_base, mode="real", **stressed_kw)
    stress_ok = (stressed["summary"]["pf"] or 0) >= 1.0
    return {
        "ranks": ranks, "consistency": cons, "validation": validation,
        "real_summary": real_sum,
        "synth_means": {m: float(np.mean([s[m] for s in synth_sums if np.isfinite(s[m])])) for m in metrics},
        "stress_pf": stressed["summary"]["pf"], "stress_ok": bool(stress_ok),
        "n_sims": n_sims,
    }
