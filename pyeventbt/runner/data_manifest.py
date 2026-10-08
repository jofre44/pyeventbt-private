"""Manifiesto de datos + data_check + embargo en tiempo + leakage firewall (Etapa 0).

- CSV MT5: `fecha,hora,open,high,low,close,tickvol,volume,spread` sin cabecera,
  fecha `YYYY.MM.DD`, hora `HH:MM` (sin segundos) o `HH:MM:SS`.
- Embargo EN TIEMPO: max(max_holding, warmup) + extra_bars, en unidad del TF señal.
- Leakage firewall: checks que devuelven FAIL (nunca warnings silenciosos).
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger("pyeventbt.pipeline")

TF_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440, "W1": 10080,
}


def tf_to_timedelta(tf: str) -> timedelta:
    return timedelta(minutes=TF_MINUTES[tf])


def file_hash_sampled(path: str | Path, sample_bytes: int = 1_000_000) -> str:
    """Hash rápido: tamaño + primer/último MB (evita hashear 4.5GB completos)."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(sample_bytes))
        if size > 2 * sample_bytes:
            f.seek(size - sample_bytes)
            h.update(f.read(sample_bytes))
    return h.hexdigest()[:16]


def _parse_dt_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    # Acepta "HH:MM" o "HH:MM:SS"
    lf = lf.with_columns(
        pl.when(pl.col("time").str.len_chars() == 5)
        .then(pl.col("time") + ":00")
        .otherwise(pl.col("time"))
        .alias("time")
    )
    return lf.with_columns(
        (pl.col("date") + " " + pl.col("time"))
        .str.strptime(pl.Datetime, "%Y.%m.%d %H:%M:%S")
        .alias("datetime")
    )


def build_manifest(symbol: str, csv_path: str | Path, broker_server_tz: str = "Europe/Athens") -> dict[str, Any]:
    p = Path(csv_path)
    lf = pl.scan_csv(
        str(p), has_header=False,
        new_columns=["date", "time", "open", "high", "low", "close", "tickvol", "volume", "spread"],
    )
    lf = _parse_dt_lazy(lf)
    df = (
        lf.with_columns([
            pl.col("open").cast(pl.Float64), pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64), pl.col("close").cast(pl.Float64),
            pl.col("spread").cast(pl.Float64),
        ])
        .sort("datetime")
        .collect(streaming=True)
    )
    dts = df["datetime"]
    diffs = dts.diff().dt.total_minutes()
    gap_mask = diffs > 1.5  # tolera M1; huecos de fin de semana son normales
    n_gaps = int(gap_mask.sum()) if len(df) else 0
    spread = df["spread"].drop_nulls()
    return {
        "symbol": symbol,
        "source": str(p),
        "rows": int(len(df)),
        "start": str(dts.min()),
        "end": str(dts.max()),
        "n_gaps_gt_1min": n_gaps,
        "n_duplicates": int(len(df) - df.unique("datetime").height),
        "n_null_ohlc": int(df.select(["open", "high", "low", "close"]).null_count().sum_horizontal().sum()),
        "spread_mean": float(spread.mean()) if len(spread) else float("nan"),
        "spread_p95": float(spread.quantile(0.95)) if len(spread) else float("nan"),
        "file_hash": file_hash_sampled(p),
        "file_bytes": int(p.stat().st_size),
        "broker_server_tz": broker_server_tz,
    }


def data_check(manifest: dict[str, Any], fail_on_weekend_gaps: bool = False) -> dict[str, Any]:
    """Firewall de integridad: cualquier violación estructural => FAIL."""
    fails: list[str] = []
    warns: list[str] = []
    if manifest["n_duplicates"] > 0:
        fails.append(f"duplicados datetime: {manifest['n_duplicates']}")
    if manifest["n_null_ohlc"] > 0:
        fails.append(f"OHLC nulos: {manifest['n_null_ohlc']}")
    if manifest["rows"] == 0:
        fails.append("cero filas")
    if manifest["n_gaps_gt_1min"] > 0 and fail_on_weekend_gaps:
        fails.append(f"gaps: {manifest['n_gaps_gt_1min']}")
    elif manifest["n_gaps_gt_1min"] > 0:
        warns.append(f"gaps>1min (fines de semana/feriados): {manifest['n_gaps_gt_1min']}")
    verdict = "FAIL" if fails else "PASS"
    return {"verdict": verdict, "fails": fails, "warns": warns, "manifest": manifest}


def compute_embargo(
    timeframe: str,
    max_holding_bars: int,
    warmup_bars: int,
    extra_bars: int = 10,
) -> dict[str, Any]:
    """Embargo en tiempo: max(max_holding, warmup) + extra, en unidad del TF."""
    bars = int(max(int(max_holding_bars), int(warmup_bars)) + int(extra_bars))
    delta = tf_to_timedelta(timeframe) * bars
    return {"bars": bars, "timeframe": timeframe, "delta": delta}


def apply_embargo(is_end: datetime, timeframe: str, **kwargs: Any) -> datetime:
    return is_end + compute_embargo(timeframe, **kwargs)["delta"]


def check_window_purity(
    is_end: datetime,
    forward_start: datetime,
    timeframe: str,
    max_holding_bars: int,
    warmup_bars: int,
    extra_bars: int = 10,
) -> dict[str, Any]:
    """Verifica que FORWARD empieza después del embargo desde fin de IS."""
    emb = compute_embargo(timeframe, max_holding_bars, warmup_bars, extra_bars)
    ok = forward_start >= is_end + emb["delta"]
    return {
        "verdict": "PASS" if ok else "FAIL",
        "embargo_bars": emb["bars"],
        "required_gap": str(emb["delta"]),
        "actual_gap": str(forward_start - is_end),
    }


def check_lockbox_untouched(lockbox_registry: str | Path, family: str, symbol: str, tf: str) -> dict[str, Any]:
    """El FORWARD (lockbox) no debe haberse consumido antes de las puertas."""
    import yaml

    p = Path(lockbox_registry)
    if not p.exists():
        return {"verdict": "PASS", "uses": 0}
    uses = yaml.safe_load(open(p, encoding="utf-8")) or []
    n = sum(1 for u in uses if u.get("family") == family and u.get("symbol") == symbol and u.get("timeframe") == tf)
    return {"verdict": "PASS" if n == 0 else "FAIL", "uses": n}
