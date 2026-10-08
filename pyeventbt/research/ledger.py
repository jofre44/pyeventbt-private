"""Ledger global de investigación (append-only) + registro de usos del lockbox.

`research/ledger.parquet` (polars, sin pyarrow): cada trial de cada run.
`S_global` = nº de trials del programa sobre la misma ventana.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml

logger = logging.getLogger("pyeventbt.pipeline")

LEDGER_COLS = [
    "run_id", "strategy_family", "combo", "trial_number", "params_hash",
    "window", "sharpe_net", "n_trades", "daily_returns_ref", "timestamp",
]


def append_trials(ledger_path: str | Path, rows: list[dict[str, Any]]) -> int:
    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    clean = []
    for r in rows:
        rec = {c: r.get(c) for c in LEDGER_COLS if c != "timestamp"}
        rec["timestamp"] = r.get("timestamp", now)
        clean.append(rec)
    new = pl.DataFrame(clean, schema={c: pl.String for c in LEDGER_COLS} | {})
    # Tipos numéricos donde corresponde
    new = new.with_columns([
        pl.col("trial_number").cast(pl.Int64, strict=False),
        pl.col("sharpe_net").cast(pl.Float64, strict=False),
        pl.col("n_trades").cast(pl.Int64, strict=False),
    ])
    if p.exists():
        old = pl.read_parquet(p)
        # Protección append-only: nunca encoger
        if len(old) > 0 and len(new) == 0:
            raise RuntimeError("append vacío sobre ledger existente: abortado")
        prev_len = len(old)
        combined = pl.concat([old, new], how="diagonal")
        if len(combined) < prev_len:
            raise RuntimeError("ledger append-only violado: el combinado es menor")
        combined.write_parquet(p)
    else:
        new.write_parquet(p)
    return len(new)


def read_ledger(ledger_path: str | Path) -> pl.DataFrame:
    p = Path(ledger_path)
    if not p.exists():
        return pl.DataFrame({c: [] for c in LEDGER_COLS})
    return pl.read_parquet(p)


def s_global(ledger_path: str | Path, window: str) -> int:
    df = read_ledger(ledger_path)
    if df.is_empty():
        return 0
    return int(df.filter(pl.col("window") == window).height)


def register_lockbox_use(registry_path: str | Path, entry: dict[str, Any]) -> None:
    """Registra UN uso del FORWARD. Sin reintentos: si ya existe
    (familia, símbolo, TF, set) => RuntimeError (run contaminado)."""
    p = Path(registry_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    uses: list = []
    if p.exists():
        uses = yaml.safe_load(open(p, encoding="utf-8")) or []
    key = (entry.get("family"), entry.get("symbol"), entry.get("timeframe"), entry.get("set_id"))
    for u in uses:
        if (u.get("family"), u.get("symbol"), u.get("timeframe"), u.get("set_id")) == key:
            raise RuntimeError(f"lockbox ya consumido para {key}: segundo uso prohibido")
    entry = dict(entry)
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    uses.append(entry)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(uses, f, sort_keys=True)
