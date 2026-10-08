"""Promoción con trazabilidad completa (Etapa 4.9).

`promoted.csv`: params + gates + ranks + dsr(S,N_eff) + edge_p_adj + fdr +
pbo/wfe + jitter + peor celda + régimen + forward(Holm) + lots + SL/TP +
cost_cfg + manifest_hashes + master_hash + ledger_snapshot_hash + commit + magic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROMOTED_COLS = [
    "set_id", "strategy_family", "combo", "params", "gates", "regime",
    "rank_sharpe", "rank_pf", "rank_ntrades", "dsr_S", "dsr_Neff",
    "edge_p_adj", "fdr_adj_p", "pbo", "wfe", "jitter", "worst_cell_pf",
    "forward_pf", "forward_sharpe", "forward_holm", "lots", "sl_pct", "tp_pct",
    "cost_cfg", "manifest_hashes", "master_hash", "ledger_snapshot_hash",
    "commit", "magic",
]


def promote_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {c: record.get(c) for c in PROMOTED_COLS}
    for k, v in row.items():
        if isinstance(v, (dict, list)):
            row[k] = str(v)
    return row


def write_promoted(rows: list[dict[str, Any]], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([promote_row(r) for r in rows],
                       columns=PROMOTED_COLS) if rows else pd.DataFrame(columns=PROMOTED_COLS)
    df.to_csv(p, index=False)
    return p
