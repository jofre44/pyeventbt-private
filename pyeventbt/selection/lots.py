"""Sizing: Lots = min(Lots_dd, Lots_max_los, Lots_mean_loss) (Etapa 4.9)."""
from __future__ import annotations

import pandas as pd


def compute_lots(closed_net: pd.Series, lots_cfg: dict, base_lots: float = 0.01) -> dict:
    losses = closed_net[closed_net < 0].astype(float)
    max_los = float(-losses.min()) if len(losses) else 0.0
    mean_los = float(-losses.mean()) if len(losses) else 0.0
    curve = closed_net.cumsum()
    dd = float((curve.cummax() - curve).max()) if len(curve) else 1.0
    lots_dd = lots_cfg.get("max_dd_allow", 1200) / max(dd, 1e-9) * base_lots
    lots_max = lots_cfg.get("max_trade_lost_allow", 200) / max(max_los, 1e-9) * base_lots
    lots_mean = lots_cfg.get("mean_trade_lost_allow", 100) / max(mean_los, 1e-9) * base_lots
    lots = max(min(lots_dd, lots_max, lots_mean, 1.0), base_lots)
    return {"lots": round(float(lots), 2), "lots_dd": lots_dd,
            "lots_max_los": lots_max, "lots_mean_loss": lots_mean}
