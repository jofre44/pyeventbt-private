"""SL/TP por percentil MFE/MAE p10/p90 (aprox H1) o ATR. Nunca min/max (Etapa 4.9).

El motor no expone MFE nativo: se aproxima con la excursión máxima de barras
H1 entre open y close de cada trade. Si no hay barras, fallback ATR(14).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sl_tp(closed: pd.DataFrame, h1: pd.DataFrame | None, sl_tp_cfg: dict) -> dict:
    p_sl, p_tp = sl_tp_cfg.get("p_sl", 10), sl_tp_cfg.get("p_tp", 90)
    try:
        mfes, maes = [], []
        if h1 is not None and not closed.empty:
            bars = h1.copy()
            bars.index = pd.to_datetime(bars.index, utc=True).tz_convert(None)
            for _, t in closed.iterrows():
                ot = pd.Timestamp(t["OPEN_TIME"])
                ct = pd.Timestamp(t["CLOSE_TIME"])
                ot = ot.tz_convert(None) if getattr(ot, "tzinfo", None) else ot
                ct = ct.tz_convert(None) if getattr(ct, "tzinfo", None) else ct
                seg = bars[(bars.index >= ot) & (bars.index <= ct)]
                if seg.empty:
                    continue
                entry = float(t["ENTRY_PRICE"])
                is_long = "BUY" in str(t.get("DIRECTION", "")).upper()
                hi, lo = float(seg["high"].max()), float(seg["low"].min())
                if is_long:
                    mfes.append((hi - entry) / entry)
                    maes.append((entry - lo) / entry)
                else:
                    mfes.append((entry - lo) / entry)
                    maes.append((hi - entry) / entry)
        if mfes and maes:
            return {"sl_pct": float(np.percentile(maes, 100 - p_sl)),
                    "tp_pct": float(np.percentile(mfes, p_tp)),
                    "method": "percentil_MFE_MAE_aprox_H1", "n_sampled": len(mfes)}
        if h1 is not None and not h1.empty:
            tr = (h1["high"] - h1["low"]) / h1["close"].clip(lower=1e-9)
            atr = float(tr.rolling(14).mean().median())
            return {"sl_pct": atr, "tp_pct": 2 * atr, "method": "ATR_fallback", "n_sampled": 0}
    except Exception:
        pass
    return {"sl_pct": None, "tp_pct": None, "method": "unavailable", "n_sampled": 0}
