"""Reconciliación e idempotencia (Etapa 6.5): cada 60s interno vs broker.

Discrepancia ⇒ halt_and_alert. Órdenes idempotentes (magic + comentario único;
verificar antes de reenviar tras timeout). Estado persistido para reinicios.
"""
from __future__ import annotations

import json
from pathlib import Path


def order_key(magic: int, comment: str) -> str:
    return f"{magic}:{comment}"


class OrderBook:
    """Registro de órdenes enviadas: reintento tras timeout NO duplica."""

    def __init__(self, state_path: str | Path | None = None):
        self.path = Path(state_path) if state_path else None
        self.sent: dict[str, dict] = {}
        if self.path and self.path.exists():
            self.sent = json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.sent), encoding="utf-8")

    def already_sent(self, magic: int, comment: str) -> bool:
        return order_key(magic, comment) in self.sent

    def record(self, magic: int, comment: str, ticket: str) -> None:
        self.sent[order_key(magic, comment)] = {"ticket": ticket}
        self._save()

    def send_once(self, magic: int, comment: str, send_fn) -> dict:
        """Idempotente: si ya existe ticket, no reenvía."""
        if self.already_sent(magic, comment):
            return {"status": "duplicate_suppressed", **self.sent[order_key(magic, comment)]}
        ticket = send_fn()
        self.record(magic, comment, ticket)
        return {"status": "sent", "ticket": ticket}


def reconcile(internal: dict, broker: dict) -> dict:
    """Compara posiciones/órdenes/equity. Mismatch ⇒ halt_and_alert."""
    mism = []
    for k in ("positions", "orders", "equity"):
        a, b = internal.get(k), broker.get(k)
        if isinstance(a, float) or isinstance(b, float):
            if abs(float(a or 0) - float(b or 0)) > 1e-6:
                mism.append(k)
        elif a != b:
            mism.append(k)
    if mism:
        return {"action": "halt_and_alert", "mismatches": mism}
    return {"action": "ok", "mismatches": []}
