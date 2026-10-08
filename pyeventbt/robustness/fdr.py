"""FDR Benjamini-Hochberg sobre p del edge test (Etapa 3.2e, NUNCA sobre ranks).

Con 50 sims un p de rank ronda 1/51 ≈ 0.02 y con C=250 BH exigiría 0.0004:
inalcanzable (el FDR de v1 era decorativo). Check de resolución: el p mínimo
alcanzable del método (1/(B+1)) debe ser ≤ q/C; si no, ABORTA con aviso.
"""
from __future__ import annotations


def bh_fdr(p_values: dict[str, float], q: float = 0.10) -> dict:
    names = sorted(p_values, key=lambda k: p_values[k])
    m = len(names)
    adj: dict[str, float] = {}
    rejected: list[str] = []
    if m == 0:
        return {"adj_p": {}, "rejected": [], "m": 0}
    prev = 1.0
    for i in reversed(range(m)):
        p = p_values[names[i]]
        a = min(p * m / (i + 1), prev)
        prev = a
        adj[names[i]] = a
    for i, name in enumerate(names):
        if p_values[name] <= (i + 1) / m * q:
            rejected.append(name)
    return {"adj_p": adj, "rejected": rejected, "m": m, "q": q}


def resolution_check(n_resamples: int, q: float, C: int) -> dict:
    """p_min alcanzable = 1/(B+1). Exige p_min ≤ q/C o ABORTA."""
    p_min = 1.0 / (n_resamples + 1)
    need = q / max(C, 1)
    ok = p_min <= need
    return {"verdict": "PASS" if ok else "ABORT", "p_min": p_min,
            "required": need, "n_resamples": n_resamples}


def assert_resolution(n_resamples: int, q: float, C: int) -> None:
    r = resolution_check(n_resamples, q, C)
    if r["verdict"] == "ABORT":
        raise RuntimeError(
            f"FDR sin resolución: p_min={r['p_min']:.5f} > q/C={r['required']:.5f}. "
            f"Sube n_resamples (B) o reduce C (nº de sets)."
        )
