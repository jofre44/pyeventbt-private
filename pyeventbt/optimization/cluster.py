"""Clustering DBSCAN/HDBSCAN de parámetros (Etapa 2).

Corrige DMRI: scaler robusto (IQR), eps auto (k-distance p90), representante
MEDOIDE (punto real, no moda que fabrica puntos en valles), singletons
top-decila conservados, opción params_plus_behavior, ensemble3.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("pyeventbt.pipeline")


def _robust_scale(X: np.ndarray) -> tuple[np.ndarray, dict]:
    med = np.median(X, axis=0)
    q75, q25 = np.percentile(X, 75, axis=0), np.percentile(X, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    return (X - med) / iqr, {"median": med, "iqr": iqr}


def build_features(
    df: pd.DataFrame,
    search_space: dict[str, Any],
    feature_set: str = "params_only",
    daily_returns: pd.DataFrame | None = None,
    behavior_weight: float = 0.3,
) -> tuple[np.ndarray, list[str]]:
    cols: list[np.ndarray] = []
    names: list[str] = []
    for name, spec in search_space.items():
        if name not in df.columns:
            raise ValueError(f"parámetro '{name}' del espacio no está en candidatos")
        t = spec["type"]
        if t in ("int", "float"):
            cols.append(df[name].astype(float).to_numpy().reshape(-1, 1))
            names.append(name)
        elif t == "categorical":
            for choice in spec["choices"]:
                cols.append((df[name].astype(str) == str(choice)).to_numpy().reshape(-1, 1))
                names.append(f"{name}={choice}")
        else:
            raise ValueError(f"tipo desconocido: {t}")
    X = np.hstack(cols) if cols else np.zeros((len(df), 0))
    Xs, _ = _robust_scale(X)
    if feature_set == "params_plus_behavior":
        if daily_returns is None:
            raise ValueError("params_plus_behavior exige daily_returns")
        behav = _monthly_behavior(df, daily_returns)  # (n_trials, n_months)
        b, _ = _robust_scale(behav)
        Xs = np.hstack([Xs, behavior_weight * b])
        names += [f"beh_m{i}" for i in range(b.shape[1])]
    return Xs, names


def _monthly_behavior(df: pd.DataFrame, daily_returns: pd.DataFrame) -> np.ndarray:
    dr = daily_returns.copy()
    dr.index = pd.to_datetime(dr.index)
    out = []
    for _, row in df.iterrows():
        col = f"trial_{int(row['trial_number'])}"
        s = dr[col].dropna() if col in dr.columns else pd.Series(dtype=float)
        if s.empty:
            out.append(np.zeros(12))
            continue
        m = (1 + s).resample("ME").prod() - 1
        v = m.to_numpy(dtype=float)[-12:]
        if len(v) < 12:
            v = np.pad(v, (12 - len(v), 0))
        out.append(np.nan_to_num(v, nan=0.0))
    return np.vstack(out)


def auto_eps(X: np.ndarray, min_samples: int = 5, quantile: float = 0.90,
             metric: str = "manhattan") -> tuple[float, np.ndarray]:
    from sklearn.metrics import pairwise_distances

    n = X.shape[0]
    if n < 2:
        return float("nan"), np.array([])
    k = min(min_samples, n - 1)
    D = pairwise_distances(X, metric=metric)
    kth = np.sort(D, axis=1)[:, k]  # distancia al k-ésimo vecino
    return float(np.quantile(kth, quantile)), np.sort(kth)


def round_to_space(params: dict[str, Any], search_space: dict[str, Any]) -> dict[str, Any]:
    import math

    out = {}
    for name, spec in search_space.items():
        v = params[name]
        t = spec["type"]
        if t == "int":
            step = spec.get("step", 1)
            v = int(math.floor((v - spec["low"]) / step + 0.5) * step + spec["low"])
            v = max(spec["low"], min(spec["high"], v))
        elif t == "float":
            step = spec.get("step")
            v = float(v)
            if step:
                v = math.floor((v - spec["low"]) / step + 0.5) * step + spec["low"]
                v = round(v, 10)
            v = max(spec["low"], min(spec["high"], v))
        out[name] = v
    return out


def cluster_params(
    df: pd.DataFrame,
    search_space: dict[str, Any],
    cfg: dict[str, Any],
    daily_returns: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """`df`: trial_number + params + score (sharpe). Devuelve etiquetas,
    representantes (medoides) y diagnósticos."""
    from sklearn.metrics import pairwise_distances

    method = cfg.get("method", "dbscan")
    min_samples = int(cfg.get("min_samples", 5))
    metric = cfg.get("metric", "manhattan")
    X, feat_names = build_features(df, search_space, cfg.get("feature_set", "params_only"),
                                   daily_returns)
    eps_cfg = cfg.get("eps", "auto")
    kdist = None
    tiny = len(df) < 2 or X.shape[0] < 2
    if tiny:
        labels = np.full(len(df), -1)
        eps = float("nan")
    else:
        if eps_cfg == "auto":
            eps, kdist = auto_eps(X, min_samples, float(cfg.get("eps_quantile", 0.90)), metric)
        else:
            eps = float(eps_cfg)

        if method == "dbscan":
            from sklearn.cluster import DBSCAN

            labels = DBSCAN(eps=eps, min_samples=min(min_samples, len(df)),
                            metric=metric).fit_predict(X)
        elif method == "hdbscan":
            try:
                import hdbscan  # type: ignore
            except ImportError as exc:
                raise RuntimeError("HDBSCAN exige `pip install hdbscan`") from exc
            labels = hdbscan.HDBSCAN(min_cluster_size=min_samples, metric=metric).fit_predict(X)
        else:
            raise ValueError(f"método cluster desconocido: {method}")

    D = pairwise_distances(X, metric=metric)
    scores = df["score"].to_numpy(dtype=float) if "score" in df.columns else np.zeros(len(df))
    rep_mode = cfg.get("representative", "medoid")
    keep_single = bool(cfg.get("keep_singleton_outliers", True))
    decile = np.quantile(scores[np.isfinite(scores)], 0.9) if np.isfinite(scores).any() else np.inf

    reps: list[dict[str, Any]] = []
    for lab in sorted(set(labels.tolist())):
        idx = np.where(labels == lab)[0]
        if lab == -1:
            if keep_single:
                for i in idx:
                    if scores[i] >= decile:
                        reps.append(_rep(df, search_space, int(i), "singleton_outlier", scores))
            continue
        members = [int(df.iloc[i]["trial_number"]) for i in idx]
        if rep_mode == "medoid":
            sub = D[np.ix_(idx, idx)]
            med = idx[int(np.argmin(sub.sum(axis=1)))]
            reps.append(_rep(df, search_space, int(med), f"medoid_c{lab}", scores, members))
        elif rep_mode == "best_sharpe":
            best = idx[int(np.nanargmax(scores[idx]))]
            reps.append(_rep(df, search_space, int(best), f"best_c{lab}", scores, members))
        elif rep_mode == "ensemble3":
            order = idx[np.argsort(-np.nan_to_num(scores[idx], nan=-np.inf))][:3]
            for j, oi in enumerate(order):
                reps.append(_rep(df, search_space, int(oi), f"ens{lab}_{j}", scores, members))
        else:
            raise ValueError(f"representante desconocido: {rep_mode}")

    return {
        "labels": labels, "eps": eps, "eps_auto": eps_cfg == "auto",
        "kdist_sorted": kdist, "representatives": reps,
        "n_clusters": len(set(labels.tolist()) - {-1}),
        "n_singletons_kept": sum(1 for r in reps if r["kind"] == "singleton_outlier"),
        "feature_names": feat_names,
    }


def _rep(df, search_space, i, kind, scores, members=None):
    row = df.iloc[i]
    params = round_to_space({k: row[k] for k in search_space}, search_space)
    return {
        "kind": kind, "trial_number": int(row["trial_number"]),
        "score": float(scores[i]), "params": params,
        "members": members or [int(row["trial_number"])],
    }


def save_diagnostics(res: dict[str, Any], X_probe: np.ndarray, out_dir: str | Path) -> None:
    """PNGs: kdistance, sizes_hist, pca2d. No bloquea si falta display (Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if res["kdist_sorted"] is not None:
        plt.figure()
        plt.plot(res["kdist_sorted"])
        plt.axhline(res["eps"], linestyle="--")
        plt.title(f"k-distance (eps_auto={res['eps']:.3f})")
        plt.savefig(out / "kdistance.png", dpi=90)
        plt.close()
    labs = res["labels"]
    uniq, counts = np.unique(labs, return_counts=True)
    plt.figure()
    plt.bar([str(u) for u in uniq], counts)
    plt.title("tamaños de cluster (-1 = ruido)")
    plt.savefig(out / "sizes_hist.png", dpi=90)
    plt.close()
    if X_probe.shape[0] >= 2 and X_probe.shape[1] >= 1:
        from sklearn.decomposition import PCA

        k = min(2, X_probe.shape[1])
        Z = PCA(n_components=k).fit_transform(X_probe)
        plt.figure()
        sc = plt.scatter(Z[:, 0], Z[:, 1] if k > 1 else np.zeros(len(Z)), c=labs)
        plt.legend(*sc.legend_elements(), title="cluster")
        plt.title("PCA-2D de candidatos")
        plt.savefig(out / "pca2d.png", dpi=90)
        plt.close()
