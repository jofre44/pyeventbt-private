"""Informe HTML resumen del pipeline (Global + ficha por TF-instrumento).

Genera ``report.html`` offline y autocontenido en el run_dir:
sin CDN, sin plotly, solo matplotlib SVG inline + PNGs de cluster
embebidos en base64 + JS vanilla para sort/filter y <details> colapsables.

Uso:
    python -m pyeventbt.robustness.report_html RUN_DIR [--topN 15] [--output report.html]

Integración: se invoca automáticamente al final de
``bots_dispatcher_pipeline.run_pipeline`` y ``bots_dispatcher_optuna.main``
(best-effort: nunca rompe el pipeline si falla).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
from pathlib import Path

logger = logging.getLogger("pyeventbt.report_html")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 110, "font.size": 8, "axes.titlesize": 9})


# --------------------------------------------------------------------------
# utilidades
# --------------------------------------------------------------------------

def _esc(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if np.isnan(x) or np.isinf(x):
            return "—"
        return f"{x:.4g}"
    return html.escape(str(x))


def _badge(v: str) -> str:
    v = str(v or "—")
    cls = "b-nd"
    u = v.upper()
    if u in ("PASS", "DONE", "TRUE", "GREEN"):
        cls = "b-pass"
    elif u in ("FAIL", "RED", "ERROR"):
        cls = "b-fail"
    elif u in ("AMBAR", "AMBER", "AWAITING_LOCKBOX", "INCONCLUSIVE", "SKIPPED"):
        cls = "b-ambar"
    return f'<span class="badge {cls}">{html.escape(v)}</span>'


def _fig_to_svg(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    return buf.getvalue().decode("utf-8", errors="replace")


def _png_b64(path: Path) -> str | None:
    try:
        if not path.exists():
            return None
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return None


def _read_yaml(path: Path) -> dict:
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv_rows(path: Path, limit: int = 5000) -> list[dict]:
    try:
        import pandas as pd

        df = pd.read_csv(path, nrows=limit)
        return df.to_dict(orient="records")
    except Exception:
        return []


def _downsample(xs, ys, n: int = 1500):
    try:
        m = len(xs)
        if m <= n:
            return xs, ys
        idx = np.linspace(0, m - 1, n).astype(int)
        return [xs[i] for i in idx], [ys[i] for i in idx]
    except Exception:
        return xs, ys


# --------------------------------------------------------------------------
# recolección
# --------------------------------------------------------------------------

def _discover_combos(run_dir: Path) -> list[tuple[str, str]]:
    combos: set[tuple[str, str]] = set()
    plog = _read_yaml(run_dir / "process_logs.yaml")
    for key in plog.keys():
        try:
            sym, tf, _stage = str(key).split("|")
            combos.add((sym, tf))
        except ValueError:
            continue
    for sub, sep in (("optimization", "_"), ("cluster", "_"),
                     ("decorrelation", "_"), ("wf_cpcv", "_")):
        d = run_dir / sub
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.is_dir() and sep in p.name:
                parts = p.name.rsplit(sep, 1)
                if len(parts) == 2:
                    combos.add((parts[0], parts[1]))
    return sorted(combos)


def _load_trials(run_dir: Path, strategy: str, sym: str, tf: str) -> dict:
    out: dict = {"n": 0, "rows": [], "best": {}}
    base = run_dir / "optimization" / f"{sym}_{tf}"
    tag = f"{strategy}_{sym}_{tf}"
    p = base / f"{tag}_trials.parquet"
    if not p.exists():
        # fallback: cualquier *_trials.parquet en la carpeta
        cands = list(base.glob("*_trials.parquet")) if base.exists() else []
        p = cands[0] if cands else p
    if not p.exists():
        return out
    try:
        import polars as pl

        df = pl.read_parquet(p).to_dict(as_series=False)
        n = len(df.get("trial_number", []))
        out["n"] = n
        vals = np.array(df.get("value", []), dtype=float)
        feas = df.get("feasible", [None] * n)
        rows = []
        for i in range(min(n, 5000)):
            rows.append({
                "trial": int(df["trial_number"][i]),
                "value": float(vals[i]) if np.isfinite(vals[i]) else float("nan"),
                "sharpe": float(df.get("attr_sharpe_net_full", [float("nan")] * n)[i]),
                "pf": float(df.get("attr_pf", [float("nan")] * n)[i]),
                "n_trades": float(df.get("attr_n_trades", [0] * n)[i]),
                "bb_period": df.get("param_bb_period", [None] * n)[i],
                "bb_std": df.get("param_bb_std_dev", [None] * n)[i],
                "feasible": feas[i],
            })
        out["rows"] = rows
        if n:
            bi = int(np.nanargmax(np.where(np.isfinite(vals), vals, -np.inf)))
            out["best"] = rows[bi] if bi < len(rows) else {}
            out["feasible_rate"] = float(np.mean([bool(f) for f in feas])) if feas else 0.0
    except Exception as exc:
        logger.warning("trials %s %s: %s", sym, tf, exc)
    return out


def _load_daily_top(run_dir: Path, strategy: str, sym: str, tf: str,
                    trials: list[int], k: int = 3) -> dict:
    """Curvas acumuladas proxy (cumprod 1+r) para los top-k trials."""
    base = run_dir / "optimization" / f"{sym}_{tf}"
    tag = f"{strategy}_{sym}_{tf}"
    p = base / f"{tag}_daily_returns.parquet"
    if not p.exists():
        cands = list(base.glob("*_daily_returns.parquet")) if base.exists() else []
        p = cands[0] if cands else p
    if not p.exists():
        return {}
    try:
        import polars as pl

        df = pl.read_parquet(p)
        cols = df.columns
        if "date" not in cols:
            return {}
        dates = df["date"].to_list()
        curves = {}
        for t in trials[:k]:
            c = f"trial_{t}"
            if c not in cols:
                continue
            r = np.array(df[c].to_list(), dtype=float)
            r = r[np.isfinite(r)]
            if len(r) < 10:
                continue
            eq = np.cumprod(1.0 + np.clip(r, -0.5, 0.5))
            xs, ys = _downsample(dates[: len(eq)], (eq / eq[0] - 1.0 * 0 + eq).tolist())
            # normalizar a base 10000 como el cash del maestro
            ys = (np.array(ys) / ys[0] * 10000.0).tolist() if ys[0] else ys
            curves[t] = {"dates": [str(d)[:10] for d in xs], "equity": ys}
        return curves
    except Exception as exc:
        logger.warning("daily %s %s: %s", sym, tf, exc)
        return {}


def _combo_data(run_dir: Path, strategy: str, sym: str, tf: str) -> dict:
    tag = f"{strategy}_{sym}_{tf}"
    trials = _load_trials(run_dir, strategy, sym, tf)
    # cluster
    cdir = run_dir / "cluster" / f"{sym}_{tf}"
    creps = _read_csv_rows(cdir / f"{tag}.csv")
    if not creps:
        cands = list(cdir.glob("*.csv")) if cdir.exists() else []
        creps = _read_csv_rows(cands[0]) if cands else []
    neff = _read_json(cdir / "neff.json")
    # synth
    sdir = run_dir / "synthetic_validation"
    synth = _read_csv_rows(sdir / f"{tag}.csv")
    # wfcpcv
    wdir = run_dir / "wf_cpcv" / f"{sym}_{tf}"
    pbo = _read_json(wdir / "pbo.json")
    cpcv = _read_csv_rows(wdir / "cpcv.csv")
    wfsum = _read_json(wdir / "wfcpcv_summary.json")
    wf = _read_csv_rows(wdir / "wf" / "walkforward.csv")
    # select
    seldir = run_dir / "decorrelation" / f"{sym}_{tf}"
    sel = _read_json(seldir / "selection.json")
    gates = _read_csv_rows(seldir / "gates.csv")
    # curvas equity proxy para top trials (best + seleccionados)
    top_trials: list[int] = []
    if trials.get("best"):
        top_trials.append(int(trials["best"].get("trial", 0)))
    for g in gates[:3]:
        try:
            top_trials.append(int(g.get("trial_number")))
        except Exception:
            pass
    top_trials = list(dict.fromkeys(top_trials))[:3]
    curves = _load_daily_top(run_dir, strategy, sym, tf, top_trials or [0])
    return {
        "sym": sym, "tf": tf, "tag": tag,
        "trials": trials, "cluster_reps": creps, "neff": neff,
        "synth": synth, "pbo": pbo, "cpcv": cpcv,
        "wfsum": wfsum, "wf": wf, "selection": sel, "gates": gates,
        "curves": curves,
    }


def collect_run(run_dir: Path) -> dict:
    master: dict = _read_yaml(run_dir / "master_used.yaml")
    strategy = str(master.get("strategy", {}).get("name", "na"))
    plog = _read_yaml(run_dir / "process_logs.yaml")
    combos = _discover_combos(run_dir)
    per = [_combo_data(run_dir, strategy, s, t) for s, t in combos]
    # ranking: ordenar por best value desc
    per.sort(key=lambda d: float(d["trials"].get("best", {}).get("value", -1e18) or -1e18),
             reverse=True)
    return {
        "run_dir": run_dir, "master": master, "strategy": strategy,
        "plog": plog, "combos": per,
        "hash": (run_dir / "master_hash.txt").read_text(encoding="utf-8").split()
               if (run_dir / "master_hash.txt").exists() else [],
        "promoted": _read_csv_rows(run_dir / "promoted.csv"),
        "budget": _read_json(run_dir / "00_prep" / "budget.json"),
        "power": _read_json(run_dir / "00_prep" / "power.json"),
    }


# --------------------------------------------------------------------------
# figuras
# --------------------------------------------------------------------------

def _fig_hist(values: list[float], title: str, xlabel: str) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v):
        ax.hist(v, bins=min(40, max(10, len(v) // 10)))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("n")
    return _fig_to_svg(fig)


def _fig_bars(labels: list[str], values: list[float], title: str,
              ylabel: str, hline: float | None = None) -> str:
    fig, ax = plt.subplots(figsize=(max(5.2, 0.5 * len(labels)), 2.8))
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if hline is not None:
        ax.axhline(hline, color="red", linestyle="--", linewidth=1)
    return _fig_to_svg(fig)


def _fig_optuna_history(rows: list[dict], title: str) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    if rows:
        xs = [r["trial"] for r in rows]
        ys = [r["value"] for r in rows if np.isfinite(r["value"])]
        ax.scatter(xs[: len(ys)], ys, s=8)
        # mejor acumulado
        best = np.maximum.accumulate(np.where(np.isfinite([r["value"] for r in rows]),
                                              [r["value"] for r in rows], -np.inf))
        ax.plot(xs, best, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("trial")
    ax.set_ylabel("value")
    return _fig_to_svg(fig)


def _fig_param_scatter(rows: list[dict], title: str) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    xs = [r["bb_period"] for r in rows if r["bb_period"] is not None]
    ys = [r["bb_std"] for r in rows if r["bb_std"] is not None]
    vs = [r["value"] for r in rows if r["bb_period"] is not None]
    if xs and ys:
        sc = ax.scatter(xs, ys, c=vs, s=10, cmap="RdYlGn")
        fig.colorbar(sc, ax=ax, label="value")
    ax.set_title(title)
    ax.set_xlabel("bb_period")
    ax.set_ylabel("bb_std_dev")
    return _fig_to_svg(fig)


def _fig_equity(curves: dict, title: str) -> str:
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for t, c in curves.items():
        ax.plot(c["dates"], c["equity"], linewidth=1, label=f"t{t}")
    ax.set_title(title)
    ax.set_xlabel("fecha")
    ax.set_ylabel("equity proxy (base 10000)")
    if curves:
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", labelsize=6)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(25)
            lbl.set_ha("right")
        # reducir ticks
        ax.set_xticks(ax.get_xticks()[:: max(1, len(ax.get_xticks()) // 6)])
    else:
        ax.text(0.5, 0.5, "sin daily_returns", ha="center", transform=ax.transAxes)
    return _fig_to_svg(fig)


def _fig_wf(wf: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    if wf:
        lab = [str(r.get("fold", i)) for i, r in enumerate(wf)]
        tr = [float(r.get("sharpe_train", np.nan)) for r in wf]
        te = [float(r.get("sharpe_test", np.nan)) for r in wf]
        x = np.arange(len(lab))
        ax.bar(x - 0.2, tr, 0.4, label="train")
        ax.bar(x + 0.2, te, 0.4, label="test")
        ax.set_xticks(x)
        ax.set_xticklabels(lab)
        ax.legend(fontsize=7)
    ax.set_title("WalkForward train vs test (sharpe)")
    return _fig_to_svg(fig)


# --------------------------------------------------------------------------
# html
# --------------------------------------------------------------------------

CSS = """
body{font-family:Arial,Helvetica,sans-serif;margin:20px;background:#f6f7f9;color:#1c1c1c}
h1{font-size:24px}h2{margin-top:28px;border-bottom:2px solid #ddd;padding-bottom:4px}
.card{background:#fff;border:1px solid #ddd;padding:12px;margin-bottom:12px;border-radius:6px}
table{border-collapse:collapse;width:100%;background:#fff;font-size:12px}
th,td{border:1px solid #ddd;padding:5px 8px;text-align:left}
th{background:#eef1f5;cursor:pointer}tr:nth-child(even){background:#f7f9fc}
.badge{padding:2px 8px;border-radius:10px;font-size:11px;font-weight:bold;color:#fff}
.b-pass{background:#2e7d32}.b-fail{background:#c62828}.b-ambar{background:#ef6c00}.b-nd{background:#616161}
details{margin:10px 0;border:1px solid #ddd;background:#fff;border-radius:6px;padding:8px}
summary{font-weight:bold;cursor:pointer}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
svg{max-width:100%;height:auto}
.note{color:#555;font-size:12px}.mono{font-family:monospace;font-size:11px}
img.diag{max-width:32%;margin:4px;border:1px solid #ddd}
"""

JS = """
function sortTable(id,col){var t=document.getElementById(id),r=t.rows,i,n=col,dir=t.getAttribute('data-dir')==='a'?'d':'a';t.setAttribute('data-dir',dir);var rows=Array.prototype.slice.call(r,1);rows.sort(function(a,b){var x=a.cells[n].innerText,y=b.cells[n].innerText;var xn=parseFloat(x),yn=parseFloat(y);if(!isNaN(xn)&&!isNaN(yn)){x=xn;y=yn}return dir==='a'?(x>y?1:-1):(x<y?1:-1)});rows.forEach(function(x){t.appendChild(x)})}
function filt(id,q){var t=document.getElementById(id);q=q.toLowerCase();for(var i=1;i<t.rows.length;i++){var r=t.rows[i];r.style.display=r.innerText.toLowerCase().includes(q)?'':'none'}}
"""


def _kv_table(d: dict) -> str:
    rows = "".join(f"<tr><td>{_esc(k)}</td><td class='mono'>{_esc(v)}</td></tr>"
                   for k, v in d.items())
    return f"<table>{rows}</table>"


def _rows_table(headers: list[str], rows: list[list], tid: str) -> str:
    th = "".join(f"<th onclick=\"sortTable('{tid}',{i})\">{_esc(h)}</th>"
                 for i, h in enumerate(headers))
    td = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f"<input placeholder='filtrar...' oninput=\"filt('{tid}',this.value)\"/>"
            f"<table id='{tid}' data-dir='a'><tr>{th}</tr>{td}</table>")


def build_html_report(run_dir: str | Path, top_n: int = 15,
                      output: str = "report.html") -> Path:
    run_dir = Path(run_dir)
    data = collect_run(run_dir)
    master = data["master"]
    combos = data["combos"]
    uni = master.get("universe", {})
    opt = master.get("optimization", {})
    costs = master.get("costs", {})

    # ---- globales ----
    all_vals = [r["value"] for c in combos for r in c["trials"]["rows"]]
    best_sh = [c["trials"].get("best", {}).get("sharpe", np.nan) for c in combos]
    labels = [f"{c['sym']}_{c['tf']}" for c in combos]
    all_ranks = [float(r.get("rank_sharpe")) for c in combos for r in c["synth"]
                 if r.get("rank_sharpe") not in (None, "")]
    pbo_vals = [float(c["pbo"].get("pbo", np.nan)) if c["pbo"].get("pbo") is not None else np.nan
                for c in combos]
    cpcv_med = []
    for c in combos:
        meds = [float(r.get("median", np.nan)) for r in c["cpcv"]
                if r.get("median") not in (None, "")]
        cpcv_med.append(float(np.nanmedian(meds)) if meds else np.nan)

    g_hist = _fig_hist(all_vals, "Optuna — distribución value (todos los combos)", "value")
    g_best = _fig_bars(labels or ["—"],
                       [v if np.isfinite(v) else 0 for v in (best_sh or [0])],
                       "Mejor sharpe_net por combo", "sharpe", hline=0)
    g_rank = _fig_hist(all_ranks or [0], "Synth — hist rank_sharpe [10,90]", "rank")
    g_pbo = _fig_bars(labels or ["—"],
                      [v if np.isfinite(v) else 0 for v in (pbo_vals or [0])],
                      "PBO por combo (umbral 0.30)", "PBO", hline=0.30)

    # funnel por etapa desde process_logs
    funnel: dict[str, dict[str, int]] = {}
    for k, v in data["plog"].items():
        try:
            _, _, st = str(k).split("|")
        except ValueError:
            continue
        funnel.setdefault(st, {}).setdefault(str(v.get("status", "?")), 0)
        funnel[st][str(v.get("status", "?"))] += 1

    # ranking global
    rank_headers = ["combo", "best_t", "value", "sharpe", "PF", "n_tr",
                    "N_eff", "PBO", "WFE", "sel", "gates PASS", "forward"]
    rank_rows = []
    for c in combos:
        b = c["trials"].get("best", {})
        gates_pass = sum(1 for g in c["gates"] if str(g.get("gates")) == "PASS")
        fwd = ",".join(sorted({str(g.get("forward", "—")) for g in c["gates"]})) if c["gates"] else "—"
        rank_rows.append([
            f"{c['sym']}_{c['tf']}",
            _esc(b.get("trial", "—")), _esc(b.get("value", "—")),
            _esc(b.get("sharpe", "—")), _esc(b.get("pf", "—")),
            _esc(b.get("n_trades", "—")),
            _esc(c["neff"].get("N_eff", "—")),
            _esc(c["pbo"].get("pbo", "—")),
            _esc(c["wfsum"].get("wfe", "—")),
            _esc(len(c["selection"].get("selected", []))),
            _esc(gates_pass),
            _badge(fwd),
        ])

    # ---- fichas ----
    fichas = []
    for i, c in enumerate(combos):
        open_attr = " open" if i < top_n else ""
        b = c["trials"].get("best", {})
        title = (f"{c['sym']} {c['tf']} — best t{b.get('trial','—')} "
                 f"value={_esc(b.get('value','—'))} sharpe={_esc(b.get('sharpe','—'))} "
                 f"PBO={_esc(c['pbo'].get('pbo','—'))}")
        f_hist = _fig_optuna_history(c["trials"]["rows"], f"Optuna history {c['tag']}")
        f_par = _fig_param_scatter(c["trials"]["rows"], f"Espacio params {c['tag']}")
        f_eq = _fig_equity(c["curves"], f"Equity proxy top trials {c['tag']}")
        f_wf = _fig_wf(c["wf"])
        # synth ranks bars
        sr = [(str(r.get("set_id", "")).split("#t")[-1], float(r.get("rank_sharpe", np.nan)))
              for r in c["synth"]]
        sr = [(a, v) for a, v in sr if np.isfinite(v)]
        f_synth = _fig_bars([a for a, _ in sr] or ["—"], [v for _, v in sr] or [0],
                            "Synth rank_sharpe por set", "rank", hline=90) if sr else "<p class=note>sin synth</p>"
        cm = [(str(r.get("set_id", "")).split("#t")[-1], float(r.get("median", np.nan)))
              for r in c["cpcv"]]
        cm = [(a, v) for a, v in cm if np.isfinite(v)]
        f_cpcv = _fig_bars([a for a, _ in cm] or ["—"], [v for _, v in cm] or [0],
                            "CPCV mediana por set", "mediana", hline=0) if cm else "<p class=note>sin cpcv</p>"
        # diagnósticos cluster embebidos
        diag_html = ""
        for name in ("pca2d.png", "kdistance.png", "sizes_hist.png"):
            b64 = _png_b64(run_dir / "cluster" / f"{c['sym']}_{c['tf']}" / "diagnostics" / name)
            if b64:
                diag_html += (f"<img class='diag' src='data:image/png;base64,{b64}' title='{name}'/>")
        # tablas
        gates_h = ["set_id", "t", "sharpe", "pf", "n_tr", "jitter", "stress_pf", "gates", "forward"]
        gates_r = [[f"<span class=mono>{_esc(g.get('set_id'))}</span>",
                    _esc(g.get("trial_number")), _esc(g.get("sharpe_net")),
                    _esc(g.get("pf")), _esc(g.get("n_trades")),
                    _badge(g.get("jitter")), _esc(g.get("worst_cell_pf") or g.get("stress_pf")),
                    _badge(g.get("gates")), _badge(g.get("forward"))]
                   for g in c["gates"]]
        gates_tab = _rows_table(gates_h, gates_r, f"gates_{c['sym']}_{c['tf']}") if gates_r else "<p class=note>sin gates</p>"
        sel = c["selection"]
        fichas.append(f"""
<details{open_attr}><summary>{_esc(title)}{' · TOP' if i < top_n else ''}</summary>
<div class="card"><b>Cluster:</b> reps={_esc(len(c['cluster_reps']))} N_eff={_esc(c['neff'].get('N_eff','—'))}
mean_abs_corr={_esc(c['neff'].get('mean_abs_corr','—'))} · <b>Selección:</b> {_esc(sel.get('selected',[]))}
· <b>WFE:</b> {_esc(c['wfsum'].get('wfe','—'))} ({_esc(c['wfsum'].get('wf_verdict','—'))})</div>
<div class="grid"><div class="card">{f_hist}</div><div class="card">{f_par}</div></div>
<div class="card">{f_eq}</div>
<div class="grid"><div class="card">{f_synth}</div><div class="card">{f_cpcv}</div></div>
<div class="card">{f_wf}</div>
<div class="card"><h3>Diagnóstico cluster</h3>{diag_html or '<p class=note>sin PNGs</p>'}</div>
<div class="card"><h3>Puertas → lockbox → sizing</h3>{gates_tab}</div>
</details>""")

    meta = {
        "run": run_dir.name, "strategy": data["strategy"],
        "master_hash": " ".join(data["hash"]),
        "seed": master.get("seed"),
        "IS": uni.get("windows", {}).get("IS"), "FORWARD": uni.get("windows", {}).get("FORWARD"),
        "embargo": uni.get("embargo"), "costs": costs,
        "n_trials_cfg": opt.get("n_trials"),
        "combos_detectados": len(combos),
        "promoted_rows": len(data["promoted"]),
    }

    html_doc = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"/>
<title>PyEventBT pipeline — {html.escape(run_dir.name)}</title>
<style>{CSS}</style></head><body>
<h1>PyEventBT pipeline — informe resumen</h1>
<p class="note">Offline autocontenido (SVG inline, sin CDN). Puertas después del lockbox = run inválido.
Rank bajo=FAIL / alto=ÁMBAR (lo resuelve edge+3B). Veto DSR(N_eff)&lt;0.20.</p>
<h2>0 · Trazabilidad</h2><div class="card">{_kv_table(meta)}</div>
<div class="card"><h3>Funnel por etapa (process_logs.yaml)</h3>{_kv_table({k: v for k, v in funnel.items()})}</div>
<div class="card"><h3>PREP</h3>{_kv_table({"budget": data['budget'], "power": data['power']})}</div>
<h2>1 · Global — optimización y robustez</h2>
<div class="grid"><div class="card">{g_hist}</div><div class="card">{g_best}</div></div>
<div class="grid"><div class="card">{g_rank}</div><div class="card">{g_pbo}</div></div>
<h2>2 · Ranking por TF-instrumento (ranking + puertas)</h2>
<div class="card">{_rows_table(rank_headers, rank_rows, 'rank')}</div>
<h2>3 · Fichas por combo (Top-{top_n} abiertas)</h2>
{''.join(fichas) if fichas else '<p>sin combos</p>'}
<h2>4 · Promocionados</h2>
<div class="card">{html.escape(str(data['promoted'][:5])) if data['promoted'] else 'sin promoted.csv (ningún PASS lockbox o falta --auto-lockbox)'}</div>
<script>{JS}</script></body></html>"""
    out = run_dir / output
    out.write_text(html_doc, encoding="utf-8")
    logger.warning("report_html %s (%d combos, %.1f KB)", out, len(combos), len(html_doc) / 1024)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Informe HTML offline del pipeline")
    ap.add_argument("run_dir")
    ap.add_argument("--topN", type=int, default=15)
    ap.add_argument("--output", default="report.html")
    args = ap.parse_args()
    out = build_html_report(args.run_dir, top_n=args.topN, output=args.output)
    print(str(out))


if __name__ == "__main__":
    main()
