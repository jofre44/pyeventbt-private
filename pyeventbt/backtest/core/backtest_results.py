"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

import html
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

_TRADING_DAYS_YEAR = 252


class BacktestResults:

    def __init__(
        self,
        backtest_pnl: pd.DataFrame,
        trades: pd.DataFrame,
        initial_capital: float | None = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = _TRADING_DAYS_YEAR,
    ) -> None:
        self._backtest_pnl = backtest_pnl.copy() if backtest_pnl is not None else pd.DataFrame()
        self._pnl = self._backtest_pnl.astype(float) if not self._backtest_pnl.empty else self._backtest_pnl
        if not self._pnl.empty and not isinstance(self._pnl.index, pd.DatetimeIndex):
            try:
                self._pnl.index = pd.to_datetime(self._pnl.index)
            except Exception:
                pass
        self._trades = trades.copy() if trades is not None else pd.DataFrame()
        self._risk_free_rate = float(risk_free_rate or 0.0)
        self._periods_per_year = int(periods_per_year or _TRADING_DAYS_YEAR)
        self._initial_capital = (
            float(initial_capital)
            if initial_capital is not None
            else (float(self._pnl["EQUITY"].iloc[0]) if not self._pnl.empty and "EQUITY" in self._pnl else 0.0)
        )
        self._returns = self._compute_returns()
        self._closed_cache: pd.DataFrame | None = None

    # ------------------------------------------------------------------ props
    @property
    def pnl(self):
        return self._pnl

    @property
    def returns(self):
        return self._returns

    @property
    def trades(self):
        return self._trades

    @property
    def backtest_pnl(self):
        return self._backtest_pnl

    @property
    def closed_trades(self) -> pd.DataFrame:
        if self._closed_cache is None:
            self._closed_cache = self._build_closed_trades()
        return self._closed_cache.copy()

    # ------------------------------------------------------- internal helpers
    def _compute_returns(self) -> pd.Series:
        if self._pnl.empty or "EQUITY" not in self._pnl:
            return pd.Series(dtype=float)
        eq = self._pnl["EQUITY"].astype(float)
        prev = eq.shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.where((prev > 0) & np.isfinite(prev), eq / prev - 1.0, np.nan)
        return pd.Series(rets, index=self._pnl.index, name="RETURNS").replace([np.inf, -np.inf], np.nan)

    def _daily_equity(self) -> pd.Series:
        if self._pnl.empty or "EQUITY" not in self._pnl:
            return pd.Series(dtype=float)
        eq = self._pnl["EQUITY"].astype(float).sort_index()
        if isinstance(eq.index, pd.DatetimeIndex) and len(eq) > 1:
            return eq.resample("1D").last().dropna()
        return eq

    def _daily_returns(self) -> pd.Series:
        eq = self._daily_equity()
        if len(eq) < 2:
            return pd.Series(dtype=float)
        prev = eq.shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where((prev > 0) & np.isfinite(prev), eq / prev - 1.0, np.nan)
        return pd.Series(r, index=eq.index).replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def _drawdown_from_equity(eq: pd.Series) -> tuple[pd.Series, pd.Series]:
        eq = eq.astype(float).sort_index()
        roll_max = eq.cummax()
        with np.errstate(divide="ignore", invalid="ignore"):
            dd_pct = np.where(roll_max > 0, eq / roll_max - 1.0, np.nan)
        dd_pct = pd.Series(dd_pct, index=eq.index).fillna(0.0)
        dd_abs = eq - roll_max
        return dd_pct, dd_abs

    def _build_closed_trades(self) -> pd.DataFrame:
        if self._trades is None or self._trades.empty:
            return pd.DataFrame()
        df = self._trades.copy()
        if "DEAL" not in df.columns:
            return pd.DataFrame()
        outs = df[df["DEAL"].astype(str).str.upper() == "OUT"].copy()
        if outs.empty:
            return pd.DataFrame()
        ins = df[df["DEAL"].astype(str).str.upper() == "IN"].copy()

        def _num(s: pd.Series) -> pd.Series:
            return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)

        for c in ("GROSS_PROFIT", "COMMISSION", "SWAP", "FEE", "VOLUME", "PRICE"):
            if c in outs.columns:
                outs[c] = _num(outs[c])
        if not ins.empty:
            for c in ("COMMISSION", "SWAP", "FEE", "PRICE", "VOLUME"):
                if c in ins.columns:
                    ins[c] = _num(ins[c])
            ins["_OT"] = pd.to_datetime(ins["TIME_GENERATED"], errors="coerce", utc=True)
            in_by_pos = ins.groupby("POSITION_ID") if "POSITION_ID" in ins.columns else None
        else:
            in_by_pos = None

        outs["_CT"] = pd.to_datetime(outs["TIME_GENERATED"], errors="coerce", utc=True)
        rows = []
        for idx, o in outs.iterrows():
            pid = o.get("POSITION_ID")
            o_comm = float(o.get("COMMISSION", 0.0) or 0.0)
            o_swap = float(o.get("SWAP", 0.0) or 0.0)
            o_fee = float(o.get("FEE", 0.0) or 0.0)
            gross = float(o.get("GROSS_PROFIT", 0.0) or 0.0)
            i_comm, i_swap, i_fee = 0.0, 0.0, 0.0
            open_t, entry_px = pd.NaT, np.nan
            if in_by_pos is not None and pid in in_by_pos.groups:
                g = in_by_pos.get_group(pid).sort_values("_OT")
                first = g.iloc[0]
                i_comm = float(first.get("COMMISSION", 0.0) or 0.0)
                i_swap = float(first.get("SWAP", 0.0) or 0.0)
                i_fee = float(first.get("FEE", 0.0) or 0.0)
                open_t = first["_OT"]
                try:
                    entry_px = float(first.get("PRICE", np.nan))
                except Exception:
                    entry_px = np.nan
            # NET: commission/fee son costes positivos; swap viene con signo MT5 (negativo = coste)
            net = gross - (o_comm + i_comm) + (o_swap + i_swap) - (o_fee + i_fee)
            close_t = o["_CT"]
            hold_min = (close_t - open_t).total_seconds() / 60.0 if pd.notna(close_t) and pd.notna(open_t) else np.nan
            rows.append(
                {
                    "POSITION_ID": pid,
                    "SYMBOL": o.get("SYMBOL"),
                    "DIRECTION": o.get("SIGNAL_TYPE"),
                    "OPEN_TIME": open_t,
                    "CLOSE_TIME": close_t,
                    "HOLDING_MIN": hold_min,
                    "VOLUME": float(o.get("VOLUME", 0.0) or 0.0),
                    "ENTRY_PRICE": entry_px,
                    "EXIT_PRICE": float(o.get("PRICE", np.nan) or np.nan),
                    "GROSS": gross,
                    "COMMISSION": o_comm + i_comm,
                    "SWAP": o_swap + i_swap,
                    "FEE": o_fee + i_fee,
                    "NET": net,
                }
            )
        closed = pd.DataFrame(rows).sort_values("CLOSE_TIME").reset_index(drop=True)
        return closed

    # ---------------------------------------------------------------- metrics
    def stats(self) -> dict:
        out: dict = {}
        if self._pnl.empty or "EQUITY" not in self._pnl:
            return out
        eq = self._pnl["EQUITY"].astype(float).sort_index()
        bal = self._pnl["BALANCE"].astype(float).sort_index() if "BALANCE" in self._pnl else eq
        initial = float(self._initial_capital or eq.iloc[0])
        final_eq, final_bal = float(eq.iloc[-1]), float(bal.iloc[-1])
        start, end = eq.index[0], eq.index[-1]
        try:
            days = max((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 86400.0, 1 / 1440)
        except Exception:
            days = max(len(eq) / 1440.0, 1 / 1440)
        years = days / 365.25

        total_ret = (final_eq / initial - 1.0) if initial > 0 else np.nan
        cagr = (final_eq / initial) ** (1 / years) - 1.0 if initial > 0 and years > 0 and final_eq > 0 else np.nan

        dret = self._daily_returns()
        ppy = self._periods_per_year
        rf_d = self._risk_free_rate / ppy
        vol = float(dret.std(ddof=1) * np.sqrt(ppy)) if len(dret) > 1 else np.nan
        sharpe = float((dret - rf_d).mean() / dret.std(ddof=1) * np.sqrt(ppy)) if len(dret) > 1 and dret.std(ddof=1) > 0 else np.nan
        downside = dret[dret < 0]
        sortino = (
            float((dret - rf_d).mean() / downside.std(ddof=1) * np.sqrt(ppy))
            if len(downside) > 1 and downside.std(ddof=1) > 0
            else np.nan
        )

        dd_pct, dd_abs = self._drawdown_from_equity(eq)
        max_dd_pct = float(dd_pct.min())
        max_dd_abs = float(dd_abs.min())
        trough = dd_pct.idxmin()
        peak_val = float(eq.loc[:trough].max())
        peak_candidates = eq.loc[:trough][eq.loc[:trough] == peak_val]
        peak = peak_candidates.index[-1]
        after = eq.loc[trough:]
        rec = after[after >= peak_val]
        recovered = not rec.empty
        recovery_time = rec.index[0] if recovered else pd.NaT
        max_dd_days = (
            (pd.Timestamp(recovery_time) - pd.Timestamp(peak)).total_seconds() / 86400.0
            if recovered
            else (pd.Timestamp(end) - pd.Timestamp(peak)).total_seconds() / 86400.0
        )
        current_dd = float(dd_pct.iloc[-1])
        ulcer = float(np.sqrt(np.mean(dd_pct.values**2)) * 100.0)
        calmar = float((cagr / abs(max_dd_pct))) if np.isfinite(cagr) and max_dd_pct < 0 else np.nan

        if len(dret):
            var95 = float(-np.quantile(dret.values, 0.05) * 100.0)
            tail = dret.values[dret.values <= np.quantile(dret.values, 0.05)]
            cvar95 = float(-np.mean(tail) * 100.0) if len(tail) else np.nan
            skew = float(pd.Series(dret).skew())
            kurt = float(pd.Series(dret).kurt())
            best_day = float(dret.max() * 100.0)
            worst_day = float(dret.min() * 100.0)
        else:
            var95 = cvar95 = skew = kurt = best_day = worst_day = np.nan

        try:
            monthly = eq.resample("ME").last().pct_change().dropna() * 100.0
            monthly = monthly.replace([np.inf, -np.inf], np.nan).dropna()
            best_m, worst_m = (float(monthly.max()), float(monthly.min())) if len(monthly) else (np.nan, np.nan)
            pct_pos_m = float((monthly > 0).mean() * 100.0) if len(monthly) else np.nan
        except Exception:
            best_m = worst_m = pct_pos_m = np.nan

        # % tiempo en mercado (proxy): equity != balance
        try:
            exposure = float(((eq - bal).abs() > 1e-9).mean() * 100.0)
        except Exception:
            exposure = np.nan

        out.update(
            {
                "start": str(pd.Timestamp(start)),
                "end": str(pd.Timestamp(end)),
                "duration_days": round(float(days), 2),
                "n_bars": int(len(eq)),
                "initial_capital": round(initial, 2),
                "final_equity": round(final_eq, 2),
                "final_balance": round(final_bal, 2),
                "total_return_pct": round(float(total_ret * 100.0), 2) if np.isfinite(total_ret) else None,
                "cagr_pct": round(float(cagr * 100.0), 2) if np.isfinite(cagr) else None,
                "volatility_ann_pct": round(vol * 100.0, 2) if np.isfinite(vol) else None,
                "sharpe": round(sharpe, 3) if np.isfinite(sharpe) else None,
                "sortino": round(sortino, 3) if np.isfinite(sortino) else None,
                "calmar": round(calmar, 3) if np.isfinite(calmar) else None,
                "max_drawdown_pct": round(max_dd_pct * 100.0, 2),
                "max_drawdown_abs": round(max_dd_abs, 2),
                "max_dd_duration_days": round(float(max_dd_days), 2),
                "max_dd_recovered": bool(recovered),
                "current_drawdown_pct": round(current_dd * 100.0, 2),
                "ulcer_index_pct": round(ulcer, 2) if np.isfinite(ulcer) else None,
                "var95_daily_pct": round(var95, 3) if np.isfinite(var95) else None,
                "cvar95_daily_pct": round(cvar95, 3) if np.isfinite(cvar95) else None,
                "skew_daily": round(skew, 3) if np.isfinite(skew) else None,
                "kurtosis_daily": round(kurt, 3) if np.isfinite(kurt) else None,
                "best_day_pct": round(best_day, 2) if np.isfinite(best_day) else None,
                "worst_day_pct": round(worst_day, 2) if np.isfinite(worst_day) else None,
                "best_month_pct": round(best_m, 2) if best_m is not None and np.isfinite(best_m) else None,
                "worst_month_pct": round(worst_m, 2) if worst_m is not None and np.isfinite(worst_m) else None,
                "pct_positive_months": round(pct_pos_m, 1) if pct_pos_m is not None and np.isfinite(pct_pos_m) else None,
                "exposure_time_in_market_pct": round(exposure, 1) if np.isfinite(exposure) else None,
                "risk_free_rate_ann": self._risk_free_rate,
            }
        )
        return out

    def trade_stats(self) -> dict:
        c = self.closed_trades
        if c.empty:
            return {"n_trades": 0}
        net = c["NET"].astype(float)
        gross = c["GROSS"].astype(float)
        wins = net[net > 0]
        losses = net[net < 0]
        scratch = net[net == 0]
        n = len(net)
        win_rate = len(wins) / n * 100.0 if n else np.nan
        gross_won = float(wins.sum())
        gross_lost = float(-losses.sum())
        profit_factor = (gross_won / gross_lost) if gross_lost > 0 else (np.inf if gross_won > 0 else np.nan)
        expectancy = float(net.mean())
        avg_win = float(wins.mean()) if len(wins) else np.nan
        avg_loss = float(losses.mean()) if len(losses) else np.nan
        payoff = (abs(avg_win / avg_loss)) if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss != 0 else np.nan

        # rachas
        signs = np.sign(net.values)
        max_w = max_l = cur_w = cur_l = 0
        for s in signs:
            if s > 0:
                cur_w += 1
                cur_l = 0
            elif s < 0:
                cur_l += 1
                cur_w = 0
            else:
                cur_w = cur_l = 0
            max_w, max_l = max(max_w, cur_w), max(max_l, cur_l)

        costs = float(c["COMMISSION"].sum() + c["FEE"].sum())
        swaps = float(c["SWAP"].sum())
        cost_drag = (costs / gross_won * 100.0) if gross_won > 0 else np.nan
        hold_median = float(c["HOLDING_MIN"].median()) if c["HOLDING_MIN"].notna().any() else np.nan

        # R-multiple aprox: NET / |avg_loss| (si hay pérdidas)
        r_mult = float(net.mean() / abs(avg_loss)) if np.isfinite(avg_loss) and avg_loss != 0 else np.nan

        by_symbol = c.groupby("SYMBOL")["NET"].agg(["count", "sum"]).rename(columns={"count": "n", "sum": "net"})
        ct = pd.to_datetime(c["CLOSE_TIME"], errors="coerce", utc=True)
        by_dow = c.assign(_dow=ct.dt.day_name()).groupby("_dow")["NET"].sum().to_dict() if ct.notna().any() else {}
        by_hour = c.assign(_h=ct.dt.hour).groupby("_h")["NET"].sum().sort_index().to_dict() if ct.notna().any() else {}

        def _r(x):
            return round(float(x), 2) if x is not None and np.isfinite(x) else None

        return {
            "n_trades": int(n),
            "n_win": int(len(wins)),
            "n_loss": int(len(losses)),
            "n_scratch": int(len(scratch)),
            "win_rate_pct": round(float(win_rate), 2) if np.isfinite(win_rate) else None,
            "profit_factor": round(float(profit_factor), 3) if np.isfinite(profit_factor) else ("inf" if profit_factor == np.inf else None),
            "expectancy_net": _r(expectancy),
            "avg_win": _r(avg_win),
            "avg_loss": _r(avg_loss),
            "payoff_ratio": round(float(payoff), 3) if payoff is not None and np.isfinite(payoff) else None,
            "best_trade": _r(net.max()),
            "worst_trade": _r(net.min()),
            "median_trade": _r(net.median()),
            "std_trade": _r(net.std(ddof=1)) if n > 1 else None,
            "total_net": _r(net.sum()),
            "total_gross": _r(gross.sum()),
            "total_commission": _r(costs),
            "total_swap": _r(swaps),
            "cost_drag_pct": round(float(cost_drag), 2) if np.isfinite(cost_drag) else None,
            "r_multiple_expectancy": round(float(r_mult), 3) if np.isfinite(r_mult) else None,
            "avg_holding_min": _r(c["HOLDING_MIN"].mean()) if c["HOLDING_MIN"].notna().any() else None,
            "median_holding_min": round(hold_median, 1) if np.isfinite(hold_median) else None,
            "max_consec_wins": int(max_w),
            "max_consec_losses": int(max_l),
            "total_volume_lots": _r(c["VOLUME"].sum()),
            "profit_by_symbol": {k: round(float(v["net"]), 2) for k, v in by_symbol.to_dict(orient="index").items()},
            "profit_by_weekday": {k: round(float(v), 2) for k, v in by_dow.items()},
            "profit_by_hour": {int(k): round(float(v), 2) for k, v in by_hour.items()},
        }

    def summary(self) -> dict:
        return {"curve": self.stats(), "trades": self.trade_stats()}

    def monthly_returns(self) -> pd.DataFrame:
        if self._pnl.empty or "EQUITY" not in self._pnl:
            return pd.DataFrame()
        eq = self._pnl["EQUITY"].astype(float).sort_index()
        try:
            m = eq.resample("ME").last().pct_change().dropna() * 100.0
            m = m.replace([np.inf, -np.inf], np.nan).dropna()
            df = pd.DataFrame({"RETURN_PCT": m})
            df["YEAR"] = df.index.year
            df["MONTH"] = df.index.month
            return df.pivot_table(index="YEAR", columns="MONTH", values="RETURN_PCT", aggfunc="mean")
        except Exception:
            return pd.DataFrame()

    # ---------------------------------------------------------------- figures
    _PLOT_TEMPLATE = "plotly_white"

    def _has_curve(self) -> bool:
        return not self._pnl.empty and "EQUITY" in self._pnl.columns and len(self._pnl) > 0

    @staticmethod
    def _empty_fig(msg: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white", height=300, title=msg)
        return fig

    def fig_equity_drawdown(self, log_scale: bool = False) -> go.Figure:
        if not self._has_curve():
            return self._empty_fig("Sin curva de equity")
        eq = self._pnl["EQUITY"].astype(float).sort_index()
        bal = self._pnl["BALANCE"].astype(float).sort_index() if "BALANCE" in self._pnl else eq
        dd_pct, _ = self._drawdown_from_equity(eq)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.72, 0.28],
                            subplot_titles=("Equity vs Balance", "Underwater (drawdown %)"))
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity", mode="lines",
                                 line=dict(width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=bal.index, y=bal.values, name="Balance", mode="lines",
                                 line=dict(width=1.2, dash="dot"), opacity=0.85), row=1, col=1)
        fig.add_trace(go.Scatter(x=dd_pct.index, y=dd_pct.values * 100.0, name="Underwater %",
                                 mode="lines", line=dict(color="red", width=1.2),
                                 fill="tozeroy", fillcolor="rgba(255,0,0,0.25)"), row=2, col=1)
        if log_scale:
            fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(title_text="%", row=2, col=1)
        fig.update_layout(template=self._PLOT_TEMPLATE, height=560, hovermode="x unified",
                          title="Equity vs Balance + Underwater", legend=dict(orientation="h", y=1.08))
        return fig

    def fig_returns_dist(self) -> go.Figure:
        if not self._has_curve():
            return self._empty_fig("Sin datos diarios")
        dret = (self._daily_returns() * 100.0).dropna()
        if dret.empty:
            return self._empty_fig("Sin datos diarios")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=dret.values,
                                   nbinsx=min(60, max(10, len(dret) // 5)),
                                   name="Retornos diarios", opacity=0.75))
        fig.add_vline(x=float(dret.mean()), line_dash="dash", line_color="red",
                      annotation_text=f"Media {float(dret.mean()):.3f}%")
        fig.add_vline(x=0, line_color="black", line_width=1)
        fig.update_layout(template=self._PLOT_TEMPLATE, height=380, hovermode="x unified",
                          title="Distribución retornos diarios (%)", xaxis_title="% diario",
                          bargap=0.05)
        return fig

    def fig_monthly_heatmap(self) -> go.Figure:
        if not self._has_curve():
            return self._empty_fig("Sin datos mensuales")
        piv = self.monthly_returns()
        if piv.empty:
            return self._empty_fig("Sin datos mensuales")
        full = piv.reindex(columns=list(range(1, 13)))
        months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        vmax = float(np.nanmax(np.abs(full.values))) if np.isfinite(np.abs(full.values)).any() else 1.0
        vmax = vmax if vmax > 0 else 1.0
        fig = go.Figure(data=go.Heatmap(
            z=full.values, x=months, y=[str(y) for y in full.index],
            colorscale="RdYlGn", zmin=-vmax, zmax=vmax, zmid=0,
            text=np.round(full.values, 1), texttemplate="%{text:.1f}", hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
            colorbar=dict(title="%"),
        ))
        fig.update_layout(template=self._PLOT_TEMPLATE, height=max(300, 80 * len(full.index) + 180),
                          title="Heatmap retornos mensuales (%)")
        return fig

    def fig_rolling(self, window: int = 63) -> go.Figure:
        if not self._has_curve():
            return self._empty_fig("Sin datos para rolling")
        dret = self._daily_returns()
        if len(dret) < window:
            return self._empty_fig("Pocos datos para rolling")
        roll_sharpe = dret.rolling(window).mean() / dret.rolling(window).std(ddof=1) * np.sqrt(self._periods_per_year)
        roll_vol = dret.rolling(window).std(ddof=1) * np.sqrt(self._periods_per_year) * 100.0
        eq = self._daily_equity()
        roll_max = eq.rolling(window, min_periods=1).max()
        roll_dd = (eq / roll_max - 1.0) * 100.0
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                            subplot_titles=(f"Rolling Sharpe ({window}d)",
                                            f"Rolling volatilidad anualizada ({window}d) %",
                                            f"Rolling drawdown ({window}d) %"))
        fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values, name="Sharpe",
                                 mode="lines", line=dict(color="purple")), row=1, col=1)
        fig.add_hline(y=0, line_color="black", line_width=1, row=1, col=1)
        fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values, name="Vol %",
                                 mode="lines", line=dict(color="darkorange")), row=2, col=1)
        fig.add_trace(go.Scatter(x=roll_dd.index, y=roll_dd.values, name="DD %",
                                 mode="lines", line=dict(color="red"),
                                 fill="tozeroy", fillcolor="rgba(255,0,0,0.25)"), row=3, col=1)
        fig.update_layout(template=self._PLOT_TEMPLATE, height=700, hovermode="x unified",
                          title="Rolling Sharpe / Vol / Drawdown", showlegend=False)
        return fig

    def fig_trades(self) -> go.Figure:
        c = self.closed_trades
        if c.empty:
            return self._empty_fig("Sin trades cerrados (OUT)")
        nets = c["NET"].astype(float).values
        colors = ["green" if v > 0 else "red" for v in nets]
        x = list(range(len(nets)))
        cumsum = np.cumsum(nets)
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
                            row_heights=[0.62, 0.38],
                            subplot_titles=("PnL neto por trade + acumulado", "Distribución PnL por trade"),
                            specs=[[{"secondary_y": True}], [{}]])
        fig.add_trace(go.Bar(x=x, y=nets, name="Net por trade", marker_color=colors,
                             opacity=0.7, hovertemplate="Trade %{x}: %{y:.2f}<extra></extra>"),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=cumsum, name="Acumulado", mode="lines",
                                 line=dict(color="navy", width=2)), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Histogram(x=nets, name="Distribución",
                                   nbinsx=min(50, max(10, len(nets) // 4)),
                                   opacity=0.8, hovertemplate="%{x:.2f} (n=%{y})<extra></extra>"),
                      row=2, col=1)
        fig.add_vline(x=0, line_color="black", line_width=1, row=2, col=1)
        fig.update_xaxes(title_text="# trade (orden cronológico)", row=1, col=1)
        fig.update_layout(template=self._PLOT_TEMPLATE, height=640, hovermode="x unified",
                          title="PnL por trade", bargap=0.1)
        return fig

    def fig_profit_breakdown(self) -> go.Figure:
        c = self.closed_trades
        if c.empty:
            return self._empty_fig("Sin trades")
        by_sym = c.groupby("SYMBOL")["NET"].sum().sort_values()
        ct = pd.to_datetime(c["CLOSE_TIME"], errors="coerce", utc=True)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Net por símbolo", "Net por hora de cierre (UTC)"))
        fig.add_trace(go.Bar(y=[str(i) for i in by_sym.index], x=by_sym.values, orientation="h",
                             name="Por símbolo",
                             marker_color=["green" if v > 0 else "red" for v in by_sym.values],
                             hovertemplate="%{y}: %{x:.2f}<extra></extra>"), row=1, col=1)
        if ct.notna().any():
            by_h = c.assign(_h=ct.dt.hour).groupby("_h")["NET"].sum().sort_index()
            fig.add_trace(go.Bar(x=[str(i) for i in by_h.index], y=by_h.values, name="Por hora",
                                 marker_color=["green" if v > 0 else "red" for v in by_h.values],
                                 hovertemplate="h %{x}: %{y:.2f}<extra></extra>"), row=1, col=2)
        else:
            fig.add_annotation(text="Sin fechas válidas", x=0.75, y=0.5, showarrow=False, xref="paper", yref="paper")
        fig.update_layout(template=self._PLOT_TEMPLATE, height=420, title="Desglose de PnL", showlegend=False)
        return fig

    # ------------------------------------------------------------------- plot
    def plot(self, log_scale: bool = False):
        fig = self.fig_equity_drawdown(log_scale=log_scale)
        fig.show()
        return fig

    def plot_old(self):
        warnings.warn("plot_old() está obsoleto; usa plot() (plotly).", DeprecationWarning, stacklevel=2)
        return self.plot()

    # ----------------------------------------------------------------- report
    @staticmethod
    def _dict_to_html_table(d: dict, title: str) -> str:
        rows = []
        for k, v in d.items():
            if isinstance(v, dict):
                v = ", ".join(f"{kk}: {vv}" for kk, vv in v.items()) or "—"
            rows.append(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>")
        return f"<h3>{html.escape(title)}</h3><table><tbody>{''.join(rows)}</tbody></table>"

    def to_html(self, title: str = "PyEventBT — Backtest Report",
                include_plotlyjs: str = "cdn") -> str:
        curve = self.stats()
        tstats = self.trade_stats()
        figs = {
            "Equity + Underwater": self.fig_equity_drawdown(),
            "PnL por trade": self.fig_trades(),
            "Heatmap mensual": self.fig_monthly_heatmap(),
            "Distribución diaria": self.fig_returns_dist(),
            "Rolling Sharpe / Vol / DD": self.fig_rolling(),
            "Desglose por símbolo y hora": self.fig_profit_breakdown(),
        }
        divs = []
        for i, (name, fig) in enumerate(figs.items()):
            divs.append(f"<section><h2>{html.escape(name)}</h2>"
                        f"{pio.to_html(fig, full_html=False, include_plotlyjs=(include_plotlyjs if i == 0 else False))}</section>")
        blocks = "\n".join(divs)
        month_piv = self.monthly_returns()
        month_html = (
            month_piv.round(2).to_html(classes="monthly", na_rep="—")
            if not month_piv.empty
            else "<p>Sin datos mensuales.</p>"
        )
        closed = self.closed_trades
        trades_html = (
            closed.tail(50).round(2).to_html(index=False, classes="trades")
            if not closed.empty
            else "<p>Sin trades cerrados.</p>"
        )
        return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/>
<title>{html.escape(title)}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:24px;background:#f7f8fa;color:#1a1a1a}}
h1{{font-size:26px}}h2{{margin-top:32px;border-bottom:2px solid #ddd;padding-bottom:6px}}h3{{margin-top:20px}}
table{{border-collapse:collapse;width:100%;max-width:900px;background:#fff}}
td,th{{border:1px solid #ddd;padding:6px 10px;font-size:13px;text-align:left}}
tr:nth-child(even){{background:#f2f4f7}}
section{{margin-bottom:24px;background:#fff;border:1px solid #ddd;padding:12px}}
.plotly-graph-div{{width:100%}}
.note{{color:#555;font-size:13px;max-width:950px}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="note">Generado {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Gráficos interactivos plotly (zoom/pan/hover) ·
NET por trade = GROSS − COMMISSION(IN+OUT) + SWAP(MT5 con signo) − FEE ·
Trades cerrados = fills OUT pareados con su IN por POSITION_ID · Métricas diarias con {self._periods_per_year} periodos/año.</p>
<h2>1 · Curva de equity</h2>
{self._dict_to_html_table(curve, "Métricas de curva")}
<h2>2 · Trades</h2>
{self._dict_to_html_table(tstats, "Métricas de trades")}
<h2>3 · Gráficos interactivos</h2>
{blocks}
<h2>4 · Tabla mensual (%)</h2>
{month_html}
<h2>5 · Últimos 50 trades cerrados</h2>
{trades_html}
</body></html>"""

    def save_report(self, path: str, title: str = "PyEventBT — Backtest Report",
                    include_plotlyjs: str = "cdn") -> str:
        content = self.to_html(title=title, include_plotlyjs=include_plotlyjs)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
