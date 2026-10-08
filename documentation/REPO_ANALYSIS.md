# PyEventBT — Extensive Repository Analysis
## Expert Algorithmic-Trader Perspective: From Strategy Creation → Research → Backtesting → Robustness → Production Portfolio

> Repo: `pyeventbt-private` (`pyeventbt` v0.0.13, Python 3.12, Poetry, Apache-2.0)
> Date of analysis: 2026-10-06
> Goal under evaluation: **create strategies (data mining / arbitrage / classic), do research, backtest + robustness-test, and run a production portfolio of strategies.**
> Verdict up front: **solid event-driven MT5 backtest/live execution core with good anti-lookahead hygiene, but not yet a strategy-factory / research / robustness / multi-strategy portfolio platform.** `BacktestResults` is essentially equity-curve + plot. No optimizer, no walk-forward runner, no Monte-Carlo, no risk overlay, no portfolio allocator, no statistical-arbitrage or data-mining scaffolding. Several simulator realism gaps must be fixed before trusting Sharpe / DD / sizing.

---

## Table of Contents

1. [General: What This Repo Is](#1-general-what-this-repo-is)
2. [General: What It Is Not](#2-general-what-it-is-not)
3. [Specific: Architecture & Data Flow](#3-specific-architecture--data-flow)
4. [Specific: Module-by-Module Deep Dive](#4-specific-module-by-module-deep-dive)
5. [What Can Be Done Today](#5-what-can-be-done-today--capability-matrix)
6. [Red Flags & Correctness Risks](#6-red-flags--correctness-risks-found-in-code)
7. [How to Improve (Fixes & Hardening)](#7-how-to-improve-fixes--hardening)
8. [New Development Needed for Stated Goal](#8-new-development-needed-for-stated-goal)
9. [Suggested Roadmap (Phased)](#9-suggested-roadmap-phased)
10. [Appendix: Key Files Index](#10-appendix-key-files-index)

---

## 1. General: What This Repo Is

**PyEventBT is an institutional-style, event-driven backtesting + live-trading framework that couples pure Python strategy code to MetaTrader 5.**

Core promise in `README.md` and `pyeventbt/strategy/strategy.py`:

> **"One codebase, two modes"** — full mock of the MT5 API, so the same strategy code runs in backtest and live with no MQL5.

Pipeline (documented in README):

```text
BarEvent → Your Strategy → SignalEvent → Sizing Engine → Risk Engine → OrderEvent → Execution Engine → FillEvent → Portfolio
```

Key properties:

- **Event-driven, not vectorized.** `pyeventbt/trading_director/trading_director.py:TradingDirector` pumps a `Queue` of `BAR / SIGNAL / ORDER / FILL / SCHEDULED` events (`pyeventbt/events/events.py`). This is the right paradigm for realistic multi-timeframe / multi-symbol / pending-order logic and for backtest→live parity. Competitors compared in README: Backtrader, Zipline, VectorBT.
- **MT5-native.** Live path uses `MetaTrader5` Python package (Windows-only); backtest path uses `Mt5SimulatorWrapper` / `Mt5SimulatorExecutionEngineConnector` that emulates `order_send`, `positions_get`, `symbol_info`, account/margin semantics (`pyeventbt/broker/mt5_broker/`, `pyeventbt/execution_engine/connectors/mt5_simulator_execution_engine_connector.py` ~1394 lines).
- **Modular engines.** Signal / Sizing / Risk are independent, interchangeable blocks:
  - Custom via decorators: `@strategy.custom_signal_engine`, `@strategy.custom_sizing_engine`, `@strategy.custom_risk_engine`.
  - Predefined: `SignalMACrossover`, `SignalPassthrough`, `MT5MinSizing`, `MT5FixedSizing`, `MT5RiskPctSizing`, `PassthroughRiskEngine`.
- **Multi-everything (in principle).** `StrategyTimeframes` supports `1min … 12M` (`pyeventbt/strategy/core/strategy_timeframes.py`), CSV connector resamples M1 → any TF, examples show `[1h, 1D]` multi-TF. Symbols list is free-form (`symbols_to_trade`).
- **Python quant stack.** `polars`, `pandas`, `numpy`, `numba`, `scipy`, `scikit-learn`, `matplotlib`, `pydantic`, `PyYAML`, `quantdle` (`pyproject.toml`, `requirements.txt`). Indicators are `numba.njit` + `numpy` (`pyeventbt/indicators/indicators.py`).
- **Data inputs today:** local MT5-format M1 CSVs (`date,time,open,high,low,close,tickvol,volume,spread`), bundled Forex M1 sample (`pyeventbt/data_provider/connectors/historical_csv_data/`), `QuantdleDataUpdater` for M1/H1/D1 download → CSV cache (`example_quantdle_ma_crossover.py`), live MT5 `copy_rates_from_pos`. Standalone `FXMacroDataCalendarConnector` exists but is **not wired** into the event loop.
- **Maturity:** v0.0.13, `Development Status :: 3 - Alpha`. `CHANGELOG.md` shows recent work is correctness fixes: multi-symbol equity, SharedData reset, SL-modification price, live race-condition, Decimal→float SLTP, margin rounding down, duplicate-symbol dedup. Good signal — maintainers fix simulator bugs — but also signal that the simulator had material PnL-affecting bugs until recently.

Purpose in one sentence: **let an MT5 trader write Python signal logic once, backtest it bar-by-bar with pending orders + SL/TP + margin/commission, then run the identical logic live against MT5.**

---

## 2. General: What It Is Not

As an algo-trader building a **portfolio of robust strategies**, be clear on what is missing:

| Needed for your goal | Status in repo |
|---|---|
| Strategy library (trend, mean-reversion, breakout, carry, intraday seasonal) | Only 2 toy demos: MA-crossover dominance, BB-stop-straddle. No library. |
| Data-mining / feature-research harness | None. No screener, factor store, notebook helpers, labeling, purged CV. |
| Statistical / latency arbitrage scaffolding | None. Single MT5 account model, no multi-venue, no pair/cointegration, no borrow/funding, `OrderType.CONT` defined but unused. |
| Optimizer (grid / random / Bayesian) | None. Must manually loop `strategy.backtest()`. `HyperParameter` / `Variable` are DTOs, nothing consumes them. Commented-out `hyperopt` imports in `strategy.py`. |
| Walk-forward runner | DTO only: `strategy/core/walk_forward.py:WalkForwardResults`, `WalkforwardType(ANCHORED/UNANCHORED)` with `to/from_csv`. No splitter, retrainer, or `Strategy.walkforward()`. |
| Robustness suite (MC, jitter, slippage/spread shocks, PBO, DSR) | None. |
| Performance analytics | Minimal: `BacktestResults.pnl/returns/trades + plot()` (`pyeventbt/backtest/core/backtest_results.py`). No Sharpe/Sortino/CAGR/Calmar/MaxDD/duration, win-rate, profit factor, expectancy, exposure, monthly heatmap. Lots of commented-out imports (`numpy`, `sklearn`, `scipy.stats.norm`). |
| Portfolio of strategies (allocator, cross-strategy risk) | `strategy_id → magic_number` dict supports multi-ID internally, but `backtest(strategy_id=...)` / `run_live(...)` runs **one** strategy/magic at a time. No capital allocator, no combined `BacktestResults`, `Portfolio` is single-strategy view. |
| Production ops (monitoring, kill-switch, reconciliation, journaling) | Hooks (`ON_START/SIGNAL/ORDER/FILL/ON_END`) + `TradeArchiver` CSV/parquet export exist, but no health checks, no daily loss-limit halt, no broker-vs-local reconciliation, no Telegram/Discord alerting. |
| Test suite / CI | No `tests/` directory found. Only `.github/` workflow + manual examples. High risk for silent regressions given Decimal/float/margin complexity. |

Bottom line: **execution core = usable foundation; research + robustness + portfolio layers = to be built.**

---

## 3. Specific: Architecture & Data Flow

Entry: `pyeventbt/__init__.py`, `pyeventbt/strategy/strategy.py:Strategy`, `pyeventbt/app.py`, `example_ma_crossover.py`, `example_bbands_breakout.py`, `example_quantdle_ma_crossover.py`.

Event types (`pyeventbt/events/events.py`): `EventType(BAR/SIGNAL/ORDER/FILL/SCHEDULED)`, `SignalType(BUY/SELL)`, `OrderType(MARKET/LIMIT/STOP/CONT)`, `DealType(IN/OUT)`, `Bar(open/high/low/close/tickvol/volume/spread/digits + *_f float props)`, `BarEvent(symbol,datetime,data,timeframe)`, `SignalEvent(...,order_price,sl,tp,rollover,forecast)`, `OrderEvent(+volume,buffer_data)`, `FillEvent(deal,...,commission,swap,fee,gross_profit)`.

`TradingDirector` loop (backtest):

1. `DataProvider.update_bars() → Queue.put(BarEvent)` — `data_provider/services/data_provider_service.py` → `connectors/csv_data_connector.py:CSVDataProvider` (backtest) / `mt5_live_data_connector.py` (live).
2. `BAR: PortfolioHandler.process_bar_event (Portfolio._update_portfolio) + ScheduleService.run_scheduled_callbacks + SignalEngineService.generate_signal(event)`.
3. User fn `(BarEvent, Modules) → SignalEvent | list` where `Modules{TRADING_CONTEXT, DATA_PROVIDER, EXECUTION_ENGINE, PORTFOLIO}` (`strategy/core/modules.py`). Puts `SignalEvent` back on queue.
4. `SIGNAL: PortfolioHandler.process_signal_event → SizingEngine.get_suggested_order → RiskEngine.assess_order → Queue.put(OrderEvent)`.
5. `ORDER: ExecutionEngine._process_order_event` → simulator or live MT5. Handles `MARKET/LIMIT/STOP`, SL/TP per base bar, `close_*/cancel_*/update_position_sl_tp`.
6. `FILL: Queue.put(FillEvent) → PortfolioHandler.process_fill_event → TradeArchiver.archive_trade`. Note: portfolio state is pulled from execution engine, not from fills (authoritative in exec-engine).
7. End: `close_positions_end_of_data` drain + `PortfolioHandler.process_backtest_end() → BacktestResults(pnl, trades)` + optional export to `~/Desktop/PyEventBT/backtest_results_{csv,parquet}/`.

`Strategy.backtest()` signature (`strategy.py:345-361`): `strategy_id, initial_capital, account_currency, account_leverage, start/end_date, backtest_name, symbols_to_trade, csv_dir, run_scheduled_taks [sic typo], export_backtest_csv/parquet, backtest_results_dir, symbols_info_override`. `run_live(mt5_configuration, strategy_id, initial_capital, symbols_to_trade, heartbeat)`.

Timeframes: `StrategyTimeframes` `1min…12M` + `to_timedelta()`. `backtest()` sorts TFs, `base_timeframe = timeframes[0]`. CSV connector loads `{SYMBOL}.csv`, filters window, `group_by_dynamic(1m)`, forward-fills gaps, resamples via `_AGG_MAP(first/max/min/last/sum/sum/first)` + `_timeframe_to_duration()`, integer-scaled `Bar` generator (zero-copy `memoryview`, `price*10^digits`). Base bar always emitted; higher TF gated by `_base_tf_bar_creates_new_tf_bar`. `get_latest_bars(symbol,tf,N)` excludes forming bar for HTF (anti-lookahead). `get_latest_tick()` peeks **next bar open+spread** for execution — intentional 1-bar execution delay. `ScheduleService` (`schedule_service/schedule_service.py`) supports `Strategy.run_every(interval)` callbacks.

---

## 4. Specific: Module-by-Module Deep Dive

### 4.1 Data provider — `pyeventbt/data_provider/`

- `connectors/csv_data_connector.py:CSVDataProvider` — M1 CSV scan (Polars `scan_csv`), resampling, gap forward-fill (`tickvol=volume=spread=1` skipped in `update_bars()`), per-symbol per-TF `latest_index`, anti-lookahead filters (`datetime <= cutoff`; HTF returns second-last). Auxiliary FX symbols auto-added for `account_currency in {USD,EUR,GBP}` for conversion (`_create_auxiliary_symbol_list`, `utils/utils.py:ALL_FX_SYMBOLS`).
- `connectors/mt5_live_data_connector.py:Mt5LiveDataProvider` — `_map_timeframe() → mt5.TIMEFRAME_*`, `copy_rates_from_pos(from_pos=1)` (last closed), per-symbol per-TF `last_bar_tf_datetime` poll. v0.0.13 fixed double-`get_latest_bar` race.
- `services/quantdle_data_updater.py:QuantdleDataUpdater.update_data(csv_dir,symbols,start,end,timeframe="1min",spread_column)` — gap-fills M1/H1/D1 via `quantdle.Client.download_data(output_format="polars")` → MT5 CSV. Only supported vendor path; good for Forex but not equities/crypto order-book.
- `connectors/fxmacrodata_calendar_connector.py` — standalone REST calendar (`fetch_calendar/has_event_between/next_events`), **not wired** into `TradingDirector`/`DataProvider`. Cannot currently filter news or use it as feature without custom code.
- `core/configurations/data_provider_configurations.py:CSVBacktestDataConfig`, `MT5LiveDataConfig`; `core/entities/bar.py:Bar`.

Trader view: data layer is **clean for M1 Forex OHLC backtests**, adequate for daily/hourly swing systems. Not sufficient for tick-level HFT research, multi-venue arb, or fundamental/alternative data mining.

### 4.2 Signal engine — `pyeventbt/signal_engine/`

- `services/signal_engine_service.py:SignalEngineService`, `core/interfaces/signal_engine_interface.py`, `core/configurations/signal_engine_configurations.py:MACrossoverConfig(fast/slow_period,ma_type)`.
- `signal_engines/signal_ma_crossover.py`, `signal_passthrough.py`.
- Custom path is the real product: any `(BarEvent,Modules)->SignalEvent` fn. Examples in `example_*.py` show correct pattern: `get_latest_bars() → SMA/BollingerBands.compute() → check open positions → close opposite → SignalEvent(time_generated = event.datetime + TF.to_timedelta() in BACKTEST else now(), order_price from latest tick)`.

Trader view: flexible, but **no strategy templates, no parameter object plumbing, no signal-combining / voting / ML forecast (`forecast` field exists on SignalEvent, `Utils.cap_forecast(-20..20)` exists, but nothing consumes it)**. You will hand-roll everything.

### 4.3 Sizing engine — `pyeventbt/sizing_engine/`

- `core/configurations/sizing_engine_configurations.py:Min/Fixed(volume)/RiskPct(risk_pct)`, `services/sizing_engine_service.py`, `sizing_engines/mt5_min_sizing.py` (volume_min via `mt5.symbol_info`), `mt5_fixed_sizing.py` (`Decimal(config.volume)` unchecked), `mt5_risk_pct_sizing.py` (requires `sl!=0`; `entry=tick ask/bid or order_price`; `tick_value=contract*tick_size` converted `profit_ccy→acct_ccy`; `distance=|entry-sl|/tick_size`; `vol=equity*risk%/(distance*tick_value)` floored to `volume_step`; v0.0.8 fixed to always round **down**).
- `set_suggested_order_function(fn)` override exists.

Trader view: `RiskPct` is the only production-relevant sizer, and it **depends on SL being set** — examples set `sl=0`, so they silently fall back to min-lot if you switch sizer. No volatility-targeting, no Kelly-fractional, no portfolio-heat-aware sizing.

### 4.4 Risk engine — `pyeventbt/risk_engine/`

- Only `PassthroughRiskEngine` (returns `suggested.volume` unchanged). `RiskEngineService._create_and_put_order_event` copies signal→order if `>0`; `set_custom_asses_order(fn)` override.
- No exposure cap, daily/weekly loss limit, max orders/positions, leverage cap, correlation filter, news blackout, weekend-flat enforcement by default.

Trader view: **risk is a stub.** Any live deployment without a custom risk overlay is unacceptable.

### 4.5 Execution engine (simulator) — `pyeventbt/execution_engine/connectors/mt5_simulator_execution_engine_connector.py`

The most important file for trust. State: `pending_orders, open_positions, executed_deals, balance/equity/used_margin/free_margin:Decimal, ticketing_counter=200M/deal=300M, margin_call:bool`. Config: `MT5SimulatedExecutionConfig(initial_balance, account_currency, account_leverage, magic_number)` — note `account_leverage` is **stored but unused**; margin derives solely from `SymbolInfo.margin_initial` (hence new `symbols_info_override` in Unreleased changelog).

- `MARKET`: validate `BUY/SELL`, `volume>=volume_min`, `_check_common_trade_values`, `required_margin=_compute_required_margin`, reject if `free<required` (`margin_call=True`, `continue_backtest=False`); `fill_price = get_latest_ask (BUY) / get_latest_bid (SELL)` (next forming-bar open+spread, 1-bar delay); SL/TP validated vs fill; creates `TradePosition + TradeDeal(IN) + FillEvent(IN,swap=0,fee=0) + OrderSendResult(10009)`; `used_margin+=req; free=equity-used; balance-=commission`.
- `PENDING`: `BUY+LIMIT→2, BUY+STOP→4, SELL+LIMIT→3, SELL+STOP→5`; same validations vs `order_price`; no margin lock; stored in `pending_orders`.
- Per `BarEvent(symbol)`: `_check_if_sl_tp_hit → _update_positions_floating_pnl → _check_if_pending_orders_filled → _check_margin_call → _check_equity_balance_are_positive`.
- Pending fill: `high_ask=high+spread, low_ask=low+spread`; `buy_limit:low_ask<price; sell_limit:high>price; buy_stop:high_ask>price; sell_stop:low<price` (penetration required, ask-aware for buys). On hit: margin re-check, `TradePosition(price_open=request.price)` exact limit price, `FillEvent(IN)+TradeDeal(IN)+commission`.
- SL/TP: only `bar_event.symbol`; **pessimistic SL-first**: `LONG:low_f<=sl; SHORT:high_f+spread_f>=sl`; else TP strict `LONG:high_f>tp; SHORT:low_f+spread_f<tp`; `closed_price=sl/tp` exact; `FillEvent(OUT)+TradeDeal(reason 4/5)`.
- `close_position(ticket,partial_volume=0)`: `close_price=latest_tick[ask if SHORT else bid]`; partial reduces volume but **releases full stored margin (bug)**.
- Margin: `FX:volume*contract*margin_initial; CFD:volume*contract*last_bid*margin_initial` → `convert_currency_amount(margin_ccy→acct_ccy via FX bid)`. Frozen at open, `free=equity-used`.
- Commission `_compute_commission_in_account_ccy` (Darwinex schedule): `FX 2.5/side/lot; XAU/XAG/XTI/XNG 0.0025%*contract*vol*price; NI225 35JPY; WS30 0.35USD; SP500 0.275pt; FCHI40/AUS200/NDX/UK100/STOXX50E/GDAXI/SPA35 2.75pt; SYNTHUSD 0; ETF 0.02USD`, converted, `abs()`. Charged **on entry and each exit** → round-trip 2×.
- Profit `_compute_trade_gross_profit`: `(close-open)*vol*contract` in `currency_profit→acct_ccy` (sign flipped for SELL); spread via ask/bid entry/exit; floating `price_current=close_f`, `equity=balance+sum(profit)`.
- Live connector (`mt5_live_execution_engine_connector.py`) is direct `MetaTrader5.order_send/positions_get/...` passthrough; `_update_values...:pass`.

### 4.6 Portfolio / handler / archiver / results

- `portfolio/portfolio.py:Portfolio` — `_update_portfolio(bar)` pulls `_get_strategy_positions/pending_orders/balance/equity`; `realised=balance-initial; unrealised=equity-balance`; `historical_balance/equity[datetime]` **only if `timeframe==base` and `symbol==first_seen` (single-symbol sampling)**; `_update_portfolio_end_of_backtest` forces `equity=balance`.
- `portfolio_handler/portfolio_handler.py:PortfolioHandler` — `process_bar_event` (skip non-base TF) → `PORTFOLIO._update`; `process_signal_event → sizer → risk → OrderEvent`; `process_fill_event` only `TradeArchiver.archive_trade`; `process_backtest_end → BacktestResults`.
- `trade_archiver/trade_archiver.py:TradeArchiver` — `export_historical_trades_dataframe/json/parquet/csv`.
- `backtest/core/backtest_results.py:BacktestResults(pnl,trades)` — `pnl/returns/trades/backtest_pnl` props, `plot()/plot_old()` (pandas EQUITY+BALANCE matplotlib). Nothing else.

### 4.7 Broker mock / config / utils / hooks / schedule

- `broker/mt5_broker/mt5_simulator_wrapper.py`, `connectors/mt5_simulator_connector.py`, `shared/shared_data.py` (global `account_info/terminal_info/symbol_info/credentials/last_error_code`, reset per `backtest()` since v0.0.7), `shared/default_symbols_info.yaml`, `core/entities/{symbol_info,trade_position,trade_order,trade_deal,trade_request,order_send_result,account_info,tick,terminal_info,...}.py`. `SymbolInfo` carries `trade_contract_size, trade_tick_size/volume_min/max/step, margin_initial, spread, swap_long/short, swap_mode, currency_{base,profit,margin}, point/digits`.
- `config/configs.py:Mt5PlatformConfig(path,login,password,server,timeout,portable)`, `strategy/core/account_currencies.py`, `verbose_level.py`, `walk_forward.py`, `strategy_timeframes.py`, `modules.py`, `services/parameter_store.py`.
- `utils/utils.py:Utils(order_type_str/int, check_new_m1_bar, currency conversion, futures suffix, dateprint, cap_forecast) + TerminalColors/LoggerColorFormatter`, `ALL_FX_SYMBOLS`.
- `hooks/hook_service.py:Hooks(ON_START/SIGNAL/ORDER/FILL/ON_END)+HookService` invoked in `TradingDirector`; duplicate at `trading_director/services/hook_service.py`.
- `schedule_service/schedule_service.py`, `trading_context/trading_context.py:TypeContext(BACKTEST/LIVE)`, `trading_director/trading_director.py`, `core/entities/variable.py`, `hyper_parameter.py`.
- Indicators (`indicators/indicators.py`, numba `njit`, numpy in/out): file defines **22** (`KAMA, ATR, SMA, EMA, RSI, ADX(+DI/-DI), Momentum, BollingerBands, DonchianChannels, MACD, KeltnerChannel, ADR, VWAP, Stochastic, CCI, WilliamsR, ROC, TRIX, DeMarker, Aroon, RVI`) but `indicators/__init__.py` only re-exports `ATR,KAMA,SMA,EMA,TRIX,DeMarker,RVI` — **import trap** for newcomers.

---

## 5. What Can Be Done Today — Capability Matrix

| Task | Can do? | How | Notes / limits |
|---|---|---|---|
| Single-symbol, single-TF backtest (Forex M1) | ✅ Yes | `Strategy + @custom_signal_engine + Min/Fixed sizing + Passthrough risk + backtest(csv_dir=...)` | Bundled sample data if `csv_dir=None`. |
| Multi-TF filter (e.g. H1 entry + D1 regime) | ✅ Yes | `strategy_timeframes=[H1,D1]`, branch on `event.timeframe`, `get_latest_bars(symbol,TF,N)` | HTF forming-bar excluded by design. Base TF = smallest. |
| Multi-symbol in one strategy | ⚠️ Partial | `symbols_to_trade=[...]` | Works, but `Portfolio.historical_*` samples single symbol/time; equity math fixed in v0.0.7 to sum all positions, historical curve still suspect for multi-symbol — verify. |
| Pending orders (Limit/Stop) + SL/TP modify | ✅ Yes | `OrderType.LIMIT/STOP`, `update_position_sl_tp`, `cancel_*` | Exact-trigger fills, no slippage; SL-first ambiguity; see §6. |
| Risk-% sizing | ✅ Yes | `RiskPctSizingConfig` | Requires non-zero SL; otherwise silent fallback issues. |
| Custom sizing / risk overlay | ✅ Yes | `@custom_sizing_engine`, `@custom_risk_engine`, `set_custom_asses_order` | You must write it; no built-ins. |
| Live trading on MT5 | ✅ Yes | `run_live(Mt5PlatformConfig, ...)` + `MetaTrader5` pkg, Windows | Magic-number filtering inconsistent sim vs live; test reconciliation. |
| CSV/parquet trade + PnL export | ✅ Yes | `export_backtest_csv/parquet`, `TradeArchiver` | Paths default to Desktop; parametrize `backtest_results_dir`. |
| Quantdle download → backtest | ✅ Yes | `QuantdleDataUpdater.update_data(...)` | Forex M1/H1/D1; no tick/order-book. |
| News-calendar filter | ⚠️ Manual | `FXMacroDataCalendarConnector` standalone | Not in event loop; wire yourself. |
| Optimization / WFA / Monte-Carlo / portfolio allocator | ❌ No | — | Build (see §8). |
| Arbitrage (cross-broker, triangular, cash-carry) | ❌ No | — | Single-account model; build multi-venue layer. |
| Data mining at scale | ❌ No | — | No runner; manual loops only, no statistical guards. |

Working minimal loop today:

```python
strategy = Strategy()
@strategy.custom_signal_engine(strategy_id="ma10_30", strategy_timeframes=[StrategyTimeframes.ONE_DAY])
def sig(event, modules): ...
strategy.configure_predefined_sizing_engine(MinSizingConfig())
strategy.configure_predefined_risk_engine(PassthroughRiskConfig())
res = strategy.backtest(strategy_id="ma10_30", initial_capital=100_000,
    symbols_to_trade=["EURUSD"], csv_dir="./data",
    start_date=datetime(2020,1,1), end_date=datetime(2023,12,1),
    account_currency=AccountCurrencies.USD)
print(res.trades.head()); print(res.pnl.tail()); res.plot()
```

---

## 6. Red Flags & Correctness Risks Found in Code

Ranked by PnL-impact for a trader. All paths given so you can verify.

### 🔴 Critical (can flip a backtest from profit to loss in live)

1. **No slippage / spread-widening / requote / partial-fill model.** `mt5_simulator_execution_engine_connector.py`: `deviation=0`, always `10009 DONE`, `FOK` ignored, fill at quote/limit, `bid==ask==price` in result. Pending/SLTP filled at **exact trigger, not through-price**, no gap handling (gap through SL still credited at `sl`). Live MT5 will be worse on news, rollover, illiquid symbols. **Effect:** systematically overstates breakout/stop strategies (e.g. `example_bbands_breakout.py` straddle).
2. **Intrabar SL-vs-TP ambiguity resolved pessimistically-but-deterministically (SL always wins).** `_check_if_sl_tp_hit` checks SL first per bar. If both SL and TP fall inside one base bar, backtest always records a loss; live outcome depends on tick path. **Effect:** biases against tight SL+TP scalpers; hides path-dependency. Needs tick-interpolation or ambiguity flag.
3. **No swap / rollover / funding.** `FillEvent(swap=0,fee=0)` always, despite `swap_long/short/mode` in YAML. No triple-Wednesday, no weekend, no crypto funding, no ETF dividend/tax, no borrow cost for shorts. **Effect:** carry / multi-day swing backtests are fiction; short-bias strategies look cheaper than they are.
4. **Margin model incomplete.** `account_leverage` stored but unused; margin solely from `margin_initial` (now overridable via `symbols_info_override`). `used_margin` frozen at open (no maintenance drift), no stop-out % (only `free<required → margin_call + continue_backtest=False` and `equity/balance<0 → stop`). No `volume_max/step` enforcement in sim (only `min`; live caps `max`), no `trade_stops_level/freeze_level`, `expiration/type_filling/time` ignored, no margin lock for pending orders. **Effect:** over-leveraged backtests pass that MT5 would reject; pending-heavy strategies understate margin.
5. **Partial-close margin bug.** `close_position(partial_volume)`: reduces `volume` but does `used_margin -= full stored margin` (over-releases). **Effect:** frees too much margin after partials → allows oversized follow-ons in backtest.
6. **`BacktestResults` has no risk-adjusted metrics.** `backtest/core/backtest_results.py` is 63 lines: `pnl/returns/trades + plot()`. No Sharpe/Sortino/CAGR/Calmar/MaxDD/duration, win-rate, profit factor, expectancy, exposure, autocorrelation of returns. Commented-out `numpy/sklearn/scipy` imports suggest analytics were planned then stripped. **Effect:** you cannot rank strategies or detect overfit from this object; every research decision requires external code.
7. **Historical equity sampling is single-symbol.** `portfolio/portfolio.py`: `historical_balance/equity[datetime]` only if `timeframe==base and symbol==first_seen`. Multi-symbol equity curve is mis-timed even though live equity sums all positions (fixed v0.0.7 for live calc, not for history). **Effect:** multi-symbol backtest plots/returns are unreliable.
8. **No test suite.** Zero `tests/` files; only examples. Given Decimal/float mixing, margin/FX conversion, and recent multi-symbol/equity bugs, regressions are likely. Do not scale capital without adding tests.

### 🟡 Major (will bite in research / live parity)

9. **Lookahead hygiene is good but leaky at edges.** Main path clean (`get_latest_bars` filters `datetime<=cutoff`; HTF excludes forming bar; `get_latest_tick` peeks next-bar open+spread for 1-bar execution delay — correct **if** strategy only uses closed bars). Residual leaks: `mt5.symbol_info().bid/ask` static YAML (stale if strategy reads directly); margin/commission/profit FX conversions mix next-open vs closed rates; `update_position_sl_tp` validates vs `price_current (close)` not tradable quote; timestamps inconsistent (`market/pending/SLTP @bar.datetime`, `close/cancel @latest_datetime+1min`).
10. **Commission is hardcoded Darwinex schedule** (`_compute_commission_in_account_ccy`), FX-conversion at `bid` without spread. If your broker differs, all net metrics are wrong. Must parametrize per-symbol commission + spread model.
11. **Examples teach risky habits.** Both `example_ma_crossover.py` and `example_bbands_breakout.py` set `sl=0, tp=0` (no protection), use `MinSizingConfig` (no risk control), `PassthroughRiskConfig` (no risk), and MA example does `close_* + SignalEvent` in same bar (relies on simulator ordering). Copy-pasting to live = unprotected min-lot martingale risk.
12. **Live/sim parity gaps.** Sim returns all positions; live filters by `magic`. Sim `sl<=fill` equality allowed (marked `TODO test-only`); MT5 may reject. `Decimal` SLTP crashed live `order_send` until v0.0.12 (now cast to float). `run_live` has no reconciliation of external/manual trades, no heartbeat-missed-bar catch-up spec, Windows-only.
13. **API/UX traps.** `indicators/__init__.py` re-exports only 7 of 22 indicators (must `from pyeventbt.indicators.indicators import BollingerBands` as README does, inconsistent). `backtest(..., run_scheduled_taks)` typo. `OrderType.CONT` defined, never handled. `HyperParameter`/`Variable` + commented `hyperopt` = dead-end optimizer hint. Duplicate `HookService` (`hooks/` vs `trading_director/services/`). Default export to `Desktop/PyEventBT` surprises CI/headless.
14. **Data gaps.** CSV must be MT5-format M1; no tick, no order-book, no corporate actions/dividends/splits, no survivorship-bias handling for equities, gap forward-fill hides illiquidity (creates flat bars with `spread=1`). Quantdle only. No point-in-time fundamental data.
15. **Performance / scale.** Bar-by-bar Python loop + per-bar `DataProvider` Polars slice + Decimal margin math will be slow for 10y M1 × many symbols × optimization grid. No multiprocessing, no caching, no vectorized pre-computation path. Optimization loops will be painful without a runner + `SharedData()` reset discipline (added v0.0.7 for sequential runs).

### 🟢 Minor (fix opportunistically)

- Logger name fixed v0.0.7 (`pyeventbt`), but `strategy.py` adds a `StreamHandler` per `Strategy()` instantiation → duplicate logs when running many backtests in one process.
- `strategy_id` must be `int(strategy_id)`-castable (used as MT5 magic). Non-numeric IDs crash `backtest()`/`run_live()`.
- `FixedSizingConfig(volume)` unchecked against `volume_min/max/step`.
- `RiskPctSizing` uses `account_info().equity` via `SharedData` — sensitive to call order; verify it reads post-update equity.
- `requirements.txt` includes `quantdle` but `pyproject.toml` does not → Poetry vs pip drift.

---

## 7. How to Improve (Fixes & Hardening)

### 7.1 Simulator realism (do before trusting any Sharpe)

- [ ] **Execution cost model:** add `ExecutionCostConfig{slippage_ticks|atr_frac, spread_multiplier, commission_per_side_per_symbol, delay_bars}`. Fill MARKET at `quote ± slippage`, pending at `trigger ± slippage-through`, SL/TP at `worst(trigger, next-open-gap)`. Add `volume_max/step` check, `stops_level/freeze_level`, pending margin lock option.
- [ ] **Intrabar ambiguity:** if SL and TP both inside bar, mark trade `ambiguous=True`, optionally resolve by `open→high→low→close` interpolation or exclude from metrics / haircut win-rate. Never silently pick SL.
- [ ] **Overnight costs:** implement `swap_long/short × days_held` (incl. triple-Wed), plus `funding_rate` for crypto/CFD and `borrow_fee` for short equities. Even a simple `swap_points_per_day` from YAML beats zero.
- [ ] **Stop-out:** add `margin_call_level%` + `stop_out_level%` (close largest-loss-first, as MT5 does), maintenance margin drift, correct partial-close margin release (`used_margin *= remaining/original`).
- [ ] **Multi-symbol equity history:** record `historical_*` on every base-bar timestamp (not first-symbol-only), with `mark-to-market` using each symbol's latest close. Backfill test: 2-symbol backtest equity must equal sum of single-symbol legs.
- [ ] **Broker-configurable costs:** move Darwinex table to `commission.yaml` / per-run override (like `symbols_info_override`), add `spread_override` + `spread_widen_events` for news testing.

### 7.2 Metrics & diagnostics (highest ROI)

- [ ] Expand `BacktestResults` to institutional tear-sheet: CAGR, vol, Sharpe (Rf=0 + configurable), Sortino, Calmar, MaxDD + duration + recovery, win-rate, profit factor, expectancy (R), avg win/loss, turnover, exposure % , monthly/yearly table, rolling Sharpe/DD, trade-duration histogram, MAE/MFE, slippage-sensitivity table. Keep current `pnl/trades/plot()` for compat.
- [ ] Add `res.summary()` (one-row dict) + `res.tearsheet()` (matplotlib figure) so optimization can rank on `Calmar / Sharpe_net / DD-adjusted return`, not raw PnL.
- [ ] Fix `returns = EQUITY.pct_change()` to handle `equity<=0` and add log-returns option.

### 7.3 API / DX / testing

- [ ] Re-export all 22 indicators in `indicators/__init__.py`; add `__all__`; add `RSI, MACD, ATR, ADX, BollingerBands, Donchian, Keltner, Stochastic, CCI, VWAP` smoke tests.
- [ ] Fix `run_scheduled_taks` typo (keep alias), validate `strategy_id` numeric early with clear error, check `Fixed` volume against min/max/step, deduplicate `HookService`, make export dir explicit (no Desktop default in headless).
- [ ] Add `tests/` + CI: simulator golden tests (market/pending/SLTP/partial/margin-call), multi-symbol equity, anti-lookahead (`get_latest_bars` cutoff), commission math, `RiskPct` rounding-down, live-connector mock tests. Gate releases on them.
- [ ] Pin `quantdle` in `pyproject.toml`, or make it optional (`pip install pyeventbt[quantdle]`).
- [ ] Avoid duplicate log handlers (`if not logger.handlers`), add `VerboseLevel` docs.

---

## 8. New Development Needed for Stated Goal

You want: **strategy creation (data mining, arbitrage, classic) → research → backtesting + robustness → production portfolio.** Here is what to build, in trader terms.

### 8.1 Strategy creation layer

**A. Classic strategy library (`pyeventbt/strategies/`, new)**

Starter pack (each: param dataclass + `make_signal_fn()` + YAML defaults + 1-page doc with in-sample/out-of-sample + turnover):

- Trend: MA-cross (already), Donchian breakout, ATR-trailing (Supertrend/Chandelier), MACD-regime, time-series momentum (12-1 skip).
- Mean-reversion: Bollinger / Keltner z-score, RSI(2) pullback, overnight-gap fade (needs session times).
- Breakout / intraday: opening-range breakout, BB-stop straddle (generalize example), time-stop + daily loss-stop.
- Carry / seasonal: FX carry basket (needs swap — blocked until §7.1), equityIntraday seasonality, end-of-month / turn-of-month.
- Each must set **real SL/TP or time-stop** (examples currently `sl=0`) and expose `HyperParameter` ranges so optimizer can consume them.

**B. Data-mining / ML research harness (`pyeventbt/research/`, new)**

- `FeatureStore`: rolling TA features (all 22 indicators + lags, volatility-normalized returns, session dummies, calendar features once `FXMacroData` is wired) with point-in-time guarantees (reuse `get_latest_bars` cutoff logic; add `asof` join helper).
- `Labeler`: triple-barrier / fixed-horizon / meta-labeling (Lopez de Prado) helpers returning `label, horizon, barrier_hits`.
- `PurgedCV`: purged + embargoed K-fold splitter for time series; ban `ShuffleSplit`.
- `MiningRunner`: exhaustive/random feature-threshold scan with **overfitting guards**: max trials budget, Bonferroni / Benjamini-Hochberg, **Deflated Sharpe Ratio (Bailey & Lopez de Prado)**, **Probabilistic Sharpe**, **White's Reality Check / SPA** stub. Output: `trials.parquet` with `sharpe, dsr, pbo_proxy, dd, turnover, n_trades`.
- Rules: minimum `n_trades >= 100–300`, `t-stat > 3` for mined sharpe, hold-out lockbox never touched until final gate.

**C. Arbitrage scaffolding (`pyeventbt/arb/`, new)**

- Realistic assessment: **true latency arb / cross-broker arb is not viable on single-MT5 retail infra.** Build instead:
  1. **Statistical arbitrage (pairs / basket):** cointegration screen (Engle-Granger / Johansen stub), Kalman hedge-ratio, z-entry/exit, borrow-cost + swap-aware PnL, half-life position cap. Needs multi-symbol sync (already) + short-fee model (new).
  2. **Triangular / cross-rate arb within one MT5 feed:** `EURUSD × GBPUSD vs EURGBP` synthetic vs quoted; model **two-leg slippage + double commission + requote probability** — will usually show edge vanishes; that negative result is valuable.
  3. **Cash-vs-CFD / calendar basis (if data added):** needs second venue data — out of scope until multi-feed `DataProvider` exists.
- Prerequisite infra: multi-feed clock sync, per-leg latency + `deviation` modeling, atomic-pair execution (both legs or neither), borrow/funding ledger. Do not attempt live arb without this.

### 8.2 Research workflow

- `notebooks/` templates: `01_data_check.ipynb` (gaps, spreads, session volumes), `02_single_backtest.ipynb` (tearsheet), `03_parameter_scan.ipynb`, `04_walk_forward.ipynb`, `05_monte_carlo.ipynb`, `06_portfolio_combine.ipynb`.
- `QuantdleDataUpdater` → generalize to `DataUpdater` interface (CSV, MT5 history export, Quantdle, Stooq/Dukascopy stub) with checksum + `DATA_MANIFEST.json` (symbol, TF, rows, start/end, gaps, hash) so research is reproducible.
- Wire `FXMacroDataCalendarConnector` into `DataProvider` as `news_blackout(symbol, minutes_before/after, min_impact)` filter + as feature (`minutes_to_next_red_event`).

### 8.3 Backtesting upgrades (beyond realism fixes)

- `BacktestResults` tear-sheet (§7.2) + trade-level `MAE/MFE`, `slippage attribution` (`gross → net waterfall`), `exposure%`, `risk heatmap`.
- Latency control: `signal_lag_bars` param (default 1, already implicit via next-open fill — make explicit + testable), `allow_same_bar_exit=False` toggle.
- Corporate-action / roll handling if you leave FX: split/dividend adjustment, futures roll calendar.

### 8.4 Robustness suite (`pyeventbt/robustness/`, new — the gate before production)

Every candidate must pass all four; otherwise reject or haircut size:

1. **Parameter stability:** grid ±20% around chosen params; require `Sharpe_net > 0` and `MaxDD < 1.5× base` across neighborhood; plot stability surface. Kills knife-edge overfit.
2. **Walk-forward (anchored + rolling):** implement real `Strategy.walkforward(train_years, test_months, anchored: bool, metric)` using `WalkForwardResults` DTO; report OOS Sharpe decay (`OOS/IS < 0.5 → reject`), retraining timestamps, hyperparameter drift. Commented hyperopt imports suggest this was intended — finish it with `optuna` or `scikit-optimize`, not bare hyperopt.
3. **Monte-Carlo:** trade-reshuffle (bootstrap with replacement, 2000 paths), equity-path resampling, skip-random-trades (e.g. drop 10% worst — proxies live missed fills), slippage ×2 / spread ×2 shocks. Output: `P(ruin), median DD, 95% DD, P(Sharpe<0)`. Size so `95% DD < risk budget`.
4. **Overfitting quantification:** `PBO (Combinatorially Symmetric Cross-Validation)`, `DSR`, number-of-trials haircut. If you mined 1000 variants to find Sharpe 2.0, DSR may be 0.3 — believe DSR.

Add `robustness/report.py:robustness_gate(backtest_results, trades) → PASS/FAIL + reasons` so no strategy reaches portfolio without a signed gate.

### 8.5 Production portfolio of strategies (`pyeventbt/portfolio_book/`, new — biggest gap)

- **Multi-strategy runner:** `PortfolioBook(strategies: list[(Strategy, strategy_id, weight)], allocator)` that runs one shared `DataProvider` clock, routes `BarEvent` to each sub-signal-engine, aggregates `SignalEvents` with `strategy_id/magic` preserved, and produces **combined `BacktestResults`** + per-leg attribution. Today `backtest()` is single-`strategy_id`; extend, don't hack with multiple processes.
- **Capital allocator:** fixed-fractional, volatility-target (inverse-vol), risk-parity, max-DD-scaled, Kelly-fractional (capped 0.25 Kelly). Rebalance monthly/quarterly. Enforce `portfolio_heat = sum(risk%) <= cap` in `RiskEngine` (cross-strategy).
- **Cross-strategy risk overlay (real `RiskEngine`, replaces Passthrough):** global daily/weekly loss halt, max concurrent positions, max per-symbol exposure, correlation brake (if pairwise 60d return corr > 0.8, halve smaller Sharpe leg), news blackout, weekend-flat, margin-headroom buffer (e.g. halt new entries if `free/equity < 30%`).
- **Execution in production:** per-strategy magic, idempotent order placement (survive restart), broker-vs-local reconciliation on `ON_START`, slippage/commission live logging vs backtest assumptions, kill-switch (`disable_trading` already exists — wire to risk overlay + external trigger file/API).
- **Monitoring:** daily email/Telegram digest (equity, DD, per-strategy PnL, fills vs expected, margin %, halt flags), `TradeArchiver` → append-only journal (parquet partitioned by day) + dashboard (Streamlit stub is enough to start).

Portfolio construction rules of thumb to encode:

- Max 3–8 uncorrelated sleeves (trend + MR + breakout + carry); target pairwise |corr| < 0.5.
- No single sleeve > 35% risk budget; portfolio heat ≤ 6–8R open risk.
- Promote to production only after WFA OOS + MC 95% DD within budget + 3-month paper trade matching backtest turnover within 20%.

---

## 9. Suggested Roadmap (Phased)

**Phase 0 — Trust the simulator (1–2 weeks)**
Fix §7.1 margin/partial/equity-history bugs, parametrize commission/spread, add slippage + gap-through-SL handling, add `tests/` golden suite. No new strategies until green.

**Phase 1 — See performance clearly (1 week)**
Full `BacktestResults` tear-sheet + `summary()` + trade MAE/MFE + net-waterfall. Re-run examples; you will likely find MA-cross and BB-straddle are unprofitable net — good, now you have a baseline.

**Phase 2 — Research + robustness harness (2–4 weeks)**
`research/` (features/labels/purged CV) + `robustness/` (param stability, WFA runner, Monte-Carlo, DSR/PBO). Gate: no strategy without `robustness_gate() == PASS`.

**Phase 3 — Strategy factory (ongoing)**
Build `strategies/` library (trend/MR/breakout × FX majors), each with real stops + `HyperParameter` ranges; run WFA + MC; keep 2–3 survivors. Only then attempt pairs-stat-arb pilot (needs borrow/swap model).

**Phase 4 — Portfolio + production (3–6 weeks)**
`portfolio_book/` multi-strategy runner + allocator + cross-strategy risk overlay + reconciliation + kill-switch + daily digest + paper-trade shadow period. Start with 2–3 sleeves, fixed-fractional, tight heat caps. Scale only if live turnover/costs match backtest within tolerance.

**What not to do:** cross-broker latency arb on MT5 retail, tick-HFT on M1 bars, or live deployment with `PassthroughRisk + sl=0` examples.

---

## 10. Appendix: Key Files Index

| Area | Key files |
|---|---|
| Entry / facade | `pyeventbt/__init__.py`, `pyeventbt/strategy/strategy.py`, `pyeventbt/app.py`, `example_ma_crossover.py`, `example_bbands_breakout.py`, `example_quantdle_ma_crossover.py` |
| Events | `pyeventbt/events/events.py` |
| Director / context | `pyeventbt/trading_director/trading_director.py`, `pyeventbt/trading_context/trading_context.py`, `pyeventbt/schedule_service/schedule_service.py`, `pyeventbt/hooks/hook_service.py` |
| Data | `pyeventbt/data_provider/connectors/csv_data_connector.py`, `pyeventbt/data_provider/connectors/mt5_live_data_connector.py`, `pyeventbt/data_provider/services/quantdle_data_updater.py`, `pyeventbt/data_provider/services/data_provider_service.py`, `pyeventbt/data_provider/connectors/fxmacrodata_calendar_connector.py`, `pyeventbt/data_provider/connectors/historical_csv_data/` |
| Signal | `pyeventbt/signal_engine/services/signal_engine_service.py`, `pyeventbt/signal_engine/signal_engines/signal_ma_crossover.py`, `pyeventbt/signal_engine/signal_engines/signal_passthrough.py` |
| Sizing | `pyeventbt/sizing_engine/sizing_engines/mt5_min_sizing.py`, `mt5_fixed_sizing.py`, `mt5_risk_pct_sizing.py`, `pyeventbt/sizing_engine/services/sizing_engine_service.py` |
| Risk | `pyeventbt/risk_engine/risk_engines/passthrough_risk_engine.py`, `pyeventbt/risk_engine/services/risk_engine_service.py` |
| Execution | `pyeventbt/execution_engine/connectors/mt5_simulator_execution_engine_connector.py`, `mt5_live_execution_engine_connector.py`, `pyeventbt/execution_engine/services/execution_engine_service.py`, `pyeventbt/execution_engine/core/configurations/execution_engine_configurations.py` |
| Broker mock | `pyeventbt/broker/mt5_broker/mt5_simulator_wrapper.py`, `connectors/mt5_simulator_connector.py`, `connectors/live_mt5_broker.py`, `shared/shared_data.py`, `shared/default_symbols_info.yaml`, `core/entities/symbol_info.py`, `trade_position.py`, `trade_deal.py`, `trade_request.py`, `order_send_result.py`, `account_info.py`, `tick.py` |
| Portfolio | `pyeventbt/portfolio/portfolio.py`, `pyeventbt/portfolio_handler/portfolio_handler.py`, `pyeventbt/trade_archiver/trade_archiver.py`, `pyeventbt/backtest/core/backtest_results.py` |
| Strategy core | `pyeventbt/strategy/core/strategy_timeframes.py`, `modules.py`, `account_currencies.py`, `walk_forward.py`, `verbose_level.py`, `services/parameter_store.py`, `core/entities/hyper_parameter.py`, `variable.py` |
| Indicators | `pyeventbt/indicators/indicators.py`, `pyeventbt/indicators/__init__.py` |
| Config/utils | `pyeventbt/config/configs.py`, `pyeventbt/utils/utils.py`, `pyproject.toml`, `requirements.txt`, `README.md`, `CHANGELOG.md` |

*End of analysis. Next action recommended: implement Phase 0 simulator fixes + `BacktestResults.summary()` tear-sheet, then re-baseline the two examples net-of-costs before writing any new strategy.*
