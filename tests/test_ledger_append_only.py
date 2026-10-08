"""Tests Etapa 0: ledger append-only + S_global + un solo uso del lockbox."""
import pytest

from pyeventbt.research.ledger import append_trials, read_ledger, register_lockbox_use, s_global


def _row(trial, window="IS[2021-01-03,2023-12-31]", sharpe=0.5):
    return {
        "run_id": "run1", "strategy_family": "bbands_breakout", "combo": "AUDCAD_H1",
        "trial_number": trial, "params_hash": f"h{trial}", "window": window,
        "sharpe_net": sharpe, "n_trades": 120, "daily_returns_ref": f"dr/{trial}.parquet",
    }


def test_ledger_append_only(tmp_path):
    p = tmp_path / "ledger.parquet"
    append_trials(p, [_row(0), _row(1)])
    assert len(read_ledger(p)) == 2
    append_trials(p, [_row(2)])
    df = read_ledger(p)
    assert len(df) == 3  # nunca encoge, solo añade
    assert sorted(df["trial_number"].to_list()) == [0, 1, 2]


def test_s_global_counts_program_trials_on_same_window(tmp_path):
    p = tmp_path / "ledger.parquet"
    append_trials(p, [_row(0), _row(1)])
    append_trials(p, [_row(0, window="OTHER")])
    assert s_global(p, "IS[2021-01-03,2023-12-31]") == 2


def test_lockbox_single_use(tmp_path):
    reg = tmp_path / "lockbox_uses.yaml"
    entry = {"family": "bbands_breakout", "symbol": "AUDCAD", "timeframe": "H1", "set_id": "s01"}
    register_lockbox_use(reg, entry)
    with pytest.raises(RuntimeError):
        register_lockbox_use(reg, entry)  # segundo uso => run contaminado
