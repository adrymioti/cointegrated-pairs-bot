from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

from .Backtest import run_backtest, Window
from .Config import ParamsFromCFG
from .Metrics import Performance, BreakEvenCost


def Summ(res, initial, K):
    perf = Performance(res.equity["equity"], initial, K) if len(res.equity) else {}
    return dict(n_trades=len(res.ledger), net_return=perf.get("total_return", np.nan),
                sharpe=perf.get("sharpe", np.nan), max_dd=perf.get("max_drawdown", np.nan),
                net_pnl=float(res.ledger["net_pnl"].sum()) if len(res.ledger) else 0.0)


def SenstivityGrid(open_df, close_df, window: Window, cfg: dict) -> pd.DataFrame:
    """Declared grid on development/validation data only (never the final test)."""
    r = cfg["robustness"]
    init = cfg["portfolio"]["initial_equity"]
    K = cfg["portfolio"]["hours_per_year"]
    rows = []
    for L, ent, ex in itertools.product(r["lookback_grid"], r["entry_grid"], r["exit_grid"]):
        p = ParamsFromCFG(cfg, lookback=L, entry_z=ent, exit_z=ex)
        res = run_backtest(open_df, close_df, [window], p, init)
        rows.append(dict(lookback=L, entry_z=ent, exit_z=ex, **Summ(res, init, K)))
    return pd.DataFrame(rows)


def CostStress(open_df, close_df, windows, cfg, bps_list=None, **over) -> pd.DataFrame:
    """Total fee+slippage per executed notional (split 50/50). Borrow cost kept separate and unchanged."""
    init, K = cfg["portfolio"]["initial_equity"], cfg["portfolio"]["hours_per_year"]
    rows = []
    for b in bps_list or cfg["robustness"]["cost_grid_bps"]:
        p = ParamsFromCFG(cfg, fee_bps=b / 2, slippage_bps=b / 2, **over)
        res = run_backtest(open_df, close_df, windows, p, init)
        rows.append(dict(total_cost_bps=b, **Summ(res, init, K)))
    return pd.DataFrame(rows)


def BreakEven(stress: pd.DataFrame):
    return BreakEvenCost(stress["total_cost_bps"].tolist(), stress["net_return"].tolist())


def DelaySensitivity(open_df, close_df, windows, cfg, delays=(0, 1, 2)) -> pd.DataFrame:
    init, K = cfg["portfolio"]["initial_equity"], cfg["portfolio"]["hours_per_year"]
    rows = []
    for d in delays:
        res = run_backtest(open_df, close_df, windows, ParamsFromCFG(cfg, delay_bars=d), init)
        rows.append(dict(delay_bars=d, **Summ(res, init, K)))
    return pd.DataFrame(rows)


def LegFailureStress(open_df, close_df, windows, cfg, probs=(0.0, 0.1, 0.25)) -> pd.DataFrame:
    init, K = cfg["portfolio"]["initial_equity"], cfg["portfolio"]["hours_per_year"]
    rows = []
    for pr in probs:
        res = run_backtest(open_df, close_df, windows, ParamsFromCFG(cfg, leg_fail_prob=pr), init)
        rows.append(dict(leg_fail_prob=pr, **Summ(res, init, K)))
    return pd.DataFrame(rows)


def Subperiods(open_df, close_df, window: Window, cfg, n=2) -> pd.DataFrame:
    """Same frozen pairs, equal-length consecutive sub-windows (each starts flat with fresh capital)."""
    init, K = cfg["portfolio"]["initial_equity"], cfg["portfolio"]["hours_per_year"]
    edges = pd.date_range(window.start, window.end, periods=n + 1)
    rows = []
    for i in range(n):
        w = Window(f"{window.name}_part{i + 1}", edges[i], edges[i + 1], window.pairs)
        res = run_backtest(open_df, close_df, [w], ParamsFromCFG(cfg), init)
        rows.append(dict(part=i + 1, start=edges[i], end=edges[i + 1], **Summ(res, init, K)))
    return pd.DataFrame(rows)
