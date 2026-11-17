import numpy as np
import pandas as pd

from src.Backtest import Engine, Window, run_backtest
from src.Config import Params
from src.Signals import SizeLegs, Decide
from tests.Helpers import SPEC, bar, clean_params, random_cointegrated, ts

def test_position_sizing_hand_example():
    qx, qy = SizeLegs(1000.0, 1.5, px=1.0, py=1.0, side=+1)
    assert abs(qx - 400) < 1e-9 and abs(qy + 600) < 1e-9          
 
 
def test_hand_calculated_long_short_trade_with_fees():
    e = Engine(clean_params(), 100_000.0)
    bar(e, 0, (100, 50), (100, 50), z=2.5)           
    bar(e, 1, (100, 50), (98, 51), z=0.1)            
    bar(e, 2, (95, 52), (95, 52), z=np.nan)         
    t = e.ledger[0]
    assert abs(t["qx"] + 80) < 1e-9 and abs(t["qy"] - 240) < 1e-9
    assert abs(t["gross_pnl"] - 880.0) < 1e-9       
    assert abs(t["fees"] - (20.0 + 20.08)) < 1e-9    
    assert abs(t["net_pnl"] - 839.92) < 1e-9
    assert abs(e.equity_last - 100_839.92) < 1e-6    
 
 
def test_no_trade_case():
    e = Engine(clean_params(), 100_000.0)
    for i in range(20):
        bar(e, i, (100, 50), (100, 50), z=0.3)
    assert len(e.ledger) == 0 and len(e.pos) == 0
    assert abs(e.equity_last - 100_000.0) < 1e-9
 
 
def test_missing_data_block():
    e = Engine(clean_params(), 100_000.0)
    bar(e, 0, (100, 50), (100, 50), z=2.5)
    bar(e, 1, (100, np.nan), (100, np.nan), z=0.0)   
    assert len(e.pos) == 0
    reasons = [b["reason"] for b in e.blocked]
    assert "missing_price" in reasons and "signal_suspended_missing_data" in reasons
 
 
def test_oversized_order_rejected_not_scaled():
    e = Engine(clean_params(pair_gross_frac=0.5), 100_000.0)     
    bar(e, 0, (100, 50), (100, 50), z=2.5)
    bar(e, 1, (100, 50), (100, 50), z=2.4)
    assert len(e.pos) == 0 and e.blocked[-1]["reason"] == "pair_limit"
 
 
def test_asset_limit_aggregates_across_pairs():
    from src.Cointegration import PairSpec
    s2 = PairSpec("X-Z", "X", "Z", 0.0, 1.0)
    e = Engine(clean_params(max_asset=0.15), 100_000.0)          
    for i, z in enumerate([2.5, 2.4]):
        e.process_bar(ts(i), dict(X=100, Y=50, Z=20), dict(X=100, Y=50, Z=20), [SPEC, s2], {"X-Y": z, "X-Z": z})
    assert any(b["reason"] == "asset_limit" for b in e.blocked)
 
 
def test_stop_and_no_entry_beyond_stop():
    e = Engine(clean_params(), 100_000.0)
    bar(e, 0, (100, 50), (100, 50), z=3.8)           
    bar(e, 1, (100, 50), (100, 50), z=3.8)
    assert len(e.pos) == 0 and not e.pending
 
 
def test_no_same_bar_reentry():
    e = Engine(clean_params(), 100_000.0)
    bar(e, 0, (100, 50), (100, 50), z=2.5)
    bar(e, 1, (100, 50), (100, 50), z=0.1)           
    bar(e, 2, (100, 50), (100, 50), z=2.6)           
    assert len(e.ledger) == 1 and "X-Y" not in e.pending
 
 
def test_equity_identity_with_all_costs_on_random_data():
    opn, close = random_cointegrated()
    p = Params(fee_bps=5, slippage_bps=5, borrow_rate_annual=0.10)
    from src.Cointegration import olsHedge
    a, b, _ = olsHedge(np.log(close["X"].values), np.log(close["Y"].values))
    from src.Cointegration import PairSpec
    w = Window("t", close.index[0], close.index[-1] + pd.Timedelta(hours=1), [PairSpec("X-Y", "X", "Y", a, b)])
    r = run_backtest(opn, close, [w], p, 100_000.0)
    assert len(r.ledger) > 5
    assert abs(r.ledger["net_pnl"].sum() - (r.equity["equity"].iloc[-1] - 100_000.0)) < 1e-6
    assert (r.ledger["fees"] > 0).all() and (r.ledger["borrow"] > 0).all()
    assert abs((r.ledger["gross_pnl"] - r.ledger["fees"] - r.ledger["slippage"] - r.ledger["borrow"]
                - r.ledger["net_pnl"]).abs().max()) < 1e-9
 
 
def test_future_prices_cannot_change_earlier_results():
    opn, close = random_cointegrated()
    from src.Cointegration import olsHedge, PairSpec
    a, b, _ = olsHedge(np.log(close["X"].values[:1500]), np.log(close["Y"].values[:1500]))
    spec = PairSpec("X-Y", "X", "Y", a, b)
    p = Params()
    N = 2000
    full = run_backtest(opn, close, [Window("f", close.index[0], close.index[-1] + pd.Timedelta(hours=1), [spec])], p)
    cut_close, cut_open = close.copy(), opn.copy()
    cut_close.iloc[N:] = cut_close.iloc[N:] * np.random.default_rng(0).uniform(0.5, 2, size=cut_close.iloc[N:].shape)
    cut_open.iloc[N:] = cut_open.iloc[N:] * 1.3
    alt = run_backtest(cut_open, cut_close, [Window("f", close.index[0], close.index[-1] + pd.Timedelta(hours=1), [spec])], p)
    e1, e2 = full.equity["equity"].iloc[:N - 1], alt.equity["equity"].iloc[:N - 1]
    assert np.allclose(e1.values, e2.values)
 
 
def test_duplicate_or_out_of_order_bar_ignored():
    e = Engine(clean_params(), 100_000.0)
    assert bar(e, 5, (100, 50), (100, 50), z=0.0) is True
    assert bar(e, 5, (100, 50), (100, 50), z=0.0) is False
    assert bar(e, 4, (100, 50), (100, 50), z=0.0) is False
 
 
def test_drawdown_halt_flattens_and_blocks_entries():
    e = Engine(clean_params(dd_halt=0.0001, halt_cooldown_bars=50), 100_000.0)
    bar(e, 0, (100, 50), (100, 50), z=2.5)
    bar(e, 1, (100, 50), (100, 50), z=2.4)           
    assert e.halted
    bar(e, 2, (100, 50), (100, 50), z=2.5)
    assert len(e.pos) == 0 and any(l["exit_reason"] == "drawdown_halt" for l in e.ledger)
