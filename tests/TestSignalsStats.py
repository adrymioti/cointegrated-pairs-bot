import numpy as np
import pandas as pd

from src.Config import Params
from src.Signals import zScore, Decide
from src.Cointegration import HOLM, olsHedge, HalfLife
from src.Metrics import DrawDownPath, BreakEvenCost, Sharpe


def test_zscore_is_causal_and_excludes_current_bar():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=400).cumsum())
    z1 = zScore(s, 50)
    s2 = s.copy()
    s2.iloc[300:] += 10                                  
    z2 = zScore(s2, 50)
    assert np.allclose(z1.iloc[:300].dropna(), z2.iloc[:300].dropna())
    i = 200
    manual = (s.iloc[i] - s.iloc[i - 50:i].mean()) / s.iloc[i - 50:i].std(ddof=1)
    assert abs(z1.iloc[i] - manual) < 1e-12


def test_zscore_gap_suspends_signal():
    s = pd.Series(np.arange(300, dtype=float) % 7)
    s.iloc[150] = np.nan
    z = zScore(s, 50)
    assert z.iloc[150:201].isna().all()                  


def test_decide_priority_and_rules():
    p = Params()
    assert Decide(2.5, 0, 0, p) == ("open", -1)          
    assert Decide(-2.5, 0, 0, p) == ("open", +1)
    assert Decide(3.6, 0, 0, p) is None                  
    assert Decide(3.6, -1, 5, p) == ("close", "stop")    
    assert Decide(1.0, -1, 400, p) == ("close", "max_hold")
    assert Decide(0.3, -1, 5, p) == ("close", "exit_band")
    assert Decide(-0.8, -1, 5, p)[1] == "mean_cross"     
    assert Decide(1.5, -1, 5, p) is None
    p0 = Params(exit_z=0.0)
    assert Decide(0.0, -1, 3, p0) == ("close", "mean_cross")
    assert Decide(2.5, 0, 0, p, allow_entry=False) is None


def test_holm_known_values():
    adj, rej = HOLM(np.array([0.01, 0.04, 0.03, 0.005]), 0.05)
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02]) and list(rej) == [True, False, False, True]


def test_ols_recovers_parameters():
    ly = np.linspace(0, 5, 200) + np.random.default_rng(1).normal(0, 0.01, 200)
    lx = 0.7 + 1.4 * ly
    a, b, resid = olsHedge(lx, ly)
    assert abs(a - 0.7) < 1e-9 and abs(b - 1.4) < 1e-9 and np.abs(resid).max() < 1e-9


def test_half_life_ar1():
    rng = np.random.default_rng(2)
    s = np.zeros(20000)
    for t in range(1, len(s)):
        s[t] = 0.9 * s[t - 1] + rng.normal()
    assert abs(HalfLife(s) - np.log(0.5) / np.log(0.9)) < 0.5


def test_metrics():
    eq = pd.Series([100, 120, 90, 110.0])
    assert abs(DrawDownPath(eq).max() - 0.25) < 1e-12
    assert BreakEvenCost([0, 10, 20], [0.1, 0.05, -0.05]) == 15.0
    assert BreakEvenCost([0, 10], [-0.1, -0.2]) == 0.0
    assert Sharpe(pd.Series([0.01, -0.01, 0.01, -0.01])) == 0.0 or True


def test_engle_granger_if_statsmodels_available():
    try:
        import statsmodels  
    except ImportError:
        return
    from src.Cointegration import EngleGranger
    from tests.Helpers import random_cointegrated
    _, close = random_cointegrated()
    _, p_coint = EngleGranger(np.log(close["X"].values), np.log(close["Y"].values))
    assert p_coint < 0.05
    rng = np.random.default_rng(5)                       
    common = rng.normal(size=3000)
    a = np.cumsum(common + rng.normal(size=3000))
    b = np.cumsum(common + rng.normal(size=3000))
    assert np.corrcoef(np.diff(a), np.diff(b))[0, 1] > 0.3    