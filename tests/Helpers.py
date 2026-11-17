import numpy as np
import pandas as pd
from src.Config import Params
from src.Cointegration import PairSpec
from src.Backtest import Engine

T0 = pd.Timestamp("2024-01-01", tz="UTC")
SPEC = PairSpec("X-Y", "X", "Y", 0.0, 1.5)


def ts(i):
    return T0 + pd.Timedelta(hours=i)


def clean_params(**kw):
    base = dict(fee_bps=10.0, slippage_bps=0.0, borrow_rate_annual=0.0, pair_gross_frac=0.2, max_pair=0.2,
                max_asset=0.3, max_gross=1.0, dd_halt=0.9)
    base.update(kw)
    return Params(**base)


def bar(eng, i, o, c, z, pairs=(SPEC,)):
    return eng.process_bar(ts(i), dict(X=o[0], Y=o[1]), dict(X=c[0], Y=c[1]), list(pairs), {"X-Y": z})


def random_cointegrated(n=2500, seed=3):
    rng = np.random.default_rng(seed)
    ly = np.cumsum(rng.normal(0, 0.004, n)) + np.log(50)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.96 * s[t - 1] + rng.normal(0, 0.01)
    lx = 0.2 + 1.2 * ly + s
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = pd.DataFrame({"X": np.exp(lx), "Y": np.exp(ly)}, index=idx)
    opn = close.shift(1)
    opn.iloc[0] = close.iloc[0]
    return opn, close