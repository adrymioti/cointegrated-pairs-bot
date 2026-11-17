from __future__ import annotations
import numpy as np
import pandas as pd

from .Metrics import DrawDownPath, SimpleReturns


def RegimeLabels(close: pd.DataFrame, benchmark: str, window: int, train_end) -> tuple[pd.Series, dict]:
    """vol regime: trailing std of hourly benchmark returns vs. its TRAINING median.
    trend regime: trailing `window`-hour benchmark return > 0 (rising) or <= 0 (falling).
    Both are shifted by one bar so the label at bar t uses information through bar t-1 only."""
    lr = np.log(close[benchmark]).diff()
    vol = lr.rolling(window, min_periods=window).std().shift(1)
    trend = np.log(close[benchmark]).diff(window).shift(1)
    thr = float(vol[vol.index < train_end].median())
    lab = pd.Series(np.where(vol > thr, "high_vol", "low_vol"), index=close.index) + "|" + \
        pd.Series(np.where(trend > 0, "rising", "falling"), index=close.index)
    lab[vol.isna() | trend.isna()] = np.nan
    return lab, dict(vol_threshold=thr, window=window)


def RegimeTable(res, labels: pd.Series, initial: float, K: int) -> pd.DataFrame:
    eq = res.equity
    r = SimpleReturns(eq["equity"], initial)
    lab = labels.reindex(r.index)
    rows = []
    led = res.ledger
    for reg in sorted(lab.dropna().unique()):
        m = lab == reg
        rr = r[m]
        path = (1 + rr).cumprod()
        dd = float(DrawDownPath(pd.concat([pd.Series([1.0]), path.reset_index(drop=True)])).max()) if len(rr) else np.nan
        trades = led[pd.Series(led["fill_time_entry"]).map(lambda t: lab.get(pd.Timestamp(t), np.nan)) == reg] \
            if len(led) else led
        rows.append(dict(regime=reg, hours=int(m.sum()), share_of_hours=float(m.mean()),
                         net_return_sum=float(rr.sum()), mean_hourly_return=float(rr.mean()) if len(rr) else np.nan,
                         max_drawdown_within_regime=dd, trades_entered=len(trades),
                         net_pnl_of_entered_trades=float(trades["net_pnl"].sum()) if len(trades) else 0.0,
                         mean_hold_hours=float(trades["hold_bars"].mean()) if len(trades) else np.nan,
                         exposure=float((eq["gross"].reindex(rr.index) > 0).mean()) if len(rr) else np.nan))
    return pd.DataFrame(rows)


def RollingDiagnostics(close: pd.DataFrame, pairs, window: int = 2160, step: int = 168, trend="c") -> pd.DataFrame:
    """EXPLORATORY: rolling OLS beta and rolling Engle-Granger p-value on historical windows.
    Repeated p-values are not independent confirmations of a stable relationship."""
    from .Cointegration import olsHedge, EngleGranger
    rows = []
    for sp in pairs:
        d = close[[sp.x, sp.y]].dropna()
        for end in range(window, len(d) + 1, step):
            w = d.iloc[end - window:end]
            lx, ly = np.log(w[sp.x].values), np.log(w[sp.y].values)
            a, b, _ = olsHedge(lx, ly)
            try:
                _, p = EngleGranger(lx, ly, trend=trend)
            except Exception:
                p = np.nan
            rows.append(dict(pair=sp.key, time=w.index[-1], beta=b, p_raw=p))
    return pd.DataFrame(rows)