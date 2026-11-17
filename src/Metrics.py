from __future__ import annotations
import numpy as np
import pandas as pd


def SimpleReturns(equity: pd.Series, initial: float | None = None) -> pd.Series:
    eq = equity.astype(float)
    if initial is not None:
        eq = pd.concat([pd.Series([initial], index=[eq.index[0] - pd.Timedelta(hours=1)]), eq])
    return eq.pct_change().dropna()


def DrawDownPath(equity: pd.Series) -> pd.Series:
    return 1.0 - equity / equity.cummax()


def Sharpe(r: pd.Series, K: int = 24 * 365, rf_annual: float = 0.0) -> float:
    ex = r - rf_annual / K
    sd = ex.std(ddof=1)
    return float(np.sqrt(K) * ex.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")


def Performance(equity: pd.Series, initial: float, K: int = 24 * 365, rf_annual: float = 0.0) -> dict:
    if len(equity) < 3:
        return {}
    r = SimpleReturns(equity, initial)
    dd = DrawDownPath(pd.concat([pd.Series([initial], index=[equity.index[0] - pd.Timedelta(hours=1)]), equity]))
    return dict(n_hours=len(r), start=equity.index[0], end=equity.index[-1],
                total_return=float(equity.iloc[-1] / initial - 1),
                ann_vol=float(r.std(ddof=1) * np.sqrt(K)),
                sharpe=Sharpe(r, K, rf_annual),
                max_drawdown=float(dd.max()),
                mean_hourly_return=float(r.mean()),
                note="Sharpe uses sqrt(K) scaling; serial correlation can make it misleading")


def TradeDiagnostics(ledger: pd.DataFrame, eq_df: pd.DataFrame) -> dict:
    if ledger is None or len(ledger) == 0:
        return dict(n_trades=0, win_rate=np.nan, mean_win=np.nan, mean_loss=np.nan, mean_net_pnl=np.nan,
                    mean_hold_hours=np.nan, exposure=float((eq_df["gross"] > 0).mean()) if len(eq_df) else 0.0,
                    turnover=0.0)
    net = ledger["net_pnl"]
    wins, losses = net[net > 0], net[net <= 0]
    avg_eq = eq_df["equity"].mean()
    return dict(n_trades=int(len(ledger)), win_rate=float((net > 0).mean()),
                mean_win=float(wins.mean()) if len(wins) else np.nan,
                mean_loss=float(losses.mean()) if len(losses) else np.nan,
                mean_net_pnl=float(net.mean()), mean_hold_hours=float(ledger["hold_bars"].mean()),
                exposure=float((eq_df["gross"] > 0).mean()),
                turnover=float(ledger["traded_notional"].sum() / avg_eq))


def GrossNetTable(res, initial: float, K: int = 24 * 365) -> pd.DataFrame:
    eq = res.equity
    net = eq["equity"]
    gross = eq["equity"] + eq["cum_costs"]
    out = {}
    for name, series in (("gross", gross), ("net", net)):
        out[name] = Performance(series, initial, K)
    t = pd.DataFrame(out).T
    d = TradeDiagnostics(res.ledger, eq)
    for k, v in d.items():
        t[k] = [v, v]
    t["fees"] = [0, eq["cum_fees"].iloc[-1]] if len(eq) else [0, 0]
    t["slippage"] = [0, eq["cum_slippage"].iloc[-1]] if len(eq) else [0, 0]
    t["borrow"] = [0, eq["cum_borrow"].iloc[-1]] if len(eq) else [0, 0]
    return t


def BuyAndHold(close: pd.DataFrame, start, end, initial: float, assets=None) -> pd.Series:
    w = close[(close.index >= start) & (close.index < end)]
    if assets is not None:
        w = w[assets]
    w = w.dropna(axis=1, how="all")
    w = w.loc[:, w.iloc[0].notna()].ffill()
    rel = w / w.iloc[0]
    return initial * rel.mean(axis=1)


def BlockBoostrap(r: pd.Series, K: int, block: int = 168, reps: int = 1000, seed: int = 0) -> dict:
    x = r.values
    n = len(x)
    if n < 2 * block:
        return {}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    sh, tr = [], []
    for _ in range(reps):
        starts = rng.integers(0, n, nb)
        samp = np.concatenate([np.take(x, np.arange(s, s + block), mode="wrap") for s in starts])[:n]
        sd = samp.std(ddof=1)
        sh.append(np.sqrt(K) * samp.mean() / sd if sd > 0 else np.nan)
        tr.append(np.prod(1 + samp) - 1)
    q = lambda a: tuple(float(v) for v in np.nanpercentile(a, [5, 95]))
    return dict(sharpe_5=q(sh)[0], sharpe_95=q(sh)[1], total_return_5=q(tr)[0], total_return_95=q(tr)[1],
                block=block, reps=reps)


def BreakEvenCost(bps: list, net_returns: list):
    b, r = np.asarray(bps, float), np.asarray(net_returns, float)
    if r[0] <= 0:
        return 0.0                      # unprofitable even with zero cost
    for i in range(1, len(b)):
        if r[i] <= 0:
            return float(b[i - 1] + (b[i] - b[i - 1]) * r[i - 1] / (r[i - 1] - r[i]))
    return float("inf")                 # still profitable at the largest tested cost