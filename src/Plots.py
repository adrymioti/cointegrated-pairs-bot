from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .Metrics import DrawDownPath


def Save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def PriceHistory(close, path, label=""):
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    (close / close.bfill().iloc[0]).plot(ax=ax[0], lw=0.7, legend=False)
    ax[0].set_ylabel("Price / first price (x)"); ax[0].set_title(f"Normalised prices {label}")
    np.log(close).plot(ax=ax[1], lw=0.7, legend=False)
    ax[1].set_ylabel("ln(price, USDT)"); ax[1].set_xlabel("UTC time")
    Save(fig, path)


def RollingVol(close, path, window=168):
    fig, ax = plt.subplots(figsize=(10, 4))
    (np.log(close).diff().rolling(window).std() * 100).plot(ax=ax, lw=0.6, legend=False)
    ax.set_ylabel(f"{window}h rolling st.dev. of hourly log return (%)"); ax.set_xlabel("UTC time")
    ax.set_title("Rolling volatility")
    Save(fig, path)


def CorrMatrix(close, path):
    r = np.log(close).diff().dropna(how="all")
    c = r.corr(min_periods=500)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(c.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=90)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index)
    fig.colorbar(im, ax=ax, label="Correlation of hourly log returns")
    ax.set_title("Return correlation (co-movement, NOT cointegration)")
    Save(fig, path)
    return c


def SplitDiagram(split, wf_windows, path):
    fig, ax = plt.subplots(figsize=(10, 2.6))
    segs = [("Development (train/select)", split["dev_start"], split["val_start"], "#4c78a8"),
            ("Validation (grid)", split["val_start"], split["test_start"], "#f58518"),
            ("Final test (frozen + walk-forward)", split["test_start"], split["test_end"], "#54a24b")]
    for name, a, b, c in segs:
        ax.barh(0, (b - a).total_seconds() / 86400, left=matplotlib.dates.date2num(a), color=c, label=name)
    ax.xaxis_date(); ax.set_yticks([]); ax.set_xlabel("UTC date"); ax.legend(loc="upper center", ncol=3, fontsize=8)
    ax.set_title("Chronological split (no shuffling)")
    Save(fig, path)


def SpreadWithTrades(close, spec, z, ledger, path, entry=2.0, exit_=0.5, stop=3.5, start=None, end=None):
    from .Signals import Spread
    s = Spread(close[spec.x], close[spec.y], spec.alpha, spec.beta)
    sl = slice(start, end)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ss, zz = s.loc[sl], z.loc[sl]
    ax[0].plot(ss.index, ss.values, lw=0.6); ax[0].set_ylabel("Spread: ln Px - a - b ln Py")
    ax[0].set_title(f"{spec.key}: spread and z-score (trades marked)")
    ax[1].plot(zz.index, zz.values, lw=0.6)
    for lv in (entry, -entry): ax[1].axhline(lv, color="g", ls="--", lw=0.7)
    for lv in (exit_, -exit_): ax[1].axhline(lv, color="gray", ls=":", lw=0.7)
    for lv in (stop, -stop): ax[1].axhline(lv, color="r", ls="--", lw=0.7)
    if ledger is not None and len(ledger):
        L = ledger[ledger["pair"] == spec.key]
        for _, t in L.iterrows():
            for col, mk, c in (("fill_time_entry", "^", "g"), ("fill_time_exit", "v", "r")):
                ts = pd.Timestamp(t[col])
                if (start is None or ts >= pd.Timestamp(start)) and (end is None or ts <= pd.Timestamp(end)):
                    zi = z.index.get_indexer([ts], method="nearest")[0]
                    ax[1].plot(z.index[zi], z.iloc[zi], mk, color=c, ms=5)
    ax[1].set_ylabel("z-score"); ax[1].set_xlabel("UTC time")
    Save(fig, path)


def EquityDrawdown(curves: dict, path, title="Equity and drawdown"):
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for name, s in curves.items():
        (s / s.iloc[0]).plot(ax=ax[0], lw=1, label=name)
        DrawDownPath(s).mul(100).plot(ax=ax[1], lw=0.8, label=name)
    ax[0].set_ylabel("Equity / initial equity (x)"); ax[0].legend(); ax[0].set_title(title)
    ax[1].invert_yaxis(); ax[1].set_ylabel("Drawdown (%)"); ax[1].set_xlabel("UTC time")
    Save(fig, path)


def Heatmaps(grid: pd.DataFrame, path, value="sharpe"):
    Ls = sorted(grid["lookback"].unique())
    fig, axes = plt.subplots(1, len(Ls), figsize=(4.6 * len(Ls), 4.2), squeeze=False)
    vmax = np.nanmax(np.abs(grid[value])) if grid[value].notna().any() else 1
    for ax, L in zip(axes[0], Ls):
        g = grid[grid["lookback"] == L]
        pv = g.pivot(index="entry_z", columns="exit_z", values=value)
        pn = g.pivot(index="entry_z", columns="exit_z", values="n_trades")
        im = ax.imshow(pv.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns)
        ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v, n = pv.values[i, j], pn.values[i, j]
                ax.text(j, i, f"{v:.1f}\n(n={int(n)})" if np.isfinite(v) else f"n/a\n(n={int(n)})",
                        ha="center", va="center", fontsize=7)
        ax.set_title(f"lookback {L}h"); ax.set_xlabel("exit |z| (0 = mean cross)"); ax.set_ylabel("entry |z|")
    fig.suptitle(f"Validation-window net {value} (n = trade count). Declared grid; all cells shown.")
    Save(fig, path)


def Stability(roll: pd.DataFrame, path):
    pairs = roll["pair"].unique() if len(roll) else []
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for p in pairs:
        d = roll[roll["pair"] == p]
        ax[0].plot(d["time"], d["beta"], label=p); ax[1].plot(d["time"], d["p_raw"], label=p)
    ax[0].set_ylabel("Rolling OLS hedge ratio b"); ax[0].legend(fontsize=7)
    ax[1].axhline(0.05, color="r", ls="--", lw=0.7); ax[1].set_ylabel("Rolling Engle-Granger p (exploratory)")
    ax[1].set_xlabel("UTC time (window end)")
    ax[0].set_title("Relationship stability (historical rolling windows)")
    Save(fig, path)
