from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations

import pandas as pd
import numpy as np

@dataclass(frozen=True)
class PairSpec:
    key: str
    x: str
    y: str
    alpha: float
    beta: float

def LogPrices(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close)

def olsHedge(lx: np.ndarray, ly: np.ndarray) -> tuple[float, float, np.ndarray]:
    A = np.column_stack([np.ones_like(ly), ly])
    coef, *_ = np.linalg.lstsq(A, lx, rcond=None)
    resid = lx - A @ coef
    return float(coef[0]), float(coef[1]), resid

def EngleGranger(lx, ly, trend="c", autolag="aic", maxlag=None) -> tuple[float, float]:
    from statsmodels.tsa.stattools import coint
    stat, p, _ = coint(lx, ly, trend=trend, maxlag=maxlag, autolag=autolag)
    return float(stat), float(p)

def ADFonSpread(resid: np.ndarray, autolag="aic") -> tuple[float, float]:
    from statsmodels.tsa.stattools import adfuller
    out = adfuller(resid, autolag=autolag)
    return float(out[0]), float(out[1])

def HOLM(pvals: np.ndarray, alpha: float = 0.05) -> tuple(np.ndarray, np.ndarray):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj, adj <= alpha

def HalfLife(spread: np.ndarray) -> float:
    s = np.asarray(spread, dtype=float)
    y, x = s[1:], s[:-1]
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    phi = coef[1]
    if 0 < phi < 1:
        return float(np.log(0.5) / np.log(phi))
    return float("nan")

def SpreadSeries(close_x: pd.Series, close_y: pd.Series, alpha: float, beta: float) -> pd.Series:
    return np.log(close_x) - alpha - beta * np.log(close_y)

def Window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    return df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]

def EligibleAssests(close: pd.DataFrame, qvol: pd.DataFrame, start, end, sel_cfg: dict, data_cfg: dict):
    w = Window(close, start, end)
    qv = qvol.loc[w.index]
    rows, ok = [], []
    for a in close.columns:
        cov = float(w[a].notna().mean()) if len(w) else 0.0
        medq = float(qv[a].median()) if len(w) else 0.0
        first_valid = w[a].first_valid_index()
        listed_whole_Window = first_valid is not None and first_valid <= w.index[min(len(w) - 1, 72)]
        reason = None
        if cov < data_cfg["min_coverage"]:
            reason = f"coverage {cov:.3f} < {data_cfg['min_coverage']}"
        elif not listed_whole_Window:
            reason = "Not listed for the whole training window"
        elif not medq >= data_cfg["min_median_quote_volume"]:
            reason = f"median quote volume {medq:,.0f} < {data_cfg['min_median_quote_volume']:,}"
        if reason:
            rows.append(dict(window_start=start, window_end=end, symbol=a, reason=reason))
        else:
            ok.append(a)
    return ok, rows

def SelectPairs(close: pd.DataFrame, qvol: pd.DataFrame, start, end, cfg: dict):
    sel, data_cfg = cfg["selection"], cfg["data"]
    train = Window(close, start, end)
    assets, excl = EligibleAssests(close, qvol, start, end, sel, data_cfg)
    rows = []
    for a, b in combinations(sorted(assets), 2):
        pair = pd.concat([train[a], train[b]], axis=1, keys=["x", "y"]).dropna()
        row = dict(pair=f"{a}-{b}", x=a, y=b, train_start=train.index[0], train_end=train.index[-1],
                   n_obs=len(pair), alpha=np.nan, beta=np.nan, eg_stat=np.nan, p_raw=np.nan,
                   adf_stat_spread=np.nan, adf_p_spread_descriptive=np.nan, half_life_h=np.nan,
                   beta_h1=np.nan, beta_h2=np.nan, coint_trend=sel["coint_trend"], autolag=sel["autolag"],
                   maxlag=sel["maxlag"])
        if len(pair) < sel["min_obs_fraction"] * len(train):
            row["note"] = "Insufficient Aligned Observations"
            rows.append(row)
            continue
        lx, ly = np.log(pair["x"].values), np.log(pair["y"].values)
        row["alpha"], row["beta"], resid = olsHedge(lx, ly)
        h = len(lx) // 2
        row["beta_h1"] = olsHedge(lx[:h], ly[:h])[1]
        row["beta_h2"] = olsHedge(lx[h:], ly[h:])[1]
        row["eg_stat"], row["p_raw"] = EngleGranger(lx, ly, sel["coint_trend"], sel["autolag"], sel["maxlag"])
        row["adf_stat_spread"], row["adf_p_spread_descriptive"] = ADFonSpread(resid, sel["autolag"])
        row["half_life_h"] = HalfLife(resid)
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table, [], pd.DataFrame(excl)
    tested = table["p_raw"].notna()
    table["p_holm"] = np.nan
    table["holm_reject"] = False
    if tested.any():
        adj, rej = HOLM(table.loc[tested, "p_raw"].values, sel["family_alpha"])
        table.loc[tested, "p_holm"] = adj
        table.loc[tested, "holm_reject"] = rej
    decisions, reasons = [], []
    for _, r in table.iterrows():
        why = []
        if not bool(r["holm_reject"]):
            why.append("Holm: fail to reject no-cointegration")
        if not (sel["beta_min"] <= r["beta"] <= sel["beta_max"]):
            why.append(f"beta {r['beta']:.2f} outside [{sel['beta_min']}, {sel['beta_max']}]")
        stable = (np.isfinite(r["beta_h1"]) and np.isfinite(r["beta_h2"]) and r["beta"] != 0 and
                  abs(r["beta_h2"] - r["beta_h1"]) / abs(r["beta"]) <= sel["beta_stability_tol"]
                  and np.sign(r["beta_h1"]) == np.sign(r["beta_h2"]))
        if not stable:
            why.append("beta unstable across training halves")
        hl = r["half_life_h"]
        if not (np.isfinite(hl) and sel["half_life_min_hours"] <= hl <= sel["half_life_max_hours"]):
            why.append(f"half-life {hl:.1f}h outside [{sel['half_life_min_hours']}, {sel['half_life_max_hours']}]")
        decisions.append(len(why) == 0)
        reasons.append("; ".join(why))
    table["passes_filters"] = decisions
    table["reject_reasons"] = reasons
    table = table.sort_values(["passes_filters", "p_holm"], ascending=[False, True]).reset_index(drop=True)
    table["selected"] = False
    top = table.index[table["passes_filters"]][: sel["max_pairs"]]
    table.loc[top, "selected"] = True
    pairs = [PairSpec(key=r["pair"], x=r["x"], y=r["y"], alpha=float(r["alpha"]), beta=float(r["beta"]))
             for _, r in table[table["selected"]].iterrows()]
    return table, pairs, pd.DataFrame(excl)