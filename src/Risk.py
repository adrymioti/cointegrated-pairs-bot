from __future__ import annotations
from collections import defaultdict
import numpy as np
 
from .Config import Params
 
 
def GrossByAsset(positions: dict, marks: dict) -> dict:
    g = defaultdict(float)
    for pos in positions.values():
        g[pos["x"]] += abs(pos["qx"]) * marks.get(pos["x"], 0.0)
        g[pos["y"]] += abs(pos["qy"]) * marks.get(pos["y"], 0.0)
    return g
 
 
def ShortNotational(positions: dict, marks: dict) -> float:
    tot = 0.0
    for pos in positions.values():
        for a, q in ((pos["x"], pos["qx"]), (pos["y"], pos["qy"])):
            if q < 0:
                tot += abs(q) * marks.get(a, 0.0)
    return tot
 
 
def CheckEntry(p: Params, equity: float, cash: float, positions: dict, marks: dict, legs: list):
    if not np.isfinite(equity) or equity <= 0:
        return False, "no_equity", ""
    new_gross = sum(abs(q) * px for _, q, px in legs)
    cur = GrossByAsset(positions, marks)
    tol = 1e-9 * equity
    if new_gross > p.max_pair * equity + tol:
        return False, "pair_limit", f"{new_gross:.2f} > {p.max_pair * equity:.2f}"
    if sum(cur.values()) + new_gross > p.max_gross * equity + tol:
        return False, "gross_limit", f"{sum(cur.values()) + new_gross:.2f} > {p.max_gross * equity:.2f}"
    for a, q, px in legs:
        if cur.get(a, 0.0) + abs(q) * px > p.max_asset * equity + tol:
            return False, "asset_limit", f"{a}: {cur.get(a, 0.0) + abs(q) * px:.2f} > {p.max_asset * equity:.2f}"
    cash_after = cash - sum(q * px for _, q, px in legs)
    short_after = ShortNotational(positions, marks) + sum(abs(q) * px for _, q, px in legs if q < 0)
    if cash_after - p.collateral_ratio * short_after < 0:
        return False, "insufficient_collateral", f"free cash {cash_after - p.collateral_ratio * short_after:.2f}"
    return True, "", ""