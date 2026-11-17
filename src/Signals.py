from __future__ import annotations
import numpy as np
import pandas as pd
 
from .Config import Params
 
 
def Spread(close_x: pd.Series, close_y: pd.Series, alpha: float, beta: float) -> pd.Series:
    return np.log(close_x) - alpha - beta * np.log(close_y)
 
 
def zScore(s: pd.Series, lookback: int, eps: float = 1e-12) -> pd.Series:
    r = s.rolling(lookback, min_periods=lookback)
    mu = r.mean().shift(1)
    sd = r.std(ddof=1).shift(1)
    z = (s - mu) / sd
    return z.where(sd > eps)
 
 
def Decide(z: float, side: int, held_bars: int, p: Params, allow_entry: bool = True):
    if z is None or not np.isfinite(z):
        if side != 0 and held_bars >= p.max_hold_bars:
            return ("close", "max_hold")
        return None
    if side != 0:
        if abs(z) > p.stop_z:
            return ("close", "stop")
        if held_bars >= p.max_hold_bars:
            return ("close", "max_hold")
        if p.exit_z > 0 and abs(z) < p.exit_z:
            return ("close", "exit_band")
        crossed = (side == +1 and z >= 0) or (side == -1 and z <= 0)
        if crossed:          # covers exit_z == 0 and a jump across the mean that skips the exit band
            return ("close", "mean_cross")
        return None
    if allow_entry and p.entry_z <= abs(z) <= p.stop_z:
        return ("open", -1 if z > 0 else +1)      # z high -> short the spread (short X, long Y)
    return None
 
 
def SizeLegs(gross: float, beta: float, px: float, py: float, side: int) -> tuple[float, float]:
    nx = gross / (1.0 + abs(beta))
    ny = abs(beta) * nx
    return side * nx / px, -side * ny / py