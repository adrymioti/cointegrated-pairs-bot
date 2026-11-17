from __future__ import annotations
import pandas as pd

from .Cointegration import SelectPairs
from .Backtest import Window, BAR


def split_points(index: pd.DatetimeIndex, cfg: dict) -> dict:
    """Chronological 60/40 development/final-test split; last 40% of development = validation."""
    n = len(index)
    s = cfg["split"]
    i_test = int(n * s["dev_fraction"])
    i_val = int(i_test * (1 - s["validation_fraction_of_dev"]))
    return dict(dev_start=index[0], val_start=index[i_val], test_start=index[i_test], test_end=index[-1] + BAR)


def frozen_window(close, qvol, cfg, split):
    """Select pairs and estimate alpha/beta on DEVELOPMENT only, then freeze them for the final test."""
    table, pairs, excl = SelectPairs(close, qvol, split["dev_start"], split["test_start"], cfg)
    return Window("frozen_test", split["test_start"], split["test_end"], pairs), table, excl


def validation_window(close, qvol, cfg, split):
    """Selection on the first part of development; evaluation on the validation segment (sensitivity grid)."""
    table, pairs, excl = SelectPairs(close, qvol, split["dev_start"], split["val_start"], cfg)
    return Window("validation", split["val_start"], split["test_start"], pairs), table, excl


def walk_forward_windows(close, qvol, cfg, start, end):
    """Estimate on the previous `train_hours`, trade the next `trade_hours`; re-select pairs each time
    using only past data. Non-overlapping trading windows; positions are closed at each boundary."""
    wf = cfg["walk_forward"]
    idx = close.index
    tables, windows, excls = [], [], []
    t = pd.Timestamp(start)
    k = 0
    while t < pd.Timestamp(end):
        t_end = min(t + pd.Timedelta(hours=wf["trade_hours"]), pd.Timestamp(end))
        tr_start = t - pd.Timedelta(hours=wf["train_hours"])
        if tr_start < idx[0]:
            t = t_end
            continue
        table, pairs, excl = SelectPairs(close, qvol, tr_start, t, cfg)
        name = f"wf_{k:02d}"
        for df in (table, excl):
            if len(df):
                df.insert(0, "window", name)
        tables.append(table)
        excls.append(excl)
        windows.append(Window(name, t, t_end, pairs))
        t = t + pd.Timedelta(hours=wf["step_hours"])
        k += 1
    table_all = pd.concat([x for x in tables if len(x)], ignore_index=True) if any(len(x) for x in tables) else pd.DataFrame()
    excl_all = pd.concat([x for x in excls if len(x)], ignore_index=True) if any(len(x) for x in excls) else pd.DataFrame()
    return windows, table_all, excl_all

