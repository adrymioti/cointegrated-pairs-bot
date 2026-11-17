from __future__ import annotations
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .Backtest import Engine, run_backtest, Window, BAR
from .Cointegration import PairSpec
from .Config import ParamsFromCFG, Resolve, ROOT
from .Signals import Spread, zScore


def ParseKlineMessage(msg: dict):
    """Binance combined-stream payload -> (symbol, open_ts, o, c, quote_vol) for CLOSED candles only."""
    k = msg.get("data", msg).get("k")
    if not k or not k.get("x"):
        return None
    return (k["s"], pd.Timestamp(k["t"], unit="ms", tz="UTC").as_unit("ns"), float(k["o"]), float(k["c"]), float(k["q"]))


class BarAggregator:
    """Collect closed candles per hour; emit a complete row when every symbol arrived, or when the
    grace period expires (missing symbols become NaN -> blocked, never forward-filled)."""

    def __init__(self, symbols, grace_seconds=600):
        self.symbols, self.grace = list(symbols), grace_seconds
        self.buf: dict = {}
        self.emitted: set = set()

    def add(self, sym, ts, o, c):
        if ts in self.emitted:
            return                                   # duplicate / late event for an emitted hour
        self.buf.setdefault(ts, {})[sym] = (o, c)    # duplicate within an hour just overwrites identically

    def ready(self, now: pd.Timestamp):
        out = []
        for ts in sorted(self.buf):
            row = self.buf[ts]
            complete = all(s in row for s in self.symbols)
            expired = now >= ts + BAR + pd.Timedelta(seconds=self.grace)
            if complete or expired:
                opx = {s: row.get(s, (np.nan, np.nan))[0] for s in self.symbols}
                cpx = {s: row.get(s, (np.nan, np.nan))[1] for s in self.symbols}
                out.append((ts, opx, cpx))
                self.emitted.add(ts)
        for ts, _, _ in out:
            del self.buf[ts]
        return out


def ReadBarsLog(path):
    """Flat CSV (time, open_<SYM>..., close_<SYM>...) -> (open_df, close_df) with UTC ns index."""
    b = pd.read_csv(path, index_col=0, parse_dates=True)
    b.index = pd.DatetimeIndex(b.index).tz_convert("UTC").as_unit("ns")
    b = b[~b.index.duplicated()]
    bo = b[[c for c in b.columns if c.startswith("open_")]].rename(columns=lambda c: c[5:])
    bc = b[[c for c in b.columns if c.startswith("close_")]].rename(columns=lambda c: c[6:])
    return bo, bc


class LiveRunner:
    """Feeds completed bars to the shared Engine and persists everything."""

    def __init__(self, cfg: dict, pairs: list, history_close: pd.DataFrame, state_dir: Path):
        self.cfg, self.pairs = cfg, pairs
        self.p = ParamsFromCFG(cfg)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "engine_state.json"
        self.bars_path = self.state_dir / "bars_log.csv"
        self.close_hist = history_close.copy()          # warm-up closes (>= lookback + 1 bars)
        self.open_hist = history_close.copy()
        if self.state_path.exists():
            self.engine = Engine.from_state(self.p, self.state_path.read_text())
        else:
            self.engine = Engine(self.p, cfg["portfolio"]["initial_equity"])
        if self.bars_path.exists():                      # rebuild in-memory history after a restart
            bo, bc = ReadBarsLog(self.bars_path)
            for ts in bc.index:
                if ts > self.close_hist.index[-1]:
                    self.close_hist.loc[ts] = bc.loc[ts, self.close_hist.columns]
                    self.open_hist.loc[ts] = bo.loc[ts, self.open_hist.columns]

    def OnBar(self, ts, opx: dict, cpx: dict) -> bool:
        if self.engine.last_ts is not None and ts <= self.engine.last_ts:
            return False                                 # duplicate or out of order
        cols = self.close_hist.columns
        self.close_hist.loc[ts] = [cpx.get(c, np.nan) for c in cols]
        self.open_hist.loc[ts] = [opx.get(c, np.nan) for c in cols]
        zs = {}
        for sp in self.pairs:
            tail = self.close_hist.iloc[-(self.p.lookback + 2):]
            z = zScore(Spread(tail[sp.x], tail[sp.y], sp.alpha, sp.beta), self.p.lookback)
            zs[sp.key] = z.iloc[-1]
        n_led, n_blk = len(self.engine.ledger), len(self.engine.blocked)
        ok = self.engine.process_bar(ts, opx, cpx, self.pairs, zs)
        self._persist(ts, opx, cpx, zs, n_led, n_blk)
        return ok

    def Persist(self, ts, opx, cpx, zs, n_led, n_blk):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(self.engine.to_state())
        os.replace(tmp, self.state_path)                 # atomic: restart sees old or new state, never half
        row = pd.DataFrame([{**{f"open_{k}": v for k, v in opx.items()}, **{f"close_{k}": v for k, v in cpx.items()}}],
                           index=[ts])
        row.index.name = "time"
        row.to_csv(self.bars_path, mode="a", header=not self.bars_path.exists())
        eq = pd.DataFrame(self.engine.equity_rows[-1:]) if self.engine.equity_rows else pd.DataFrame()
        if len(eq):
            eq.to_csv(self.state_dir / "equity_log.csv", mode="a", header=not (self.state_dir / "equity_log.csv").exists(), index=False)
        pd.DataFrame(self.engine.ledger).to_csv(self.state_dir / "trade_ledger_live.csv", index=False)
        pd.DataFrame(self.engine.blocked).to_csv(self.state_dir / "blocked_orders_live.csv", index=False)
        pd.DataFrame([dict(time=ts, pair=k, z=v) for k, v in zs.items()]).to_csv(
            self.state_dir / "signals_log.csv", mode="a", header=not (self.state_dir / "signals_log.csv").exists(), index=False)


# ------------------------------------------------------------------ network I/O (not run in the sandbox)
def BackFillRest(cfg, symbol, start_ts: pd.Timestamp, end_ts: pd.Timestamp):
    """Fetch missing CLOSED hourly bars via REST after an outage/reconnect."""
    import requests
    out, t = [], start_ts
    while t < end_ts:
        r = requests.get(cfg["live"]["rest_url"], params=dict(symbol=symbol, interval="1h",
                         startTime=int(t.timestamp() * 1000), limit=1000), timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        for k in rows:
            close_time = pd.Timestamp(k[6], unit="ms", tz="UTC")
            if close_time < pd.Timestamp.now(tz="UTC"):
                out.append((symbol, pd.Timestamp(k[0], unit="ms", tz="UTC").as_unit("ns"), float(k[1]), float(k[4]), float(k[7])))
        t = pd.Timestamp(rows[-1][0], unit="ms", tz="UTC") + BAR
    return out


async def StreamForever(cfg, symbols, on_closed):
    """Reconnecting WebSocket reader with a stale-data watchdog. `on_closed(sym, ts, o, c, q)` per closed candle."""
    import websockets
    streams = "/".join(f"{s.lower()}@kline_1h" for s in symbols)
    url = f"{cfg['live']['ws_url']}?streams={streams}"
    backoff = 1
    while True:
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                backoff = 1
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=cfg["live"]["stale_seconds"])
                    parsed = ParseKlineMessage(json.loads(raw))
                    if parsed:
                        on_closed(*parsed)
        except Exception as e:                           # reconnect on timeout/stale/any error
            print(f"[{pd.Timestamp.now(tz='UTC')}] stream error {type(e).__name__}: {e}; reconnect in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


def LoadFrozenPairs(cfg, csv_path=None) -> list:
    path = Path(csv_path) if csv_path else Resolve(cfg, "results") / "tables" / "pair_selection_development.csv"
    t = pd.read_csv(path)
    return [PairSpec(r["pair"], r["x"], r["y"], float(r["alpha"]), float(r["beta"])) for _, r in t[t["selected"]].iterrows()]


def RunPaperTrading(cfg, pairs_csv=None):
    from .DataLoader import LoadProcessed
    pairs = LoadFrozenPairs(cfg, pairs_csv)
    symbols = sorted({s for p in pairs for s in (p.x, p.y)})
    hist = LoadProcessed(cfg)["close"][symbols].dropna().iloc[-(cfg["signals"]["lookback"] + 50):]
    runner = LiveRunner(cfg, pairs, hist, Resolve(cfg, "results") / "live")
    agg = BarAggregator(symbols, cfg["live"]["grace_seconds"])
    agg_lock = asyncio.Lock()

    def OnClosed(sym, ts, o, c, q):
        agg.add(sym, ts, o, c)
        for t, opx, cpx in agg.ready(pd.Timestamp.now(tz="UTC")):
            runner.OnBar(t, opx, cpx)
            print(f"bar {t} processed; equity={runner.engine.equity_last:,.2f}; open={list(runner.engine.pos)}")

    asyncio.run(StreamForever(cfg, symbols, OnClosed))


def RelayAndCompare(cfg, pairs_csv=None) -> pd.DataFrame:
    """Replay logged live bars through the backtester; list differences vs. the live ledger."""
    state = Resolve(cfg, "results") / "live"
    pairs = LoadFrozenPairs(cfg, pairs_csv)
    bo, bc = ReadBarsLog(state / "bars_log.csv")
    from .DataLoader import LoadProcessed
    proc = LoadProcessed(cfg)
    close = pd.concat([proc["close"][bc.columns], bc]).loc[lambda d: ~d.index.duplicated()]
    open_ = pd.concat([proc["open"][bo.columns], bo]).loc[lambda d: ~d.index.duplicated()]
    res = run_backtest(open_, close, [Window("live_replay", bc.index[0], bc.index[-1] + BAR, pairs)],
                       ParamsFromCFG(cfg), cfg["portfolio"]["initial_equity"])
    live = pd.read_csv(state / "trade_ledger_live.csv") if (state / "trade_ledger_live.csv").stat().st_size > 5 else pd.DataFrame()
    cols = ["pair", "fill_time_entry", "fill_time_exit", "net_pnl"]
    return pd.concat({"backtest_replay": res.ledger[cols] if len(res.ledger) else pd.DataFrame(columns=cols),
                      "live": live[cols] if len(live) else pd.DataFrame(columns=cols)}, axis=0)
