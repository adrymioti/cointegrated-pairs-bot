from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
 
import numpy as np
import pandas as pd
 
from src.Config import Params
from src.Cointegration import PairSpec
from src.Signals import Decide, Spread, zScore
from . import Risk
 
BAR = pd.Timedelta(hours=1)
 
 
def _fin(x) -> bool:
    return x is not None and np.isfinite(x)
 
 
def _json_default(o):
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))
 
 
class Engine:
    def __init__(self, p: Params, initial_equity: float):
        self.p = p
        self.initial_equity = float(initial_equity)
        self.cash = float(initial_equity)
        self.pos: dict = {}
        self.pending: dict = {}
        self.marks: dict = {}
        self.bar = -1
        self.last_ts = None
        self.equity_last = float(initial_equity)
        self.peak = float(initial_equity)
        self.halted = False
        self.halt_until = -1
        self.cum_fees = self.cum_slip = self.cum_borrow = 0.0
        self.ledger: list = []
        self.blocked: list = []
        self.equity_rows: list = []
        self.rng = np.random.default_rng(p.seed)
        self.closed_now: set = set()

    def _block(self, ts, key, action, reason, detail=""):
            self.blocked.append(dict(time=ts, pair=key, action=action, reason=reason, detail=detail))
    
    def _equity(self) -> float:
            v = self.cash
            for pos in self.pos.values():
                v += pos["qx"] * self.marks.get(pos["x"], 0.0) + pos["qy"] * self.marks.get(pos["y"], 0.0)
            return v
    
    def _exposure(self):
            g = n = 0.0
            for pos in self.pos.values():
                vx = pos["qx"] * self.marks.get(pos["x"], 0.0)
                vy = pos["qy"] * self.marks.get(pos["y"], 0.0)
                g += abs(vx) + abs(vy)
                n += vx + vy
            return g, n
    
    def _record_equity(self, ts, overwrite=False):
            eq = self._equity()
            g, n = self._exposure()
            row = dict(time=ts, equity=eq, cash=self.cash, gross=g, net=n, n_pos=len(self.pos),
                    cum_costs=self.cum_fees + self.cum_slip + self.cum_borrow,
                    cum_fees=self.cum_fees, cum_slippage=self.cum_slip, cum_borrow=self.cum_borrow)
            if overwrite and self.equity_rows:
                self.equity_rows[-1] = row
            else:
                self.equity_rows.append(row)
            self.equity_last = eq
            return eq
    
    def _fill(self, asset, qty, ref):
            s = self.p.slippage_bps / 1e4
            fill = ref * (1 + s) if qty > 0 else ref * (1 - s)
            fee = abs(qty) * fill * self.p.fee_bps / 1e4
            slip_cost = abs(qty) * abs(fill - ref)
            self.cash -= qty * fill + fee
            self.cum_fees += fee
            self.cum_slip += slip_cost
            self.marks[asset] = ref
            return fill, fee, slip_cost
    
        
    def _exec_open(self, key, o, ts, opx):
            spec: PairSpec = o["spec"]
            if self.halted:
                return self._block(ts, key, "open", "drawdown_halt")
            if key in self.pos:
                return self._block(ts, key, "open", "position_exists")
            px, py = opx.get(spec.x), opx.get(spec.y)
            if not (_fin(px) and _fin(py)):
                return self._block(ts, key, "open", "missing_price", "stale/missing open price; order cancelled")
            G = self.p.pair_gross_frac * self.equity_last
            nx = G / (1 + abs(spec.beta))
            ny = abs(spec.beta) * nx
            side = o["side"]
            qx, qy = side * nx / px, -side * ny / py
            ok, reason, detail = Risk.CheckEntry(self.p, self.equity_last, self.cash, self.pos, self.marks,
                                                [(spec.x, qx, px), (spec.y, qy, py)])
            if not ok:
                return self._block(ts, key, "open", reason, detail)
            leg_fail = self.p.leg_fail_prob > 0 and self.rng.random() < self.p.leg_fail_prob
            pos = dict(key=key, x=spec.x, y=spec.y, alpha=spec.alpha, beta=spec.beta, side=side, qx=qx,
                    qy=0.0 if leg_fail else qy, signal_ts=o["signal_ts"], fill_ts=ts, z_entry=o["z"],
                    ref_x=px, ref_y=py, fees=0.0, slip=0.0, borrow=0.0, entry_bar=self.bar,
                    traded_notional=0.0, entry_reason="leg_failure" if leg_fail else o["reason"])
            for a, q, ref in ((spec.x, qx, px), (spec.y, 0.0 if leg_fail else qy, py)):
                if q == 0:
                    continue
                _, fee, sc = self._fill(a, q, ref)
                pos["fees"] += fee
                pos["slip"] += sc
                pos["traded_notional"] += abs(q) * ref
            self.pos[key] = pos
            if leg_fail:   # second leg failed: unwind the orphan leg at the next open
                self.pending[key] = dict(action="close", reason="leg_failure", signal_ts=ts, z=np.nan, due=self.bar + 1)
    
    def _exec_close(self, key, o, ts, refs, fill_ts=None):
            pos = self.pos.get(key)
            if pos is None:
                return
            legs = [(pos["x"], pos["qx"]), (pos["y"], pos["qy"])]
            if any(q != 0 and not _fin(refs.get(a)) for a, q in legs):
                self._block(ts, key, "close", "missing_price", "close retried next bar")
                self.pending[key] = dict(o, due=self.bar + 1)
                return
            gross = 0.0
            for a, q in legs:
                if q == 0:
                    continue
                ref = refs[a]
                _, fee, sc = self._fill(a, -q, ref)
                pos["fees"] += fee
                pos["slip"] += sc
                pos["traded_notional"] += abs(q) * ref
                gross += q * (ref - (pos["ref_x"] if a == pos["x"] else pos["ref_y"]))
            net = gross - pos["fees"] - pos["slip"] - pos["borrow"]
            rx, ry = refs.get(pos["x"]), refs.get(pos["y"])
            self.ledger.append(dict(
                pair=key, model_version=self.p.model_version,
                signal_time_entry=pos["signal_ts"] + BAR, fill_time_entry=pos["fill_ts"],
                signal_time_exit=o["signal_ts"] + BAR, fill_time_exit=fill_ts if fill_ts is not None else ts,
                side=pos["side"], alpha=pos["alpha"], beta=pos["beta"], qx=pos["qx"], qy=pos["qy"],
                entry_px_x=pos["ref_x"], entry_px_y=pos["ref_y"], exit_px_x=rx, exit_px_y=ry,
                z_entry=pos["z_entry"], z_exit=o["z"], entry_reason=pos["entry_reason"], exit_reason=o["reason"],
                gross_pnl=gross, fees=pos["fees"], slippage=pos["slip"], borrow=pos["borrow"], net_pnl=net,
                hold_bars=self.bar - pos["entry_bar"], traded_notional=pos["traded_notional"]))
            del self.pos[key]
            self.pending.pop(key, None)
            self.closed_now.add(key)
    
        # ------------------------------------------------------------------ public controls
    def queue_close_all(self, reason: str, ts):
            for k in self.pos:
                if k not in self.pending:
                    self.pending[k] = dict(action="close", reason=reason, signal_ts=ts, z=np.nan, due=self.bar + 1)
    
    def process_bar(self, ts, opx: dict, cpx: dict, pairs: list, zs: dict) -> bool:
            """Process one COMPLETED candle. Returns False (and does nothing) for duplicate/out-of-order bars."""
            if self.last_ts is not None and ts <= self.last_ts:
                return False
            self.bar += 1
            self.last_ts = ts
            self.closed_now = set()
            # 1. due orders (closes first so freed exposure is available)
            due = sorted([(k, o) for k, o in self.pending.items() if o["due"] <= self.bar],
                        key=lambda ko: ko[1]["action"] != "close")
            for k, o in due:
                self.pending.pop(k, None)
                if o["action"] == "close":
                    self._exec_close(k, o, ts, opx)
                else:
                    self._exec_open(k, o, ts, opx)
            # 2. borrow on short notional held during this bar
            rate = self.p.borrow_rate_annual / self.p.hours_per_year
            for pos in self.pos.values():
                for a, q in ((pos["x"], pos["qx"]), (pos["y"], pos["qy"])):
                    if q < 0:
                        px = opx.get(a)
                        px = px if _fin(px) else self.marks.get(a, 0.0)
                        c = abs(q) * px * rate
                        self.cash -= c
                        pos["borrow"] += c
                        self.cum_borrow += c
            # 3. mark at close, equity, drawdown halt
            for a, v in cpx.items():
                if _fin(v):
                    self.marks[a] = v
            eq = self._record_equity(ts)
            if self.halted and self.bar >= self.halt_until:
                self.halted = False
                self.peak = eq
            self.peak = max(self.peak, eq)
            if not self.halted and self.peak > 0 and 1 - eq / self.peak >= self.p.dd_halt:
                self.halted = True
                self.halt_until = self.bar + self.p.halt_cooldown_bars
                self.pending = {k: o for k, o in self.pending.items() if o["action"] == "close"}
                self.queue_close_all("drawdown_halt", ts)
            # 4. signals on the completed candle
            for spec in pairs:
                k = spec.key
                if k in self.pending:
                    continue
                cx, cy = cpx.get(spec.x), cpx.get(spec.y)
                z = zs.get(k, np.nan)
                if not (_fin(cx) and _fin(cy)):
                    self._block(ts, k, "signal", "signal_suspended_missing_data")
                    z = np.nan
                pos = self.pos.get(k)
                side = pos["side"] if pos else 0
                held = self.bar - pos["entry_bar"] if pos else 0
                res = Decide(z, side, held, self.p, allow_entry=(not self.halted and k not in self.closed_now))
                if not res:
                    continue
                due_bar = self.bar + 1 + self.p.delay_bars
                if res[0] == "open":
                    self.pending[k] = dict(action="open", side=res[1], spec=spec, reason="z_entry",
                                        signal_ts=ts, z=float(z), due=due_bar)
                else:
                    self.pending[k] = dict(action="close", reason=res[1], signal_ts=ts, z=float(z), due=due_bar)
            return True
    
    def final_liquidate(self, ts, cpx: dict):
            """End of sample: close everything at the last completed close (costs charged)."""
            self.pending.clear()
            for k in list(self.pos):
                self._exec_close(k, dict(action="close", reason="end_of_sample", signal_ts=ts - BAR, z=np.nan, due=0),
                                ts, cpx, fill_ts=ts + BAR)
            self._record_equity(ts, overwrite=True)
    
        # ------------------------------------------------------------------ persistence (live restarts)
    def to_state(self) -> str:
            pend = {}
            for k, o in self.pending.items():
                o2 = dict(o)
                if "spec" in o2:
                    o2["spec"] = asdict(o2["spec"])
                pend[k] = o2
            state = dict(cash=self.cash, pos=self.pos, pending=pend, marks=self.marks, bar=self.bar,
                        last_ts=self.last_ts, equity_last=self.equity_last, peak=self.peak, halted=self.halted,
                        halt_until=self.halt_until, cum_fees=self.cum_fees, cum_slip=self.cum_slip,
                        cum_borrow=self.cum_borrow, ledger=self.ledger, blocked=self.blocked,
                        initial_equity=self.initial_equity, rng=self.rng.bit_generator.state)
            return json.dumps(state, default=_json_default)
    
    @classmethod
    def from_state(cls, p: Params, text: str) -> "Engine":
            s = json.loads(text)
            e = cls(p, s["initial_equity"])
            for k in ("cash", "marks", "bar", "equity_last", "peak", "halted", "halt_until", "cum_fees", "cum_slip",
                    "cum_borrow", "ledger", "blocked"):
                setattr(e, k, s[k])
            e.last_ts = pd.Timestamp(s["last_ts"]) if s["last_ts"] else None
            e.pos = s["pos"]
            for pos in e.pos.values():
                pos["signal_ts"], pos["fill_ts"] = pd.Timestamp(pos["signal_ts"]), pd.Timestamp(pos["fill_ts"])
            e.pending = {}
            for k, o in s["pending"].items():
                o["signal_ts"] = pd.Timestamp(o["signal_ts"])
                if "spec" in o:
                    o["spec"] = PairSpec(**o["spec"])
                e.pending[k] = o
            e.rng.bit_generator.state = s["rng"]
            return e
 
 
@dataclass
class Window:
    name: str
    start: pd.Timestamp          # first bar that may trade (inclusive)
    end: pd.Timestamp            # exclusive
    pairs: list = field(default_factory=list)
 
 
@dataclass
class BacktestResult:
    equity: pd.DataFrame
    ledger: pd.DataFrame
    blocked: pd.DataFrame
    params: Params
    windows: list
 
 
def run_backtest(open_df: pd.DataFrame, close_df: pd.DataFrame, windows: list, p: Params,
                 initial_equity: float = 100_000.0) -> BacktestResult:
    """Run one portfolio path through consecutive windows. Warm-up for z-scores uses ALL prior data,
    but nothing before the first window's start is traded or counted."""
    eng = Engine(p, initial_equity)
    idx = close_df.index
    cols = list(close_df.columns)
    O, C = open_df[cols].values, close_df.values
    last_i = None
    for w in windows:
        pos_idx = np.where((idx >= w.start) & (idx < w.end))[0]
        if len(pos_idx) == 0:
            continue
        zcols = {sp.key: zScore(Spread(close_df[sp.x], close_df[sp.y], sp.alpha, sp.beta), p.lookback).values
                 for sp in w.pairs}
        for i in pos_idx:
            opx = dict(zip(cols, O[i]))
            cpx = dict(zip(cols, C[i]))
            zs = {k: v[i] for k, v in zcols.items()}
            eng.process_bar(idx[i], opx, cpx, w.pairs, zs)
            last_i = i
        eng.queue_close_all("window_boundary", idx[last_i])   # executes at next processed bar's open
    if last_i is not None:
        eng.final_liquidate(idx[last_i], dict(zip(cols, C[last_i])))
    eq = pd.DataFrame(eng.equity_rows).set_index("time") if eng.equity_rows else pd.DataFrame()
    led = pd.DataFrame(eng.ledger)
    blk = pd.DataFrame(eng.blocked, columns=["time", "pair", "action", "reason", "detail"])
    return BacktestResult(eq, led, blk, p, windows)
 