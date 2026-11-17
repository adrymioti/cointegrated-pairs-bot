from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, fields, asdict
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def LoadConfig(path: str | Path = "config/baseline.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(p)
    return cfg

def ConfigHash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys= True, default=str).encode()).hexdigest()[:12]

def Resolve(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p

@dataclass
class Params:
    lookback: int = 168
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    max_hold_bars: int = 336
    delay_bars: int = 0
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    borrow_rate_annual: float = 0.10
    leg_fail_prob: float = 0.0
    seed: int = 7
    pair_gross_frac: float = 0.20
    max_gross: float = 1.0
    max_pair: float = 0.20
    max_asset: float = 0.30
    collateral_ratio: float = 1.0
    dd_halt: float = 0.20
    halt_cooldown_bars: int = 720
    hours_per_year: int = 8760
    model_version: str = "baseline-v1"
 
    def Validate(self):
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be < entry_z")
        if self.entry_z >= self.stop_z:
            raise ValueError("entry_z must be < stop_z")
        if self.lookback < 10:
            raise ValueError("lookback too short")
 
 
def ParamsFromCFG(cfg: dict, **overrides) -> Params:
    names = {f.name for f in fields(Params)}
    vals = {}
    for sec in ("signals", "execution", "portfolio"):
        for k, v in cfg.get(sec, {}).items():
            if k in names:
                vals[k] = v
    vals["model_version"] = cfg.get("project", {}).get("model_version", "baseline-v1")
    vals.update({k: v for k, v in overrides.items() if k in names})
    p = Params(**vals)
    p.Validate()
    return p
 
 
def ParamsDict(p: Params) -> dict:
    return asdict(p)