from __future__ import annotations
import datetime as dt
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import Plots
from .Backtest import run_backtest, Window
from .Cointegration import SelectPairs
from .Config import ConfigHash, ParamsFromCFG, ParamsDict, ROOT
from .DataLoader import LoadProcessed, ProcessedHashes
from .Metrics import (Performance, GrossNetTable, BuyAndHold, BlockBoostrap, SimpleReturns)
from .Regimes import RegimeLabels, RegimeTable, RollingDiagnostics
from .Robustness import (SenstivityGrid, CostStress, BreakEven, DelaySensitivity, LegFailureStress,
                         Subperiods)
from .Signals import Spread, zScore
from .Validation import split_points, frozen_window, validation_window, walk_forward_windows


def MDTable(df: pd.DataFrame, floatfmt="{:.4g}") -> str:
    if df is None or len(df) == 0:
        return "_(empty)_"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def Save(df, path, index=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def WriteManifest(cfg, out_dir, data_dir, label, extra=None):
    def ver(pkg):
        try:
            return metadata.version(pkg)
        except Exception:
            return "not installed"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    m = dict(label=label, created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), config_hash=ConfigHash(cfg),
             git_commit=commit or "unavailable", python=platform.python_version(),
             packages={p: ver(p) for p in ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "PyYAML"]},
             input_hashes=ProcessedHashes(data_dir),
             config={k: v for k, v in cfg.items() if not k.startswith("_")}, **(extra or {}))
    (Path(out_dir) / "run_manifest.json").write_text(json.dumps(m, indent=2, default=str))


def RunAll(cfg: dict, data_dir: Path, out_dir: Path, label: str = "REAL DATA") -> dict:
    out_dir = Path(out_dir)
    T, F = out_dir / "tables", out_dir / "figures"
    T.mkdir(parents=True, exist_ok=True)
    F.mkdir(parents=True, exist_ok=True)
    data = LoadProcessed(processed_dir=data_dir)
    open_df, close, qvol = data["open"], data["close"], data["quote_volume"]
    init = cfg["portfolio"]["initial_equity"]
    K = cfg["portfolio"]["hours_per_year"]
    P = ParamsFromCFG(cfg)
    summary = dict(label=label)

    # ---------------- Stage 1: exploration
    q = data_dir / "data_quality.csv"
    if q.exists():
        Save(pd.read_csv(q), T / "data_quality.csv")
    Plots.PriceHistory(close, F / "01_prices.png", label)
    Plots.RollingVol(close, F / "02_rolling_vol.png", cfg["regimes"]["window_hours"])
    Save(Plots.CorrMatrix(close, F / "03_return_correlation.png"), T / "return_correlation.csv", index=True)

    # ---------------- Stage 6 split + Stage 2 selection (development only)
    split = split_points(close.index, cfg)
    summary["split"] = {k: str(v) for k, v in split.items()}
    fw, sel_table, sel_excl = frozen_window(close, qvol, cfg, split)
    Save(sel_table, T / "pair_selection_development.csv")
    Save(sel_excl, T / "asset_exclusions_development.csv")
    summary["n_pairs_tested"] = int(sel_table["p_raw"].notna().sum()) if len(sel_table) else 0
    summary["n_pairs_selected"] = len(fw.pairs)
    summary["selected_pairs"] = [dict(pair=p.key, alpha=p.alpha, beta=p.beta) for p in fw.pairs]

    # ---------------- Stage 7: sensitivity on validation window (never the final test)
    vw, vtable, _ = validation_window(close, qvol, cfg, split)
    Save(vtable, T / "pair_selection_validation_stage.csv")
    if vw.pairs:
        grid = SenstivityGrid(open_df, close, vw, cfg)
        Save(grid, T / "sensitivity_grid_validation.csv")
        Plots.Heatmaps(grid, F / "07_sensitivity_heatmap.png")
        summary["grid_cells"] = len(grid)
    else:
        summary["grid_cells"] = 0
    Plots.SplitDiagram(split, None, F / "06_split_diagram.png")

    if not fw.pairs:
        summary["note"] = ("No pair passed Holm + filters on the development period. "
                           "This is reported as the result; the standard was not lowered.")
        WriteManifest(cfg, out_dir, data_dir, label, dict(summary=summary))
        (out_dir / "results_summary.md").write_text(SummaryMd(summary, {}, label))
        return summary

    # ---------------- Stage 3-5: frozen-model final test, net of costs
    res = run_backtest(open_df, close, [fw], P, init)
    Save(res.ledger, T / "trade_ledger_frozen_test.csv")
    Save(res.blocked, T / "blocked_orders_frozen_test.csv")
    Save(res.equity.reset_index(), T / "equity_frozen_test.csv")
    gn = GrossNetTable(res, init, K)
    Save(gn.reset_index().rename(columns={"index": "series"}), T / "gross_vs_net_frozen_test.csv")
    net_eq = res.equity["equity"]
    bh = BuyAndHold(close, split["test_start"], split["test_end"], init)
    Plots.EquityDrawdown({"Strategy (net)": net_eq, "Strategy (gross)": net_eq + res.equity["cum_costs"],
                           "Equal-weight buy-and-hold (reference)": bh.reindex(net_eq.index)},
                          F / "05_equity_drawdown_frozen_test.png",
                          f"Final test, frozen model, net of {P.fee_bps + P.slippage_bps:g} bps | {label}")
    r = SimpleReturns(net_eq, init)
    boot = BlockBoostrap(r, K, cfg["robustness"]["bootstrap_block"], cfg["robustness"]["bootstrap_reps"])
    (T / "bootstrap_frozen_test.json").write_text(json.dumps(boot, indent=2))
    ref = dict(cash_zero_return=0.0, buy_hold_total_return=float(bh.iloc[-1] / init - 1),
               buy_hold_max_dd=float((1 - bh / bh.cummax()).max()))
    (T / "references.json").write_text(json.dumps(ref, indent=2))
    for sp in fw.pairs:
        z = zScore(Spread(close[sp.x], close[sp.y], sp.alpha, sp.beta), P.lookback)
        Plots.SpreadWithTrades(close, sp, z, res.ledger, F / f"04_spread_{sp.key}.png", P.entry_z, P.exit_z, P.stop_z,
                                 start=split["test_start"])

    stress = CostStress(open_df, close, [fw], cfg)
    Save(stress, T / "cost_stress_frozen_test.csv")
    be = BreakEven(stress)
    summary["break_even_cost_bps"] = be
    Save(CostStress(open_df, close, [fw], cfg, cfg["robustness"]["stress_cost_bps"]), T / "cost_scenarios_5_10_20bps.csv")
    Save(DelaySensitivity(open_df, close, [fw], cfg), T / "delay_sensitivity_frozen_test.csv")
    Save(LegFailureStress(open_df, close, [fw], cfg), T / "leg_failure_stress_frozen_test.csv")
    Save(Subperiods(open_df, close, fw, cfg), T / "subperiods_frozen_test.csv")

    # ---------------- Stage 6: walk-forward on the same final-test period
    wf_windows, wf_table, wf_excl = walk_forward_windows(close, qvol, cfg, split["test_start"], split["test_end"])
    Save(wf_table, T / "pair_selection_walk_forward.csv")
    wres = run_backtest(open_df, close, wf_windows, P, init)
    Save(wres.ledger, T / "trade_ledger_walk_forward.csv")
    Save(wres.equity.reset_index(), T / "equity_walk_forward.csv")
    wgn = GrossNetTable(wres, init, K)
    Save(wgn.reset_index().rename(columns={"index": "series"}), T / "gross_vs_net_walk_forward.csv")
    Plots.EquityDrawdown({"Walk-forward (net)": wres.equity["equity"], "Frozen (net)": net_eq,
                           "Equal-weight buy-and-hold (reference)": bh.reindex(wres.equity.index)},
                          F / "05b_equity_walk_forward.png", f"Final test: walk-forward vs frozen | {label}")

    # ---------------- Stage 8: regimes and stability (descriptive; filters would need fresh data)
    labels, rinfo = RegimeLabels(close, cfg["regimes"]["benchmark"], cfg["regimes"]["window_hours"], split["test_start"])
    rt = RegimeTable(res, labels, init, K)
    Save(rt, T / "regime_table_frozen_test.csv")
    roll = RollingDiagnostics(close, fw.pairs)
    Save(roll, T / "rolling_diagnostics.csv")
    Plots.Stability(roll, F / "08_stability.png")

    summary.update(frozen_net=gn.loc["net"].to_dict(), frozen_gross=gn.loc["gross"].to_dict(),
                   walk_forward_net=wgn.loc["net"].to_dict(), references=ref, bootstrap=boot,
                   regime_info=rinfo)
    tables = dict(selection=sel_table[sel_table["selected"]] if "selected" in sel_table else sel_table,
                  gross_net=gn, wf=wgn, stress=stress, regimes=rt)
    WriteManifest(cfg, out_dir, data_dir, label, dict(summary=summary, params=ParamsDict(P)))
    (out_dir / "results_summary.md").write_text(SummaryMd(summary, tables, label))
    return summary


def SummaryMd(s: dict, t: dict, label: str) -> str:
    L = [f"# Results summary - {label}", ""]
    if label != "REAL DATA":
        L += ["> **SYNTHETIC DATA - pipeline validation only. These numbers say nothing about crypto markets.**", ""]
    L += ["All headline numbers below are from the **final test period** (out of sample), "
          "frozen model fitted on development data only, net of the stated costs.", "",
          f"- Split: {s.get('split')}",
          f"- Pairs tested: {s.get('n_pairs_tested')}, selected after Holm (5% FWER) + filters: {s.get('n_pairs_selected')}",
          f"- Selected pairs: {s.get('selected_pairs')}"]
    if "note" in s:
        L += ["", f"**{s['note']}**"]
    if t:
        L += ["", "## Gross vs net (frozen model, final test)", MDTable(t["gross_net"].reset_index().rename(columns={"index": "series"})),
              "", "## Walk-forward (same period)", MDTable(t["wf"].reset_index().rename(columns={"index": "series"})),
              "", "## Cost stress (total fee+slippage bps per executed notional)", MDTable(t["stress"]),
              f"", f"Break-even cost: {s.get('break_even_cost_bps')} bps (0 = unprofitable even at zero cost; inf = profitable at largest tested cost)",
              "", "## Regimes (descriptive)", MDTable(t["regimes"]),
              "", f"Block-bootstrap (5th-95th pct): {s.get('bootstrap')}",
              f"References: {s.get('references')}"]
    L += ["", "Prespecified: pair-selection method, Holm correction, 60/40 split, walk-forward schedule, grid, cost scenarios.",
          "Exploratory: regime analysis, rolling diagnostics, sub-period and leg-failure stress tests."]
    return "\n".join(L)
