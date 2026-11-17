from __future__ import annotations
import argparse
from pathlib import Path

from .Config import LoadConfig, Resolve, ROOT


def Main(argv=None):
    ap = argparse.ArgumentParser(prog="src.cli")
    ap.add_argument("command", choices=["download", "process", "synthetic", "run", "paper-trade", "replay"])
    ap.add_argument("--config", default="config/baseline.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--pairs-csv", help="frozen pair table (default: results/tables/pair_selection_development.csv)")
    a = ap.parse_args(argv)
    cfg = LoadConfig(a.config)

    if a.command == "download":
        from .DataLoader import DownloadKlines
        m = DownloadKlines(cfg)
        print(m["status"].value_counts())
    elif a.command == "process":
        from .DataLoader import BuildProcessed
        out = BuildProcessed(cfg)
        print(out["quality"].to_string())
    elif a.command == "synthetic":
        from .Synthetic import MakeSynthetic
        MakeSynthetic(cfg, ROOT / "data_synthetic" / "processed")
        print("wrote data_synthetic/processed (SYNTHETIC)")
    elif a.command == "run":
        from .Pipeline import RunAll
        if a.synthetic:
            data_dir, out_dir, label = ROOT / "data_synthetic" / "processed", ROOT / "results_synthetic", "SYNTHETIC"
        else:
            data_dir, out_dir, label = Resolve(cfg, "processed"), Resolve(cfg, "results"), "REAL DATA"
        s = RunAll(cfg, data_dir, out_dir, label)
        print(f"done -> {out_dir}/results_summary.md  (pairs selected: {s.get('n_pairs_selected')})")
    elif a.command == "paper-trade":
        from .LiveFeed import RunPaperTrading
        RunPaperTrading(cfg, a.pairs_csv)
    elif a.command == "replay":
        from .LiveFeed import RelayAndCompare
        print(RelayAndCompare(cfg, a.pairs_csv).to_string())


if __name__ == "__main__":
    Main()