from __future__ import annotations
import datetime as dt
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .Config import LoadConfig, Resolve, ROOT, ConfigHash

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
FIELDS = ["open", "high", "low", "close", "volume", "quote_volume"]
MANIFEST_COLS = ["symbol", "month", "url", "file", "bytes", "sha256", "checksum_verified",
                 "status", "download_date_utc"]


def MonthList(start: str, end: str) -> list[str]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return [p.strftime("%Y-%m") for p in pd.period_range(s, e, freq="M")]


def KlineURL(cfg: dict, symbol: str, month: str) -> str:
    d = cfg["data"]
    return (f"{d['base_url']}/data/{d['market']}/monthly/klines/{symbol}/{d['interval']}/"
            f"{symbol}-{d['interval']}-{month}.zip")


def sha256Bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256File(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def UTCIndex(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return idx.as_unit("ns")


def HourlyCalendar(cfg: dict) -> pd.DatetimeIndex:
    d = cfg["data"]
    return UTCIndex(pd.date_range(d["start"], d["end"], freq="h", tz="UTC"))


def DownloadKlines(cfg: dict, symbols: list[str] | None = None, session=None) -> pd.DataFrame:
    """Download monthly kline zips. Existing raw files are never overwritten."""
    import requests
    session = session or requests.Session()
    raw_dir = Resolve(cfg, "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir.parent / "manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame(columns=MANIFEST_COLS)
    have = set(zip(manifest.symbol, manifest.month, manifest.status))
    rows = []
    d = cfg["data"]
    for sym in symbols or d["assets"]:
        (raw_dir / sym).mkdir(exist_ok=True)
        for month in MonthList(d["start"], d["end"]):
            url = KlineURL(cfg, sym, month)
            fpath = raw_dir / sym / f"{sym}-{d['interval']}-{month}.zip"
            if fpath.exists() and (sym, month, "ok") in have:
                continue
            if fpath.exists():          
                rows.append(dict(symbol=sym, month=month, url=url, file=str(fpath.relative_to(ROOT)),
                                 bytes=fpath.stat().st_size, sha256=sha256File(fpath),
                                 checksum_verified="not_checked", status="ok",
                                 download_date_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
                continue
            r = session.get(url, timeout=60)
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            if r.status_code != 200:
                rows.append(dict(symbol=sym, month=month, url=url, file="", bytes=0, sha256="",
                                 checksum_verified="", status=f"http_{r.status_code}", download_date_utc=now))
                continue
            digest = sha256Bytes(r.content)
            verified = "not_available"
            try:
                rc = session.get(url + ".CHECKSUM", timeout=30)
                if rc.status_code == 200:
                    verified = "yes" if rc.text.split()[0].lower() == digest else "MISMATCH"
            except Exception:
                pass
            if verified == "MISMATCH":
                raise RuntimeError(f"Checksum mismatch for {url}")
            fpath.write_bytes(r.content)
            rows.append(dict(symbol=sym, month=month, url=url, file=str(fpath.relative_to(ROOT)),
                             bytes=len(r.content), sha256=digest, checksum_verified=verified,
                             status="ok", download_date_utc=now))
    if rows:
        manifest = pd.concat([manifest, pd.DataFrame(rows)], ignore_index=True)
        manifest.to_csv(manifest_path, index=False)
    return manifest


def DetectUnit(open_time: pd.Series) -> str:
    m = float(open_time.max())
    if m >= 1e15:
        return "us"   
    if m >= 1e12:
        return "ms"
    raise ValueError(f"Unrecognised timestamp magnitude: {m}")


def ReadKlineBytes(data: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open(z.namelist()[0]) as f:
            raw = pd.read_csv(f, header=None, dtype=str)
    if not raw.iloc[0, 0].strip().isdigit():       
        raw = raw.iloc[1:]
    raw = raw.iloc[:, :12]
    raw.columns = KLINE_COLS
    for c in KLINE_COLS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    unit = DetectUnit(raw["open_time"])
    div = 1_000 if unit == "us" else 1
    raw["open_dt"] = pd.to_datetime(raw["open_time"] // div, unit="ms", utc=True).astype("datetime64[ns, UTC]")
    raw["close_dt"] = pd.to_datetime(raw["close_time"] // div, unit="ms", utc=True).astype("datetime64[ns, UTC]")
    raw.attrs["timestamp_unit"] = unit
    return raw


def read_kline_zip(path: Path) -> pd.DataFrame:
    return ReadKlineBytes(Path(path).read_bytes())


def CleanSymbol(df: pd.DataFrame, symbol: str, log: list) -> pd.DataFrame:
    """Drop duplicates, null out invalid bars (never repair them), drop incomplete candles."""
    df = df.sort_values("open_dt")
    dup = df.duplicated("open_dt", keep="first")
    if dup.any():
        log.append(dict(symbol=symbol, issue="duplicate_timestamp", count=int(dup.sum()), action="kept first"))
        df = df[~dup]
    bad = (df[["open", "high", "low", "close"]].isna().any(axis=1)
           | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
           | (df["high"] < df["low"]) | (df["close"] > df["high"]) | (df["close"] < df["low"])
           | (df["open"] > df["high"]) | (df["open"] < df["low"]) | (df["volume"] < 0))
    if bad.any():
        log.append(dict(symbol=symbol, issue="invalid_ohlcv", count=int(bad.sum()), action="set to NaN (not repaired)"))
        df.loc[bad, FIELDS] = np.nan
    now = pd.Timestamp.now(tz="UTC")
    incomplete = df["close_dt"] > now
    if incomplete.any():
        log.append(dict(symbol=symbol, issue="incomplete_candle", count=int(incomplete.sum()), action="dropped"))
        df = df[~incomplete]
    return df.set_index("open_dt")[FIELDS]


def BuildProcessed(cfg: dict, raw_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    """Raw zips -> aligned wide hourly frames (one per field) + quality table + exclusion log."""
    raw_dir = Path(raw_dir or Resolve(cfg, "raw"))
    out_dir = Path(out_dir or Resolve(cfg, "processed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cal = HourlyCalendar(cfg)
    log: list = []
    wide = {f: {} for f in FIELDS}
    quality = []
    for sym in cfg["data"]["assets"]:
        files = sorted((raw_dir / sym).glob(f"{sym}-*.zip"))
        if not files:
            log.append(dict(symbol=sym, issue="no_raw_files", count=0, action="symbol excluded"))
            continue
        frames = [read_kline_zip(f) for f in files]
        raw = pd.concat(frames, ignore_index=True)
        n_raw = len(raw)
        df = CleanSymbol(raw, sym, log)
        df = df[~df.index.duplicated()]
        df = df.reindex(cal)                         # missing hours -> NaN; NO fill
        for f in FIELDS:
            wide[f][sym] = df[f]
        valid = df["close"].notna()
        gaps = (~valid).astype(int)
        longest = int(gaps.groupby((gaps == 0).cumsum()).sum().max()) if (~valid).any() else 0
        quality.append(dict(symbol=sym, raw_rows=n_raw, expected_hours=len(cal), valid_hours=int(valid.sum()),
                            missing_hours=int((~valid).sum()), missing_pct=round(100 * (~valid).mean(), 3),
                            first_valid=df.index[valid.values][0] if valid.any() else None,
                            last_valid=df.index[valid.values][-1] if valid.any() else None,
                            longest_gap_hours=longest))
    out = {}
    for f in FIELDS:
        frame = pd.DataFrame(wide[f]).reindex(cal)
        frame.index.name = "timestamp"
        frame.to_csv(out_dir / f"{f}.csv.gz")
        out[f] = frame
    q = pd.DataFrame(quality)
    q.to_csv(out_dir / "data_quality.csv", index=False)
    pd.DataFrame(log, columns=["symbol", "issue", "count", "action"]).to_csv(out_dir / "exclusion_log.csv", index=False)
    out["quality"] = q
    return out


def LoadProcessed(cfg: dict | None = None, processed_dir: Path | None = None) -> dict:
    """Load wide frames: dict(field -> DataFrame[UTC hourly index x symbols])."""
    processed_dir = Path(processed_dir or Resolve(cfg, "processed"))
    out = {}
    for f in FIELDS:
        p = processed_dir / f"{f}.csv.gz"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing - run `python -m src.cli process` (or `synthetic`) first")
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df.index = UTCIndex(df.index)
        out[f] = df
    return out


def ProcessedHashes(processed_dir: Path) -> dict:
    return {p.name: sha256File(p) for p in sorted(Path(processed_dir).glob("*.csv*"))}
