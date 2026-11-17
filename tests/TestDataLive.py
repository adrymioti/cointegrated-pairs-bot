import io, tempfile, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

from src.DataLoader import ReadKlineBytes, CleanSymbol
from src.LiveFeed import BarAggregator, LiveRunner, ParseKlineMessage
from src.Cointegration import PairSpec
from src.Config import LoadConfig
from tests.Helpers import random_cointegrated


def _zip(rows, header=False):
    buf = io.StringIO()
    if header:
        buf.write("open_time,open,high,low,close,volume,close_time,qv,n,tb,tq,ig\n")
    for r in rows:
        buf.write(",".join(map(str, r)) + "\n")
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("x.csv", buf.getvalue())
    return b.getvalue()


def test_timestamp_units_ms_and_us():
    t_ms = 1_700_000_000_000
    row = lambda t, dt_: [t, 1, 2, 0.5, 1.5, 10, t + dt_, 15, 3, 1, 1, 0]
    a = ReadKlineBytes(_zip([row(t_ms, 3_599_999)]))
    b = ReadKlineBytes(_zip([row(t_ms * 1000, 3_599_999_000)], header=True))
    assert a["open_dt"].iloc[0] == b["open_dt"].iloc[0] == pd.Timestamp("2023-11-14 22:13:20", tz="UTC")
    assert b.attrs["timestamp_unit"] == "us"


def test_clean_symbol_flags_not_repairs():
    t = 1_700_000_000_000
    rows = [[t, 1, 2, 0.5, 1.5, 10, t + 3_599_999, 15, 3, 1, 1, 0],
            [t, 1, 2, 0.5, 1.5, 10, t + 3_599_999, 15, 3, 1, 1, 0],               # duplicate
            [t + 3_600_000, 1, 0.5, 2, 1.5, 10, t + 7_199_999, 15, 3, 1, 1, 0]]   # high < low
    log = []
    df = CleanSymbol(ReadKlineBytes(_zip(rows)), "TEST", log)
    assert {l["issue"] for l in log} == {"duplicate_timestamp", "invalid_ohlcv"}
    assert df["close"].isna().sum() == 1 and len(df) == 2


def test_parse_only_closed_candles():
    msg = {"data": {"k": {"s": "BTCUSDT", "t": 1_700_000_000_000, "o": "1", "c": "2", "q": "5", "x": False}}}
    assert ParseKlineMessage(msg) is None
    msg["data"]["k"]["x"] = True
    assert ParseKlineMessage(msg)[0] == "BTCUSDT"


def test_aggregator_grace_gives_nan_not_fill():
    agg = BarAggregator(["A", "B"], grace_seconds=600)
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    agg.add("A", t0, 1.0, 1.1)
    assert agg.ready(t0 + pd.Timedelta(hours=1)) == []
    out = agg.ready(t0 + pd.Timedelta(hours=1, minutes=11))
    assert len(out) == 1 and np.isnan(out[0][1]["B"])
    agg.add("A", t0, 1.0, 1.1)                                   # late duplicate for an emitted hour
    assert agg.ready(t0 + pd.Timedelta(hours=5)) == []


def test_restart_cannot_duplicate_trades():
    cfg = LoadConfig()
    opn, close = random_cointegrated(n=1800, seed=9)
    from src.Cointegration import olsHedge
    a, b, _ = olsHedge(np.log(close["X"].values[:900]), np.log(close["Y"].values[:900]))
    spec = PairSpec("X-Y", "X", "Y", a, b)
    hist = close.iloc[:400]
    live = list(range(400, 1800))

    def feed(runner, idxs):
        for i in idxs:
            runner.on_bar(close.index[i], dict(opn.iloc[i]), dict(close.iloc[i]))

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        r_cont = LiveRunner(cfg, [spec], hist, Path(d1))
        feed(r_cont, live)
        r1 = LiveRunner(cfg, [spec], hist, Path(d2))
        feed(r1, live[:600])
        r2 = LiveRunner(cfg, [spec], hist, Path(d2))             # "restart": state + bar log reloaded
        feed(r2, live[500:])                                     # replays 100 old bars: must be ignored
        assert len(r_cont.engine.ledger) > 3
        assert len(r2.engine.ledger) == len(r_cont.engine.ledger)
        assert abs(r2.engine.equity_last - r_cont.engine.equity_last) < 1e-6
