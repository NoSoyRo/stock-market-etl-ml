"""Tests del archivo de vintages (datos sintéticos, sin red ni datos reales)."""
from __future__ import annotations

import pandas as pd

from scripts.archive_vintage import build_snapshot, write_snapshot


def _ohlcv(n: int = 30, base: float = 10.0) -> pd.DataFrame:
    idx = pd.bdate_range("2026-09-01", periods=n)
    return pd.DataFrame(
        {"Open": base, "High": base + 1, "Low": base - 1, "Close": base + 0.5, "Volume": 1000},
        index=pd.DatetimeIndex(idx, name="Date"),
    )


def _getter(data: dict):
    return lambda t: data.get(t)


def test_snapshot_ultimas_filas_por_ticker():
    snap = build_snapshot(["A.MX", "B.MX"], _getter({"A.MX": _ohlcv(), "B.MX": _ohlcv(base=5)}), n=10)
    assert set(snap["ticker"]) == {"A.MX", "B.MX"}
    assert (snap.groupby("ticker").size() == 10).all()
    a = snap[snap.ticker == "A.MX"]
    assert a["date"].max() == _ohlcv().index.max()  # las ÚLTIMAS filas


def test_snapshot_omite_vacios_y_faltantes():
    snap = build_snapshot(["A.MX", "VACIO.MX", "NO.MX"],
                          _getter({"A.MX": _ohlcv(), "VACIO.MX": pd.DataFrame()}), n=10)
    assert set(snap["ticker"]) == {"A.MX"}


def test_dry_run_no_escribe(tmp_path):
    snap = build_snapshot(["A.MX"], _getter({"A.MX": _ohlcv()}), n=10)
    res = write_snapshot(snap, "2026-11-12", tmp_path, dry_run=True)
    assert res["estado"] == "dry-run"
    assert list(tmp_path.iterdir()) == []


def test_idempotente_mismo_dia(tmp_path):
    snap = build_snapshot(["A.MX"], _getter({"A.MX": _ohlcv()}), n=10)
    r1 = write_snapshot(snap, "2026-11-12", tmp_path)
    assert r1["estado"] == "creado"
    # Segunda corrida el mismo día, con datos "distintos": no duplica ni sobrescribe
    snap2 = build_snapshot(["A.MX"], _getter({"A.MX": _ohlcv(base=99)}), n=10)
    r2 = write_snapshot(snap2, "2026-11-12", tmp_path)
    assert r2["estado"] == "existe"
    files = list(tmp_path.glob("*.parquet"))
    assert [f.name for f in files] == ["2026-11-12.parquet"]
    assert len(pd.read_parquet(files[0])) == 10
    assert pd.read_parquet(files[0])["Close"].iloc[0] == 10.5  # conserva el primero


def test_dia_distinto_crea_otro_archivo(tmp_path):
    snap = build_snapshot(["A.MX"], _getter({"A.MX": _ohlcv()}), n=10)
    write_snapshot(snap, "2026-11-12", tmp_path)
    write_snapshot(snap, "2026-11-13", tmp_path)
    assert len(list(tmp_path.glob("*.parquet"))) == 2
