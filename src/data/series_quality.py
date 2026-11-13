"""Calidad de una serie OHLCV por año: cierres repetidos, volumen cero, OHLC inconsistente.

Complementa `quality.py` (máscara causal por fila): aquí se detectan *regímenes* completos
de mala calidad (p. ej. un feed que repite el cierre durante años).
"""
from __future__ import annotations

import pandas as pd


def series_quality_by_year(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Por año: n_rows, pct_repeated_close, pct_zero_volume, n_bad_ohlc."""
    c = ohlcv["Close"].astype(float)
    h, l = ohlcv["High"].astype(float), ohlcv["Low"].astype(float)
    o, v = ohlcv["Open"].astype(float), ohlcv["Volume"].astype(float)
    malo = (h < l) | (o > h) | (o < l) | (c > h) | (c < l) | (ohlcv[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    f = pd.DataFrame({
        "rep": (c == c.shift(1)) & c.notna(),
        "zero": v.fillna(0) == 0,
        "bad": malo,
    })
    g = f.groupby(pd.DatetimeIndex(ohlcv.index).year)
    out = pd.DataFrame({
        "n_rows": g.size(),
        "pct_repeated_close": g["rep"].mean() * 100,
        "pct_zero_volume": g["zero"].mean() * 100,
        "n_bad_ohlc": g["bad"].sum().astype(int),
    })
    out.index.name = "year"
    return out.reset_index()


def low_quality_years(
    por_ticker: dict[str, pd.DataFrame], *, max_repeated: float = 20.0, max_zero_vol: float = 10.0,
) -> list[dict]:
    """(ticker, año) con > max_repeated % de cierres repetidos o > max_zero_vol % de volumen cero."""
    out = []
    for t, q in por_ticker.items():
        for r in q.itertuples():
            if r.pct_repeated_close > max_repeated or r.pct_zero_volume > max_zero_vol:
                out.append({"ticker": t, "year": int(r.year), "pct_repeated_close": round(float(r.pct_repeated_close), 1),
                            "pct_zero_volume": round(float(r.pct_zero_volume), 1)})
    return out
