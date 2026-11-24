"""Dividendos y retorno total, sin tocar los parquet de precios (M3b).

`store._download` guarda solo OHLCV sin dividendos; aquí se bajan aparte
(`Ticker.dividends`) y se combinan con el precio. La red solo se usa en
`fetch_dividends` (inyectable para pruebas); el resto son funciones puras.

Convención: una fecha ex-dividendo se asigna al primer día hábil del índice de
precios >= esa fecha. Las ventanas se cuentan en FILAS (días hábiles), igual que
el `horizon` de `compute_target`.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

DIVIDENDS_DIR = Path(__file__).parent.parent.parent / "data" / "dividends"


def _safe(symbol: str) -> str:
    return symbol.replace("/", "_").replace("^", "")


def normalize_dividends(raw: pd.Series | None) -> pd.Series:
    """Serie de dividendos con índice = fecha local (sin zona horaria, a medianoche), sin ceros ni duplicados."""
    if raw is None or len(raw) == 0:
        return pd.Series([], index=pd.DatetimeIndex([], name="Date"), dtype=float, name="dividend")
    s = pd.Series(raw).astype(float)
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)  # conserva la hora local; luego solo la fecha
    s.index = idx.normalize()
    s = s[s > 0].groupby(level=0).sum().sort_index()
    s.index.name = "Date"
    return s.rename("dividend")


def _yf_dividends(symbol: str) -> pd.Series:
    import yfinance as yf

    return yf.Ticker(symbol).dividends


def fetch_dividends(
    symbol: str,
    *,
    cache_dir: Path | None = None,
    max_age_days: float = 7.0,
    force: bool = False,
    retries: int = 3,
    pause: float = 1.0,
    fetcher: Callable[[str], pd.Series] | None = None,
) -> pd.Series:
    """Dividendos de `symbol` (con caché en parquet). Reintenta si Yahoo falla; relanza el último error."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DIVIDENDS_DIR
    path = cache_dir / f"{_safe(symbol)}.parquet"
    if path.exists() and not force:
        if (time.time() - path.stat().st_mtime) / 86400 <= max_age_days:
            return pd.read_parquet(path)["dividend"]
    fetch = fetcher or _yf_dividends
    last: Exception | None = None
    for i in range(max(retries, 1)):
        try:
            s = normalize_dividends(fetch(symbol))
            cache_dir.mkdir(parents=True, exist_ok=True)
            s.to_frame().to_parquet(path)
            return s
        except Exception as exc:  # red/Yahoo: reintentar
            last = exc
            if i < retries - 1:
                time.sleep(pause * (2 ** i))
    assert last is not None
    raise last


def _div_by_position(index: pd.DatetimeIndex, dividends: pd.Series) -> np.ndarray:
    """Monto de dividendo por fila del índice (ex-fechas fuera del rango se ignoran)."""
    out = np.zeros(len(index))
    if len(dividends) == 0 or len(index) == 0:
        return out
    pos = index.searchsorted(dividends.index, side="left")
    ok = pos < len(index)
    np.add.at(out, pos[ok], dividends.values[ok])
    return out


def total_return_series(ohlcv: pd.DataFrame, dividends: pd.Series) -> pd.Series:
    """Retorno diario total (fracción): (Close_t + D_t) / Close_{t-1} − 1, con D_t en la fecha ex-dividendo."""
    close = ohlcv["Close"]
    d = pd.Series(_div_by_position(close.index, dividends), index=close.index)
    return ((close + d) / close.shift(1) - 1).rename("total_return")


def forward_dividend_sum(index: pd.DatetimeIndex, dividends: pd.Series, horizon: int) -> pd.Series:
    """Suma de dividendos con ex-fecha en (t, t+horizon] (filas t+1..t+horizon)."""
    d = _div_by_position(index, dividends)
    cs = np.concatenate([[0.0], np.cumsum(d)])  # cs[k] = suma de las primeras k filas
    n = len(index)
    t = np.arange(n)
    hi = np.minimum(t + horizon, n - 1)
    return pd.Series(cs[hi + 1] - cs[t + 1], index=index, name="div_fwd")


def ex_div_flag(index: pd.DatetimeIndex, dividends: pd.Series, window: int) -> pd.Series:
    """True si hay al menos un ex-dividendo en (t, t+window]."""
    return (forward_dividend_sum(index, dividends, window) > 0).rename("ex_div_flag")


def add_dividends_to_forward_return(
    fwd_ret: pd.Series,
    dividends: pd.Series,
    horizon: int,
    *,
    close: pd.Series,
) -> pd.Series:
    """Suma al `fwd_ret` (en POR CIENTO) el rendimiento por dividendos de la ventana.

    Aproximación: dividendos en (t, t+horizon] / Close_t × 100, sin reinversión.
    Se necesita `close` porque el dividendo está en pesos y el retorno en %.
    """
    div = forward_dividend_sum(fwd_ret.index, dividends, horizon)
    c = close.reindex(fwd_ret.index)
    return (fwd_ret + div / c * 100).rename(fwd_ret.name)
