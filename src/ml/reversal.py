"""M4 · Señales de reversión de corto plazo y retornos futuros por tipo de entrada.

Todo trabaja con matrices anchas (filas = fechas, columnas = tickers) de Open/Close/Volume.
Las señales son **causales**: en la fila ``t`` solo usan datos hasta el cierre de ``t``.
Los retornos futuros (``forward_returns``) miran adelante a propósito: son el objetivo, no una señal.
Todos los retornos están en **fracciones** (no en %).

Convención de signo: las señales son retornos "crudos" (``ret_5d`` alto = subió).
Para medir reversión el estudio usa ``score = -señal`` (los perdedores arriba).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BETA_WINDOW = 126
SIGNAL_NAMES = (
    "ret_1d", "ret_2d", "ret_5d", "ret_21d",
    "overnight_1d", "overnight_5d", "intraday_1d", "intraday_5d", "resid_5d",
)
ENTRY_NAMES = ("close_close", "open_close", "open_open", "skip_close")


def ret_nd(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Retorno de cierre a cierre de los últimos ``n`` días: ``C_t / C_{t-n} - 1``."""
    return close / close.shift(n) - 1


def _rolling_compound(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Compone ``n`` retornos diarios consecutivos (si falta alguno, queda NaN)."""
    return np.exp(np.log1p(x).rolling(n, min_periods=n).sum()) - 1


def overnight(open_: pd.DataFrame, close: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    """Parte nocturna: compone ``O_d / C_{d-1} - 1`` de los últimos ``n`` días hasta ``t``."""
    gap = open_ / close.shift(1) - 1
    return gap if n == 1 else _rolling_compound(gap, n)


def intraday(open_: pd.DataFrame, close: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    """Parte intradía: compone ``C_d / O_d - 1`` de los últimos ``n`` días hasta ``t``."""
    day = close / open_ - 1
    return day if n == 1 else _rolling_compound(day, n)


def market_return(close: pd.DataFrame) -> pd.Series:
    """Retorno diario del mercado = promedio equiponderado de los retornos diarios disponibles."""
    return close.pct_change(fill_method=None).mean(axis=1)


def rolling_beta(
    close: pd.DataFrame, mkt_ret: pd.Series, window: int = BETA_WINDOW
) -> pd.DataFrame:
    """Beta móvil de cada ticker contra el mercado, **rezagada un día** (``shift(1)``).

    La beta de la fila ``t`` usa retornos diarios hasta ``t-1``.
    """
    r = close.pct_change(fill_method=None)
    mp = max(window // 2, 20)
    var_m = mkt_ret.rolling(window, min_periods=mp).var()
    beta = r.apply(lambda col: col.rolling(window, min_periods=mp).cov(mkt_ret)).div(var_m, axis=0)
    return beta.shift(1)


def resid_5d(close: pd.DataFrame, window: int = BETA_WINDOW) -> pd.DataFrame:
    """Retorno de 5 días menos ``beta × retorno de 5 días del mercado`` (beta con ``shift(1)``)."""
    m = market_return(close)
    mkt5 = (1 + m.fillna(0)).cumprod()
    mkt5 = mkt5 / mkt5.shift(5) - 1
    beta = rolling_beta(close, m, window)
    return ret_nd(close, 5) - beta.mul(mkt5, axis=0)


def build_signals(open_: pd.DataFrame, close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Las 9 señales de ``SIGNAL_NAMES`` (matrices anchas, causales)."""
    return {
        "ret_1d": ret_nd(close, 1),
        "ret_2d": ret_nd(close, 2),
        "ret_5d": ret_nd(close, 5),
        "ret_21d": ret_nd(close, 21),
        "overnight_1d": overnight(open_, close, 1),
        "overnight_5d": overnight(open_, close, 5),
        "intraday_1d": intraday(open_, close, 1),
        "intraday_5d": intraday(open_, close, 5),
        "resid_5d": resid_5d(close),
    }


def forward_returns(
    open_: pd.DataFrame, close: pd.DataFrame, horizon: int = 5
) -> dict[str, pd.DataFrame]:
    """Retornos futuros según la entrada, con la señal medida al cierre de ``t``.

    - ``close_close``: ``C_{t+h} / C_t``  (entrada al cierre de t, no ejecutable).
    - ``open_close``: ``C_{t+h+1} / O_{t+1}``  (regla del backtest: Open t+1 → Close t+h+1).
    - ``open_open``: ``O_{t+h+1} / O_{t+1}``.
    - ``skip_close``: ``C_{t+h+1} / C_{t+1}``  (salta un día: descarta el rebote bid-ask).
    """
    h = horizon
    return {
        "close_close": close.shift(-h) / close - 1,
        "open_close": close.shift(-(h + 1)) / open_.shift(-1) - 1,
        "open_open": open_.shift(-(h + 1)) / open_.shift(-1) - 1,
        "skip_close": close.shift(-(h + 1)) / close.shift(-1) - 1,
    }


def adv(close: pd.DataFrame, volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volumen operado promedio (en dinero) de los últimos ``window`` días hasta ``t``."""
    return (close * volume).rolling(window, min_periods=window).mean()


def abnormal_volume(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volumen de ``t`` entre el promedio de los ``window`` días **anteriores** (excluye ``t``)."""
    base = volume.rolling(window, min_periods=window).mean().shift(1)
    return volume / base.where(base > 0)


def tercile_by_date(x: pd.DataFrame) -> pd.DataFrame:
    """Tercil por fecha (0 = más bajo, 2 = más alto) entre los tickers con dato ese día."""
    pos = (x.rank(axis=1, method="first") - 1).div(x.notna().sum(axis=1), axis=0)
    return np.floor(pos * 3).where(x.notna())


def to_panel(wide: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Matrices anchas → panel largo: índice = fecha, columnas ``ticker`` + una por matriz."""
    parts = {k: v.stack(future_stack=True) for k, v in wide.items()}
    df = pd.DataFrame(parts)
    df.index.names = ["_date", "ticker"]
    df = df.reset_index().set_index("_date")
    df.index.name = None
    return df.sort_index(kind="stable")


def block_means(ic: pd.Series, n_blocks: int = 5) -> list[dict]:
    """Media del IC en ``n_blocks`` bloques contiguos con igual número de fechas."""
    x = ic.dropna().sort_index()
    out = []
    for part in np.array_split(np.arange(len(x)), n_blocks):
        if len(part) == 0:
            continue
        seg = x.iloc[part]
        out.append({
            "desde": str(seg.index[0].date()), "hasta": str(seg.index[-1].date()),
            "n": int(len(seg)), "mean": float(seg.mean()),
        })
    return out


def weekly_bottom_weights(score: pd.DataFrame, k: int, step: int = 5) -> pd.DataFrame:
    """Pesos 1/k en los ``k`` tickers con mayor ``score`` cada ``step`` fechas (sin solape).

    Desempata por orden de columna. Filas = fechas de rebalanceo; columnas = tickers.
    """
    rows = {}
    for d in score.index[::step]:
        s = score.loc[d].dropna()
        if len(s) < k:
            continue
        top = s.sort_values(ascending=False, kind="stable").index[:k]
        w = pd.Series(0.0, index=score.columns)
        w[top] = 1.0 / k
        rows[d] = w
    return pd.DataFrame(rows).T
