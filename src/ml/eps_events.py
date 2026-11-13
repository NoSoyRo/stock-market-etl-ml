"""N5 · Eventos de EPS: ventana "con anuncio reciente" y fecha esperada de resultados.

Todo es **causal**: un anuncio solo existe para la fecha ``t`` si ``available_from <= t``
(el día hábil siguiente a ``announced``; el anuncio puede salir tras el cierre).
``history`` = DataFrame de UN ticker con columnas ``announced`` y ``available_from``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.vol_calendar import TOL_EARN, _snap_weekday, bdays_offset, event_features

SHIFT_DAYS = 364        # mismo día de la semana un año después
PAIR_DAYS = (300, 430)  # emparejar un anuncio con el del mismo trimestre del año anterior


def has_recent_event(announced_df: pd.DataFrame, dates: pd.DatetimeIndex, window: int = 5) -> pd.Series:
    """True en ``t`` si algún anuncio ya conocido cae en las últimas ``window`` fechas de la malla.

    El anuncio se ancla a la primera fecha de ``dates`` >= ``available_from`` (posición ``p``);
    cuenta en ``t`` (posición ``q``) si ``0 <= q - p < window``.
    """
    dates = pd.DatetimeIndex(dates)
    flag = np.zeros(len(dates), dtype=bool)
    for av in pd.DatetimeIndex(announced_df["available_from"]).dropna():
        p = int(dates.searchsorted(av, side="left"))
        if p < len(dates):
            flag[p:p + window] = True
    return pd.Series(flag, index=dates)


def expected_events(history: pd.DataFrame) -> pd.DataFrame:
    """Fechas esperadas = anuncio + 364 días (a día hábil) y desde cuándo se conocen (``known_from``)."""
    if len(history) == 0:
        return pd.DataFrame({"expected": pd.DatetimeIndex([]), "known_from": pd.DatetimeIndex([])})
    ann = pd.DatetimeIndex(history["announced"])
    exp = [_snap_weekday(a + pd.Timedelta(days=SHIFT_DAYS)) for a in ann]
    return pd.DataFrame({"expected": pd.DatetimeIndex(exp),
                         "known_from": pd.DatetimeIndex(history["available_from"])}).reset_index(drop=True)


def expected_next_date(history: pd.DataFrame, today: pd.Timestamp) -> pd.Timestamp:
    """Próxima fecha esperada >= ``today`` usando solo anuncios con ``available_from <= today``."""
    ex = expected_events(history)
    ex = ex[(ex["known_from"] <= today) & (ex["expected"] >= today)]
    return ex["expected"].min() if len(ex) else pd.NaT


def expected_flag(index: pd.DatetimeIndex, history: pd.DataFrame, *, before: int = 2, after: int = 8) -> pd.Series:
    """1 si hay una fecha esperada (ya conocida) en ``(t - before, t + after]`` días hábiles.

    NaN antes de conocer el primer anuncio y después de la última fecha esperada
    (sin información de calendario).
    """
    index = pd.DatetimeIndex(index)
    ex = expected_events(history)
    if ex.empty:
        return pd.Series(np.nan, index=index)
    out = np.zeros(len(index))
    for e, kf in zip(ex["expected"], ex["known_from"]):
        off = bdays_offset(index, e)
        out = np.where((index >= kf) & (off > -before) & (off <= after), 1.0, out)
    s = pd.Series(out, index=index)
    s[(index < ex["known_from"].min()) | (index > ex["expected"].max())] = np.nan
    return s


def expected_accuracy(history: pd.DataFrame) -> pd.Series:
    """Error (días hábiles, real − esperado) de la fecha esperada de cada anuncio con par del año anterior."""
    h = history.sort_values("announced").reset_index(drop=True)
    ann = pd.DatetimeIndex(h["announced"])
    errs = []
    for i, a in enumerate(ann):
        gap = (a - ann[:i]).days
        j = np.where((gap >= PAIR_DAYS[0]) & (gap <= PAIR_DAYS[1]))[0]
        if len(j) == 0:
            continue
        exp = _snap_weekday(ann[j[-1]] + pd.Timedelta(days=SHIFT_DAYS))
        errs.append(int(np.busday_count(exp.date(), a.date())))
    return pd.Series(errs, dtype=float)


def earnings_eps_features(index: pd.DatetimeIndex, history: pd.DataFrame) -> pd.DataFrame:
    """``earn_in5``, ``earn_in21``, ``earn_days`` con la fecha esperada a partir de fechas reales de EPS."""
    cols = ["earn_in5", "earn_in21", "earn_days"]
    ex = expected_events(history)
    if ex.empty:
        return pd.DataFrame(np.nan, index=index, columns=cols)
    out = event_features(index, pd.DatetimeIndex(ex["expected"]), (5, 21), tol=TOL_EARN,
                         known_from=pd.DatetimeIndex(ex["known_from"]), prefix="earn")
    out.loc[out.index < ex["known_from"].min()] = np.nan
    return out[cols]


def sharpe_diff_ci(a: pd.Series, b: pd.Series, *, periods_per_year: float, block: int = 4,
                   n_boot: int = 2000, seed: int = 0) -> dict:
    """Δ Sharpe anualizado (a − b) con IC95% por bootstrap de bloques móviles sobre periodos pareados."""
    df = pd.concat([a, b], axis=1).dropna()
    x = df.to_numpy(dtype=float)
    n = len(x)
    ann = np.sqrt(periods_per_year)

    def sharpe(m: np.ndarray) -> np.ndarray:
        sd = m.std(axis=-2, ddof=1)
        return np.where(sd > 0, m.mean(axis=-2) / np.where(sd > 0, sd, 1.0), np.nan) * ann

    pt = sharpe(x)
    if n < 8:
        return {"n": n, "sharpe_a": float(pt[0]), "sharpe_b": float(pt[1]), "delta": float(pt[0] - pt[1]),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    bl = min(max(int(block), 1), n)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / bl))
    starts = rng.integers(0, n - bl + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(bl)).reshape(n_boot, -1)[:, :n]
    s = sharpe(x[idx])                       # (n_boot, 2)
    d = s[:, 0] - s[:, 1]
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return {"n": n, "sharpe_a": float(pt[0]), "sharpe_b": float(pt[1]), "delta": float(pt[0] - pt[1]),
            "ci_lo": float(lo), "ci_hi": float(hi)}
