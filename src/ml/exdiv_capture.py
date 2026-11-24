"""Captura de la caída parcial del ex-dividendo (N3): funciones puras.

Idea: la acción baja el día ex solo una parte del dividendo; comprar en el Close de
`p - window` y vender en el Close de `p` (día ex) cobra el dividendo completo y cede
solo la caída. Ejecución al CIERRE (no al Open). Los precios no están ajustados por
dividendo, así que la caída es real.

Dos formas de conocer el evento:
- `event_returns`: con la ex-fecha REAL (oráculo; cota superior, sirve para replicar).
- `causal_trades`: la fecha se prevé con la regla "misma ex-fecha del año anterior"
  (+364 días, mismo día de la semana) usando solo ex-fechas pasadas. Es la versión operable.

Convención: una ex-fecha se asigna al primer día hábil del índice >= esa fecha. Las ventanas
se cuentan en filas (días hábiles).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRED_SHIFT_DAYS = 364  # un año menos un día: conserva el día de la semana


def _positions(idx: pd.DatetimeIndex, dates: pd.DatetimeIndex) -> np.ndarray:
    """Posición del primer día hábil de `idx` >= cada fecha (len(idx) si no existe)."""
    return idx.searchsorted(pd.DatetimeIndex(dates), side="left")


def _clean_dividends(dividends: pd.Series) -> pd.Series:
    d = pd.Series(dividends).astype(float).dropna()
    d.index = pd.DatetimeIndex(d.index).as_unit("ns").normalize()
    return d[d > 0].groupby(level=0).sum().sort_index()


def _by_position(idx: pd.DatetimeIndex, div: pd.Series) -> pd.Series:
    """Dividendos sumados por posición de día hábil (descarta los posteriores al último precio)."""
    pos = _positions(idx, div.index)
    s = pd.Series(div.to_numpy(), index=pos)
    s = s[s.index < len(idx)]
    return s.groupby(level=0).sum()


def net_return(gross: pd.Series, yld: pd.Series, *, tax: float = 0.0, cost_per_side: float = 0.003) -> pd.Series:
    """Retorno neto: bruto − retención sobre el dividendo (tax·yield) − ida y vuelta (2·costo por lado)."""
    return gross - tax * yld - 2.0 * cost_per_side


def event_returns(
    close: pd.Series,
    dividends: pd.Series,
    window: int = 5,
    max_yield: float | None = 0.15,
) -> pd.DataFrame:
    """Un renglón por evento ex-dividendo (índice = fecha de venta, el día ex en el índice de precios).

    Columnas: ex_date, buy_date, buy, sell, div, yield (div / buy), gross = (sell + div) / buy − 1.
    Se descartan eventos sin `window` días previos y con yield > `max_yield` (extraordinarios).
    """
    c = close.dropna()
    idx = pd.DatetimeIndex(c.index)
    div = _clean_dividends(dividends)
    cols = ["ex_date", "buy_date", "buy", "sell", "div", "yield", "gross"]
    if div.empty or len(idx) == 0:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="Date"))
    pos = _positions(idx, div.index)
    ok = (pos >= window) & (pos < len(idx))
    pos, ex, amt = pos[ok], div.index[ok], div.to_numpy()[ok]
    # dos dividendos en el mismo día hábil: se suman
    df = pd.DataFrame({"p": pos, "ex_date": ex, "div": amt}).groupby("p", as_index=False).agg(
        ex_date=("ex_date", "first"), div=("div", "sum"))
    p = df["p"].to_numpy()
    cv = c.to_numpy(dtype=float)
    buy, sell = cv[p - window], cv[p]
    out = pd.DataFrame({
        "ex_date": df["ex_date"].to_numpy(),
        "buy_date": idx[p - window],
        "buy": buy,
        "sell": sell,
        "div": df["div"].to_numpy(),
    }, index=pd.DatetimeIndex(idx[p], name="Date"))
    out["yield"] = out["div"] / out["buy"]
    out["gross"] = (out["sell"] + out["div"]) / out["buy"] - 1.0
    out = out[np.isfinite(out["gross"]) & (out["buy"] > 0)]
    if max_yield is not None:
        out = out[out["yield"] <= max_yield]
    return out[cols]


def predict_exdate(history: pd.DatetimeIndex, today: pd.Timestamp) -> pd.Timestamp:
    """Próxima ex-fecha prevista (> today) con la regla causal: ex-fecha pasada + 364 días.

    Solo usa eventos de `history` con fecha <= `today`. Devuelve la más cercana, o NaT si
    ningún evento del último año la genera.
    """
    h = pd.DatetimeIndex(history)
    h = h[h <= today]
    if len(h) == 0:
        return pd.NaT
    cand = h + pd.Timedelta(days=PRED_SHIFT_DAYS)
    cand = cand[cand > today]
    return cand.min() if len(cand) else pd.NaT


def causal_trades(
    close: pd.Series,
    dividends: pd.Series,
    window: int = 5,
    max_yield: float | None = 0.15,
) -> pd.DataFrame:
    """Operaciones que se habrían abierto usando solo la regla causal de fecha (índice = día de venta).

    Cada ex-fecha pasada `h` prevé `P = h + 364 d` (primer día hábil >= P), compra en `P − window`
    y vende en `P`. Retorno bruto = (sell + dividendos con ex-fecha REAL en (compra, venta]) / buy − 1;
    si la fecha se falla, puede no cobrar nada y se cuenta igual. `yield` es causal: dividendo de
    `h` / precio de compra. `offset` = distancia (días hábiles) al evento real más cercano
    (NaN si no hay); `n_div_captured` = nº de dividendos reales dentro de la ventana.
    """
    c = close.dropna()
    idx = pd.DatetimeIndex(c.index)
    div = _clean_dividends(dividends)
    cols = ["hist_ex", "pred_ex", "buy_date", "buy", "sell", "div_hist", "yield", "gross",
            "n_div_captured", "offset"]
    empty = pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="Date"))
    if div.empty or len(idx) == 0:
        return empty
    actual = _by_position(idx, div)
    act_pos = actual.index.to_numpy()
    cv = c.to_numpy(dtype=float)
    rows = []
    for h, amt in div.items():
        p = int(idx.searchsorted(h + pd.Timedelta(days=PRED_SHIFT_DAYS), side="left"))
        if p >= len(idx) or p < window:
            continue
        buy, sell = cv[p - window], cv[p]
        if not (buy > 0 and np.isfinite(buy) and np.isfinite(sell)):
            continue
        y = amt / buy
        if max_yield is not None and y > max_yield:
            continue
        inside = (act_pos > p - window) & (act_pos <= p)
        got = float(actual.to_numpy()[inside].sum())
        off = float((act_pos - p)[np.abs(act_pos - p).argmin()]) if len(act_pos) else np.nan
        rows.append({
            "hist_ex": h, "pred_ex": idx[p], "buy_date": idx[p - window], "buy": buy, "sell": sell,
            "div_hist": amt, "yield": y, "gross": (sell + got) / buy - 1.0,
            "n_div_captured": int(inside.sum()), "offset": off,
        })
    if not rows:
        return empty
    out = pd.DataFrame(rows)
    # dos previsiones al mismo día de venta: se conserva la de mayor dividendo histórico
    out = out.sort_values("div_hist").drop_duplicates("pred_ex", keep="last").sort_values("pred_ex")
    out.index = pd.DatetimeIndex(out["pred_ex"].to_numpy(), name="Date")
    return out[cols]


def quarter_mean_t(r: pd.Series) -> dict:
    """Media por evento y estadístico t sobre medias por trimestre (el n honesto = trimestres)."""
    r = pd.Series(r).dropna()
    if r.empty:
        return {"n": 0, "n_q": 0, "mean": np.nan, "mean_q": np.nan, "std_q": np.nan, "t": np.nan}
    q = r.groupby(pd.DatetimeIndex(r.index).to_period("Q")).mean()
    n_q = int(len(q))
    sd = float(q.std(ddof=1)) if n_q > 1 else np.nan
    t = float(q.mean() / (sd / np.sqrt(n_q))) if n_q > 1 and sd > 0 else np.nan
    return {"n": int(len(r)), "n_q": n_q, "mean": float(r.mean()), "mean_q": float(q.mean()),
            "std_q": sd, "t": t}


def top_n_share(values: pd.Series, n: int = 5) -> dict:
    """Cuánto de la suma viene de los `n` mejores eventos y la media sin ellos."""
    v = pd.Series(values).dropna().sort_values(ascending=False)
    if len(v) <= n:
        return {"share": np.nan, "mean_ex_top": np.nan, "n": int(len(v))}
    total = float(v.sum())
    return {"share": float(v.iloc[:n].sum() / total) if total != 0 else np.nan,
            "mean_ex_top": float(v.iloc[n:].mean()), "n": int(len(v))}
