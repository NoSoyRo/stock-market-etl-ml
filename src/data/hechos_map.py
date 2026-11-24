"""Mapeo código de `hechos` → ticker MX (funciones puras, sin E/S).

`hechos` trae ~5,000 códigos (acciones, series, ETFs, SIC...). Para usar el flujo por casa hay
que saber cuál código es cada ticker del universo. Dos métodos, en orden:

  a) **nombre / alias**: el código normalizado (sin `*`, sin `.MX`) coincide con el ticker, o está
     en la tabla explícita de alias (p. ej. `GENTERA` → `GENTERA*`).
  b) **correlación de volumen**: para los no resueltos, se correlaciona el log del volumen diario
     comprado en `hechos` (`vol_buy`) con el log del volumen de mercado (Yahoo) en los mismos días.

Reglas de confianza (escritas ANTES de ver resultados, de la idea 4.6):
  - **alta**: corr ≥ 0.90 y el segundo mejor código < 0.60 (o no hay segundo).
  - **media**: corr ≥ 0.70 (el mejor es claro pero no cumple alta).
  - **baja**: ambiguo (por correlación, el segundo mejor ≥ 0.60), un nombre/alias con corr < 0.70,
    o un nombre/alias que no se pudo validar (sin volumen de mercado).
  - Por correlación, si el mejor < 0.70 el ticker queda **sin mapear**.
Desempate: a igual correlación (diferencia < 1e-9) gana el código con mayor volumen total.
Los métodos `nombre` y `alias` también se validan con la correlación (sin la regla del segundo
mejor); así un nombre que coincide por casualidad (otro instrumento) no sale como "alta".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALTA_CORR = 0.90
ALTA_SEGUNDO_MAX = 0.60
MEDIA_CORR = 0.70
MIN_DIAS = 60  # días en común mínimos para correlacionar

# Alias explícitos: ticker de mercado -> códigos de `hechos` posibles (renombres conocidos).
ALIASES: dict[str, list[str]] = {
    "GENTERA": ["GENTERA*"],
}


def normalize_code(code: str) -> str:
    """Mayúsculas, sin espacios, sin sufijo `.MX` ni asterisco final."""
    c = str(code).strip().upper()
    if c.endswith(".MX"):
        c = c[:-3]
    return c.rstrip("*")


def log_volume(vol: pd.Series) -> pd.Series:
    """log del volumen, descartando días con volumen ≤ 0 o vacío."""
    v = pd.to_numeric(vol, errors="coerce")
    return np.log(v.where(v > 0))


def correlate_codes(target: pd.Series, wide: pd.DataFrame, min_days: int = MIN_DIAS) -> pd.Series:
    """Correlación de log-volumen entre `target` (fecha→volumen) y cada columna de `wide`.

    Solo cuenta días en que ambos tienen dato; columnas con menos de `min_days` días comunes
    se descartan. Devuelve la serie ordenada de mayor a menor.
    """
    t = log_volume(target).dropna()
    sub = np.log(wide.where(wide > 0)).reindex(t.index)
    n = sub.notna().sum()
    corr = sub.corrwith(t).dropna()  # pandas usa solo los días con dato en ambos
    return corr[n.reindex(corr.index) >= min_days].sort_values(ascending=False)


def _label(corr: float, segundo: float | None, por_correlacion: bool) -> str:
    # El segundo mejor solo importa si la elección fue por correlación; si el nombre ya casa,
    # otro código parecido (p. ej. otra emisora del mismo sector) no vuelve ambiguo el mapeo.
    if por_correlacion and segundo is not None and segundo >= ALTA_SEGUNDO_MAX and corr >= ALTA_CORR:
        return "baja"  # ambiguo: hay otro código casi igual de parecido
    if corr >= ALTA_CORR:
        return "alta"
    if corr >= MEDIA_CORR:
        return "media"
    return "baja"


def resolve_ticker(
    ticker: str,
    target: pd.Series,
    wide: pd.DataFrame,
    aliases: dict[str, list[str]] | None = None,
    min_days: int = MIN_DIAS,
) -> dict:
    """Resuelve el código de `hechos` de un ticker. Devuelve dict con código, método y confianza."""
    aliases = ALIASES if aliases is None else aliases
    corr = correlate_codes(target, wide, min_days)
    total = wide.sum()

    def _pick_by_name() -> tuple[str | None, str]:
        for cand in aliases.get(ticker, []):
            if cand in wide.columns:
                return cand, "alias"
        norm = normalize_code(ticker)
        hits = [c for c in wide.columns if normalize_code(c) == norm]
        if hits:
            return max(hits, key=lambda c: total.get(c, 0.0)), "nombre"
        return None, ""

    code, metodo = _pick_by_name()
    if code is None:
        if corr.empty:
            return {"codigo": None, "metodo": "ninguno", "confianza": "ninguna",
                    "corr": None, "segundo_corr": None}
        top = corr.iloc[0]
        if top < MEDIA_CORR:  # nada se parece lo bastante: queda sin mapear
            return {"codigo": None, "metodo": "ninguno", "confianza": "ninguna",
                    "corr": round(float(top), 4), "segundo_corr": None}
        tied = corr[(top - corr).abs() < 1e-9].index.tolist()
        code = max(tied, key=lambda c: total.get(c, 0.0))
        metodo = "correlacion"
    c_val = float(corr.get(code, np.nan))
    otros = corr.drop(labels=[code], errors="ignore")
    segundo = float(otros.iloc[0]) if len(otros) else None
    if np.isnan(c_val):
        return {"codigo": code, "metodo": metodo, "confianza": "baja",
                "corr": None, "segundo_corr": segundo}
    return {"codigo": code, "metodo": metodo, "confianza": _label(c_val, segundo, metodo == "correlacion"),
            "corr": round(c_val, 4),
            "segundo_corr": None if segundo is None else round(segundo, 4)}


def build_map(
    volumes: dict[str, pd.Series],
    wide: pd.DataFrame,
    aliases: dict[str, list[str]] | None = None,
    min_days: int = MIN_DIAS,
) -> dict[str, dict]:
    """Mapa `{codigo: {ticker, metodo, confianza, corr, segundo_corr}}`.

    Si dos tickers apuntan al mismo código, se queda el de mayor correlación (el otro queda sin
    mapear). Procesa primero nombre/alias para que no los pisen las correlaciones.
    """
    resolved = {t: resolve_ticker(t, v, wide, aliases, min_days) for t, v in volumes.items()}
    out: dict[str, dict] = {}
    for t, r in resolved.items():
        c = r["codigo"]
        if c is None:
            continue
        score = r["corr"] if r["corr"] is not None else -1.0
        if c in out and out[c]["corr"] is not None and out[c]["corr"] >= score:
            continue
        out[c] = {"ticker": t, **{k: v for k, v in r.items() if k != "codigo"}}
    return out


def missing_days(hechos_dates: pd.DatetimeIndex, market_volume: pd.Series) -> dict[str, list[str]]:
    """Días sin archivo de `hechos` dentro de [primer día de hechos, último día de mercado].

    - `captura`: el mercado operó (volumen > 0) pero `hechos` no tiene el día → hueco de captura.
    - `sin_mercado`: el mercado no operó (feriado o volumen 0) → no es hueco.
    No se rellenan; solo se listan.
    """
    have = pd.DatetimeIndex(pd.to_datetime(hechos_dates)).normalize()
    vol = pd.to_numeric(market_volume, errors="coerce")
    vol.index = pd.DatetimeIndex(vol.index).normalize()
    lo = have.min()
    days = pd.bdate_range(lo, max(vol.index.max(), have.max()))
    falta = days.difference(have)
    captura, sin_mercado = [], []
    for d in falta:
        v = vol.get(d, 0.0)
        (captura if (pd.notna(v) and v > 0) else sin_mercado).append(d.strftime("%Y-%m-%d"))
    return {"captura": captura, "sin_mercado": sin_mercado}
