"""M10 · Variantes del objetivo de entrenamiento cross-seccional (cambiar la pregunta).

Todas son causales y por fecha. ``fwd_ret`` está en POR CIENTO (convención de ``build_panel``).
La σ̂ es la EWMA de 20 días de los retornos diarios del ticker, calculada con datos hasta t
(cierre de t), en escala DIARIA y en por ciento para ser comparable con ``fwd_ret``.

Variantes (el panel debe tener índice de fechas repetidas por ticker):
  A  percentil por fecha de fwd_ret                       (baseline)
  B  percentil por fecha de fwd_ret / σ̂                   (retorno en unidades de riesgo)
  C  percentil por fecha de (fwd_ret − media de la fecha) (sin escalar)
  D  como A, pero entrena solo con |fwd_ret| ≥ k·σ̂·√h    (zona neutra; el test NO se recorta)
  E  como A, sin las features de mercado (MARKET_COLS)        (solo selección de columnas)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.vol_target import TRADING_DAYS, ewma_vol

SPAN = 20
NEUTRAL_K = 0.5
HORIZON = 5
VARIANTS = ("A", "B", "C", "D", "E")
# Features de mercado (sección F de features.py, iguales para todos los tickers de una fecha).
# Lista fija a priori: el detector por datos es frágil ante feriados distintos entre tickers.
# De las 7 de la sección F solo estas 3 están en las 24 features del Ridge (rel_str5 sí varía
# por ticker; mkt_ret10, mkt_trend y usdmxn_vs_ma20 no entran a las 24).
MARKET_COLS = ("rn_wti_ret5", "rn_sp500_ret5", "rn_usdmxn_ret5")


def daily_sigma_pct(close: pd.Series, span: int = SPAN) -> pd.Series:
    """σ̂ diaria en % (EWMA de r², causal): el valor en t usa solo cierres hasta t."""
    return ewma_vol(close.pct_change(), span) / np.sqrt(TRADING_DAYS) * 100.0


def sigma_for_panel(panel: pd.DataFrame, sigma_by_ticker: dict[str, pd.Series]) -> np.ndarray:
    """σ̂ alineada fila a fila con el panel (por fecha y ticker); NaN si no existe."""
    out = np.full(len(panel), np.nan)
    tick = panel["ticker"].to_numpy()
    for t, s in sigma_by_ticker.items():
        m = tick == t
        if m.any():
            out[m] = s.reindex(panel.index[m]).to_numpy()
    return out


def pct_by_date(values: np.ndarray | pd.Series, dates: pd.Index) -> np.ndarray:
    """Percentil (0–1] de cada valor dentro de su fecha; NaN se queda NaN."""
    s = pd.Series(np.asarray(values, dtype=float), index=pd.Index(dates))
    return s.groupby(level=0).rank(pct=True).to_numpy()


def target_a(fwd_ret: np.ndarray, dates: pd.Index) -> np.ndarray:
    """Baseline: percentil por fecha de fwd_ret."""
    return pct_by_date(fwd_ret, dates)


def target_b(fwd_ret: np.ndarray, sigma: np.ndarray, dates: pd.Index) -> np.ndarray:
    """Percentil por fecha de fwd_ret / σ̂ (σ̂ ≤ 0 o NaN => NaN)."""
    sig = np.where(np.asarray(sigma, float) > 0, sigma, np.nan)
    return pct_by_date(np.asarray(fwd_ret, float) / sig, dates)


def target_c(fwd_ret: np.ndarray, dates: pd.Index) -> np.ndarray:
    """Percentil por fecha de fwd_ret − media de la fecha (sin escalar)."""
    s = pd.Series(np.asarray(fwd_ret, dtype=float), index=pd.Index(dates))
    centered = s - s.groupby(level=0).transform("mean")
    return pct_by_date(centered.to_numpy(), dates)


def neutral_zone_mask(fwd_ret: np.ndarray, sigma: np.ndarray, *,
                      k: float = NEUTRAL_K, horizon: int = HORIZON) -> np.ndarray:
    """True en las filas "claras": |fwd_ret| ≥ k·σ̂·√horizon. σ̂ NaN => False."""
    thr = k * np.asarray(sigma, float) * np.sqrt(horizon)
    with np.errstate(invalid="ignore"):
        return np.abs(np.asarray(fwd_ret, float)) >= thr


def constant_by_date_cols(panel: pd.DataFrame, cols: list[str], *, min_frac: float = 0.9,
                          tol: float = 1e-12) -> list[str]:
    """Columnas que no varían dentro de la fecha en ≥ ``min_frac`` de las fechas.

    Las features de mercado no son estrictamente constantes: en fechas con feriados distintos
    entre tickers el valor rellenado puede diferir, así que se tolera una fracción pequeña.
    """
    g = panel[cols].groupby(panel.index)
    flat = ((g.max() - g.min()) <= tol).mean()
    return [c for c in cols if flat[c] >= min_frac]


def build_variant(panel: pd.DataFrame, variant: str, sigma: np.ndarray,
                  cols: list[str], *, k: float = NEUTRAL_K,
                  horizon: int = HORIZON) -> tuple[pd.DataFrame, list[str]]:
    """Panel con ``pct_fwd`` de la variante y las columnas de features a usar.

    El filtro de la variante D va en la columna booleana ``train_ok`` (el test debe ignorarla);
    en las demás todo es True. No modifica el panel de entrada.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variante desconocida: {variant}")
    out = panel.copy()
    fwd, dates = out["fwd_ret"].to_numpy(), out.index
    if variant == "B":
        out["pct_fwd"] = target_b(fwd, sigma, dates)
    elif variant == "C":
        out["pct_fwd"] = target_c(fwd, dates)
    else:
        out["pct_fwd"] = target_a(fwd, dates)
    out["train_ok"] = (neutral_zone_mask(fwd, sigma, k=k, horizon=horizon)
                       if variant == "D" else True)
    use = [c for c in cols if c not in MARKET_COLS] if variant == "E" else list(cols)
    return out, use
