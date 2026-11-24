"""
Risk Engine — Sizing + Stops + Correlación de portafolio.

Funciones principales:
  kelly_fraction  — fracción óptima de capital por trade (Kelly criterio)
  position_size   — unidades a comprar dado capital y precio
  atr_stop        — nivel de stop-loss basado en ATR
  r_multiple      — distancia al target en términos de R (riesgo)
  correlation_penalty — factor de reducción por correlación en el portafolio
  build_signal    — objeto Signal completo para un ticker dado el modelo
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ─── Parámetros del backtest histórico (19 tickers, 2.5 años) ────────────────
# Estos stats vienen del análisis walk-forward: señales con P > 0.55
HIST_WIN_RATE  = 0.522    # win rate empírico de señales con P > 0.55
HIST_AVG_WIN   = 3.32     # retorno promedio en trades ganadores (%)
HIST_AVG_LOSS  = 2.44     # pérdida promedio en trades perdedores (%) — positivo
HIST_HORIZON   = 5        # días de holding


# ─── Dataclass de señal ───────────────────────────────────────────────────────

@dataclass
class Signal:
    ticker:       str
    price:        float          # último precio de cierre
    p_up:         float          # P(subida 5d) del modelo logístico
    atr:          float          # ATR(14) en pesos

    # Niveles
    entry:        float = 0.0
    stop:         float = 0.0
    target_1:     float = 0.0    # 1R
    target_2:     float = 0.0    # 2R

    # Sizing
    kelly_f:      float = 0.0    # fracción Kelly (antes de reducir)
    kelly_frac:   float = 0.25   # Kelly fraccionario (25%)
    pct_capital:  float = 0.0    # % del capital total a asignar
    shares:       int   = 0      # número de acciones
    capital_mxn:  float = 0.0    # pesos invertidos

    # Risk/reward
    risk_pct:     float = 0.0    # % del capital en riesgo (stop loss)
    rr_ratio:     float = 0.0    # risk-reward ratio T1
    edge:         float = 0.0    # EV del trade (%)

    # Meta
    features:     dict = field(default_factory=dict)
    reason:       str  = ""


# ─── Kelly Criterion ──────────────────────────────────────────────────────────

def kelly_fraction(
    p: float,
    avg_win_pct: float  = HIST_AVG_WIN,
    avg_loss_pct: float = HIST_AVG_LOSS,
    kelly_frac: float   = 0.25,
    max_f: float        = 0.10,
) -> float:
    """
    Fracción de Kelly (criterio clásico discreto) para un trade.

    f* = (p × b - q) / b   donde b = avg_win / avg_loss

    Parámetros
    ----------
    p             : probabilidad de éxito predicha por el modelo
    avg_win_pct   : retorno promedio en trades ganadores (%)
    avg_loss_pct  : pérdida promedio en trades perdedores (%)
    kelly_frac    : multiplicador de seguridad (0.25 = Kelly cuarto)
    max_f         : tope máximo de la fracción final

    Returns
    -------
    Fracción de capital a asignar (0.0–max_f)
    """
    q = 1.0 - p
    b = avg_win_pct / (avg_loss_pct + 1e-8)  # odds de ganancia/pérdida
    f_full = (p * b - q) / (b + 1e-8)
    f_full = max(f_full, 0.0)                 # sin posiciones cortas por ahora
    return min(f_full * kelly_frac, max_f)


def position_size(
    capital_mxn:  float,
    pct_capital:  float,
    price:        float,
    lote_min:     int = 1,
) -> tuple[int, float]:
    """
    Devuelve (acciones, capital_invertido).

    Ajusta al múltiplo de lote_min hacia abajo.
    """
    budget = capital_mxn * pct_capital
    shares_raw = budget / (price + 1e-8)
    shares = max(lote_min, int(shares_raw // lote_min) * lote_min)
    return shares, round(shares * price, 2)


# ─── ATR Stop & Targets ───────────────────────────────────────────────────────

def atr_stop(
    entry:      float,
    atr:        float,
    mult_stop:  float = 1.5,
    mult_t1:    float = 1.5,
    mult_t2:    float = 3.0,
) -> tuple[float, float, float]:
    """
    Calcula stop, target_1 y target_2 basados en ATR.

    Stop   = entry - mult_stop × ATR
    T1     = entry + mult_t1  × ATR  → cierre parcial 50%
    T2     = entry + mult_t2  × ATR  → cierre total

    Returns
    -------
    (stop, target_1, target_2)  redondeados a 2 decimales
    """
    stop     = round(entry - mult_stop * atr, 2)
    target_1 = round(entry + mult_t1   * atr, 2)
    target_2 = round(entry + mult_t2   * atr, 2)
    return stop, target_1, target_2


def r_multiple(entry: float, stop: float, target: float) -> float:
    """R-múltiplo: cuántos R vale el target."""
    risk = entry - stop
    if risk <= 0:
        return 0.0
    return round((target - entry) / risk, 2)


def expected_value(p: float, avg_win: float, avg_loss: float) -> float:
    """EV = p × avg_win − (1-p) × avg_loss (en %)."""
    return round(p * avg_win - (1 - p) * avg_loss, 3)


# ─── Correlación de portafolio ────────────────────────────────────────────────

def compute_correlation_matrix(
    returns_dict: dict[str, pd.Series],
    window: int = 63,
) -> pd.DataFrame:
    """
    Matriz de correlación de retornos para los tickers del portafolio.
    Usa las últimas `window` observaciones para cada par.

    Parameters
    ----------
    returns_dict : {ticker: pd.Series de retornos diarios}
    window       : número de días a considerar

    Returns
    -------
    pd.DataFrame con la matriz de correlación
    """
    df = pd.DataFrame(returns_dict)
    df = df.tail(window)
    return df.corr()


def correlation_penalty(
    ticker: str,
    corr_matrix: pd.DataFrame,
    portfolio_tickers: list[str],
    high_corr_threshold: float = 0.70,
) -> float:
    """
    Factor de reducción de tamaño por correlación con posiciones ya abiertas.

    Reglas:
      - Si ningún ticker en el portafolio tiene corr > threshold → sin penalidad (1.0)
      - Si hay 1 ticker con corr > 0.70 → reducir al 75%
      - Si hay 2+ tickers con corr > 0.70 → reducir al 50%

    Returns
    -------
    Factor multiplicador (0.5–1.0)
    """
    if ticker not in corr_matrix.index or not portfolio_tickers:
        return 1.0

    n_high = sum(
        1 for t in portfolio_tickers
        if t in corr_matrix.columns and corr_matrix.loc[ticker, t] > high_corr_threshold
    )

    if n_high == 0:   return 1.0
    if n_high == 1:   return 0.75
    return 0.50


# ─── Builder de señal completa ────────────────────────────────────────────────

def build_signal(
    ticker:       str,
    price:        float,
    atr:          float,
    p_up:         float,
    capital_mxn:  float = 100_000,
    corr_factor:  float = 1.0,
    features:     dict  | None = None,
    kelly_frac:   float = 0.25,
    max_position: float = 0.10,
    sigma_daily:  float | None = None,
    payoffs:      tuple[float, float] | None = None,
) -> Signal | None:
    """
    Construye una señal de trading completa con sizing y niveles.

    Returns None si p_up < 0.50 o el EV del trade es negativo.

    M8 (opcionales; con None conservan el comportamiento histórico):
      sigma_daily : σ̂ diaria (fracción) → stop/targets = entry·(1 ∓ k·σ̂·√5) en vez de ATR.
      payoffs     : `stops.Payoffs` (o tupla win, loss en %) estimados del train → Kelly
                    y EV con esos pagos en vez de HIST_AVG_WIN/LOSS.
    """
    if p_up < 0.50:
        return None

    avg_win, avg_loss = (HIST_AVG_WIN, HIST_AVG_LOSS) if payoffs is None else tuple(payoffs)
    ev = expected_value(p_up, avg_win, avg_loss)
    if ev <= 0:
        return None

    # Sizing
    if payoffs is None:
        f_kelly = kelly_fraction(p_up, kelly_frac=kelly_frac, max_f=max_position)
    else:
        from src.ml.stops import Payoffs, kelly_fraction_estimated
        f_kelly = kelly_fraction_estimated(
            p_up, Payoffs(*payoffs), kelly_frac=kelly_frac, max_f=max_position)
    f_adj    = f_kelly * corr_factor
    shares, capital_invested = position_size(capital_mxn, f_adj, price)

    # Niveles
    entry             = round(price * 1.0025, 2)  # +0.25% de slippage estimado
    if sigma_daily is None:
        stop, t1, t2  = atr_stop(entry, atr, mult_stop=1.5, mult_t1=1.5, mult_t2=3.0)
    else:
        from src.ml.stops import sigma_stop_levels
        stop, t1 = sigma_stop_levels(entry, sigma_daily, HIST_HORIZON, 1.0, 1.0)
        _, t2    = sigma_stop_levels(entry, sigma_daily, HIST_HORIZON, 1.0, 2.0)
        stop, t1, t2 = round(stop, 2), round(t1, 2), round(t2, 2)
    rr                = r_multiple(entry, stop, t1)
    risk_pct_capital  = round((entry - stop) / entry * f_adj * 100, 3)

    # Descripción
    reasons = []
    if features:
        if features.get("williams_r", 0) < -80:
            reasons.append("Williams %R oversold")
        if features.get("stoch_k", 100) < 20:
            reasons.append("Stoch oversold")
        if features.get("ret_10d", 0) < -4:
            reasons.append(f"10d drop {features['ret_10d']:.1f}%")
        if features.get("bb_pct", 100) < 15:
            reasons.append("cerca banda Bollinger baja")
        if features.get("price_vs_sma20", 0) < -3:
            reasons.append(f"bajo SMA20 {features['price_vs_sma20']:.1f}%")

    return Signal(
        ticker       = ticker,
        price        = price,
        p_up         = round(p_up, 4),
        atr          = round(atr, 4),
        entry        = entry,
        stop         = stop,
        target_1     = t1,
        target_2     = t2,
        kelly_f      = round(f_kelly, 4),
        kelly_frac   = kelly_frac,
        pct_capital  = round(f_adj * 100, 2),
        shares       = shares,
        capital_mxn  = capital_invested,
        risk_pct     = risk_pct_capital,
        rr_ratio     = rr,
        edge         = ev,
        features     = features or {},
        reason       = " · ".join(reasons),
    )


# ─── Resumen de portafolio ────────────────────────────────────────────────────

def portfolio_summary(signals: list[Signal], capital: float) -> dict:
    """Estadísticas de riesgo del portafolio completo."""
    total_invested = sum(s.capital_mxn for s in signals)
    total_risk     = sum(
        (s.entry - s.stop) / s.entry * s.capital_mxn
        for s in signals
    )
    return {
        "n_positions":    len(signals),
        "total_invested": round(total_invested, 2),
        "total_risk_mxn": round(total_risk, 2),
        "pct_invested":   round(total_invested / capital * 100, 1),
        "pct_at_risk":    round(total_risk / capital * 100, 2),
        "avg_p_up":       round(sum(s.p_up for s in signals) / max(len(signals),1), 4),
        "avg_edge":       round(sum(s.edge for s in signals) / max(len(signals),1), 3),
    }
