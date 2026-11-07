"""
M8 · Stops, tamaño de posición y etiqueta de cola con la volatilidad predicha.

Idea: lo único que el modelo sabe predecir es la volatilidad. Entonces el riesgo
(stop, tamaño, veto de "desplome probable") se mide en unidades de σ̂ y no con un
ATR fijo ni con pagos "3.32 / 2.44" copiados de un backtest viejo.

Convenciones:
  - σ̂ aquí es **diaria** (fracción, p. ej. 0.02 = 2%), EWMA de r² (la vara de M6;
    HAR sub-estima, no se usa). σ̂ del horizonte h = σ̂_diaria · √h.
  - Retornos en fracciones salvo `estimate_payoffs`, que recibe porcentajes (como
    `fwd_ret` y `HIST_AVG_WIN/LOSS`).
  - `lag=0` significa "σ̂ hasta el cierre de la señal" (disponible al decidir);
    `lag=1` la rezaga un periodo más.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

from src.ml.vol_target import TRADING_DAYS, ewma_vol


# ─── σ̂ y niveles ─────────────────────────────────────────────────────────────

def daily_sigma(close: pd.Series, span: int = 21, lag: int = 0) -> pd.Series:
    """σ̂ diaria (EWMA de r²) causal: el valor en t usa precios hasta t - lag."""
    if lag < 0:
        raise ValueError("lag debe ser ≥ 0")
    ret = close.pct_change()
    sig = ewma_vol(ret, span) / np.sqrt(TRADING_DAYS)
    return sig.shift(lag) if lag else sig


def horizon_sigma(sigma_daily, horizon: int = 5):
    """σ̂ del horizonte: σ̂_diaria · √h."""
    return sigma_daily * np.sqrt(horizon)


def sigma_stop_levels(
    entry: float,
    sigma_daily: float,
    horizon: int = 5,
    mult_stop: float = 1.0,
    mult_target: float = 1.0,
) -> tuple[float, float]:
    """(stop, target) = entrada · (1 ∓ k·σ̂·√h). Sin redondeo."""
    sh = float(horizon_sigma(sigma_daily, horizon))
    return entry * (1.0 - mult_stop * sh), entry * (1.0 + mult_target * sh)


def inverse_vol_size(
    sigma_daily: float,
    risk_budget: float = 0.005,
    horizon: int = 5,
    stop_mult: float = 1.0,
    max_f: float = 0.10,
) -> float:
    """
    Fracción de capital inversa a σ̂: si el stop está a k·σ̂·√h, perder el stop cuesta
    ``risk_budget`` del capital. σ̂ inválida (NaN o ≤ 0) → 0 (no apostar).
    """
    if not np.isfinite(sigma_daily) or sigma_daily <= 0:
        return 0.0
    f = risk_budget / (stop_mult * float(horizon_sigma(sigma_daily, horizon)))
    return float(min(f, max_f))


# ─── Kelly con pagos estimados ───────────────────────────────────────────────

class Payoffs(NamedTuple):
    """Pagos estimados (en %): ganancia media y pérdida media (positiva)."""
    avg_win: float
    avg_loss: float


def estimate_payoffs(train_ret_pct: pd.Series, cost_pct: float = 0.0) -> Payoffs:
    """
    Pagos medios SOLO con retornos de train (%), netos de costo ida y vuelta.

    Gana (p = P(retorno bruto > 0)): ganancia bruta media − costo.
    Pierde: pérdida bruta media + costo. Sin ganadores o sin perdedores → 0 en ese lado.
    """
    r = pd.Series(train_ret_pct).dropna()
    wins, losses = r[r > 0], r[r <= 0]
    avg_win = float(wins.mean() - cost_pct) if len(wins) else 0.0
    avg_loss = float(-losses.mean() + cost_pct) if len(losses) else 0.0
    return Payoffs(avg_win=max(avg_win, 0.0), avg_loss=max(avg_loss, 0.0))


def kelly_fraction_estimated(
    p: float,
    payoffs: Payoffs,
    kelly_frac: float = 0.25,
    max_f: float = 0.10,
) -> float:
    """
    Kelly discreto con pagos estimados. f* = (p·W − (1−p)·L) / W (con W, L en %).

    Devuelve 0 si el edge estimado ``p·W − (1−p)·L`` ≤ 0, si p es NaN o si algún
    pago es ≤ 0 (no hay forma de dimensionar). Con W = HIST_AVG_WIN y L = HIST_AVG_LOSS
    es la misma fórmula de `risk_engine.kelly_fraction`.
    """
    w, l = float(payoffs.avg_win), float(payoffs.avg_loss)
    if not np.isfinite(p) or w <= 0 or l <= 0:
        return 0.0
    edge = p * w - (1.0 - p) * l
    if edge <= 0:
        return 0.0
    return float(min(edge / w * kelly_frac, max_f))


# ─── Etiqueta de cola ────────────────────────────────────────────────────────

def tail_label(fwd_ret: pd.Series, sigma_h: pd.Series, k: float = 1.0) -> pd.Series:
    """
    ``tail = 1[r_fwd < −k·σ̂_h]`` fila a fila (fracciones). NaN si falta el retorno
    futuro o σ̂ no es válida (> 0): un NaN no es "no hubo cola".
    """
    s = sigma_h.reindex(fwd_ret.index)
    ok = fwd_ret.notna() & s.notna() & (s > 0)
    out = (fwd_ret < -k * s).astype(float)
    return out.where(ok)


# ─── Evaluación ──────────────────────────────────────────────────────────────

def max_drawdown(returns: pd.Series) -> float:
    """MDD (≤ 0) de la curva compuesta de retornos por periodo."""
    eq = (1.0 + pd.Series(returns).dropna()).cumprod()
    if eq.empty:
        return float("nan")
    dd = eq / np.maximum.accumulate(np.r_[1.0, eq.to_numpy()])[1:] - 1.0
    return float(dd.min())


def paired_block_bootstrap(
    a: pd.Series,
    b: pd.Series,
    stat: Callable[[pd.Series], float],
    *,
    block: int = 4,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """
    IC 95% de ``stat(a) − stat(b)`` con bootstrap pareado de bloques móviles (mismos
    índices de remuestreo para ambas series). Devuelve diff, se, ci_lo, ci_hi, n.
    """
    df = pd.concat([pd.Series(a).reset_index(drop=True),
                    pd.Series(b).reset_index(drop=True)], axis=1).dropna()
    x, y = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    n = len(x)
    diff = float(stat(pd.Series(x)) - stat(pd.Series(y)))
    if n < 2:
        return {"diff": diff, "se": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n": n}
    bl = min(max(int(block), 1), n)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / bl))
    starts = rng.integers(0, n - bl + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(bl)).reshape(n_boot, -1)[:, :n]
    d = np.array([stat(pd.Series(x[i])) - stat(pd.Series(y[i])) for i in idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": diff, "se": float(d.std(ddof=1)), "ci_lo": float(lo),
            "ci_hi": float(hi), "n": n}
