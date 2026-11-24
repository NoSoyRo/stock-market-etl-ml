"""
Vol targeting (M6): apostar menos cuando el mercado está nervioso.

Idea: si la volatilidad se puede predecir, escalar la exposición por
1/σ̂ deja el riesgo más parejo y baja el drawdown. Reglas fijas del módulo:

  - Solo largos y SIN apalancamiento: la exposición está acotada por `cap`
    (≤ 1.0). El resto queda en efectivo (retorno 0).
  - El peso del día t usa σ̂ calculada hasta t - lag (lag=1 → cierre de ayer).
  - El costo se cobra por cambio de peso: cost · |w_t - w_{t-1}|; la
    entrada inicial (0 → w) también se cobra.

σ̂ siempre es volatilidad anualizada (fracción, p. ej. 0.15 = 15%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.purged_cv import purged_walk_forward_date_splits
from src.ml.targets import HAR_COLS
from src.ml.vol_models import HARModel

TRADING_DAYS = 252
_EPS = 1e-4


def ewma_vol(returns: pd.Series, span: int) -> pd.Series:
    """
    Vol anualizada por EWMA de r² (media cero), con alpha = 2/(span+1).

    Causal: el valor en t usa retornos hasta t. NaN hasta tener `span` datos.
    """
    var = (returns ** 2).ewm(span=span, min_periods=span, adjust=False).mean()
    return np.sqrt(var * TRADING_DAYS)


def target_exposure(
    sigma_hat: pd.Series, target_vol: float, cap: float = 1.0
) -> pd.Series:
    """Exposición = min(cap, objetivo / σ̂). NaN se mantiene NaN. Exige cap ≤ 1."""
    if cap > 1.0 or cap <= 0:
        raise ValueError("cap debe estar en (0, 1]: no se permite apalancamiento")
    w = target_vol / sigma_hat.where(sigma_hat > 0)
    return w.clip(upper=cap)


def vol_targeted_returns(
    returns: pd.Series,
    sigma_hat: pd.Series,
    target_vol: float,
    lag: int = 1,
    cost_per_unit_change: float = 0.0005,
    cap: float = 1.0,
) -> pd.Series:
    """
    Retornos escalados por el peso del día, descontando el costo por cambio.

    Args:
        returns: retornos simples por periodo (fracciones).
        sigma_hat: vol anualizada estimada al cierre de cada fecha.
        lag: periodos de rezago entre σ̂ y su uso (≥ 1; 1 = peso de mañana
            con σ̂ de hoy).
        cost_per_unit_change: costo (fracción) por unidad de peso que cambia.

    Returns:
        Serie neta, solo en fechas con peso definido.
    """
    if lag < 1:
        raise ValueError("lag debe ser ≥ 1 para no mirar al futuro")
    sig = sigma_hat.reindex(returns.index).shift(lag)
    w = target_exposure(sig, target_vol, cap)
    valid = w.notna() & returns.notna()
    w, r = w[valid], returns[valid]
    dw = w.diff()
    dw.iloc[:1] = w.iloc[:1]  # entrada desde efectivo (peso previo 0)
    return w * r - cost_per_unit_change * dw.abs()


def har_oos_vol(
    returns: pd.Series,
    *,
    horizon: int = 5,
    n_splits: int = 8,
    embargo: int = 10,
    min_train_size: float = 0.40,
) -> pd.Series:
    """
    σ̂ anualizada de HAR predicha fuera de muestra con folds purgados.

    Sobre los retornos de UNA serie (p. ej. el equiponderado). Features HAR
    causales (log vol 1d/5d/22d con r²); etiqueta = log vol de t+1..t+horizon.
    Cada bloque de test se predice con un modelo entrenado solo con fechas
    anteriores (purga `horizon` + `embargo`). Las fechas anteriores al primer
    bloque no tienen predicción (NaN).
    """
    r2 = returns ** 2

    def _vol(window: int) -> pd.Series:
        m = r2.rolling(window).mean()
        if window > 1:
            m = m.where(m > 0)
        return np.sqrt(m * TRADING_DAYS)

    feats = pd.DataFrame({
        "log_rv_1d": np.log(_vol(1) + _EPS),
        "log_rv_5d": np.log(_vol(5) + _EPS),
        "log_rv_22d": np.log(_vol(22) + _EPS),
    })[HAR_COLS]
    fwd = r2.rolling(horizon).mean().shift(-horizon)
    y = np.log(np.sqrt(fwd.where(fwd > 0) * TRADING_DAYS) + _EPS)

    ok = feats.notna().all(axis=1) & y.notna()
    pred = pd.Series(np.nan, index=returns.index)
    dates = returns.index
    for train_idx, test_idx in purged_walk_forward_date_splits(
        dates, n_splits=n_splits, horizon=horizon,
        embargo=embargo, min_train_size=min_train_size,
    ):
        tr = train_idx[ok.to_numpy()[train_idx]]
        if len(tr) < 100:
            continue
        model = HARModel().fit(feats.iloc[tr], y.to_numpy()[tr])
        te = test_idx[feats.notna().all(axis=1).to_numpy()[test_idx]]
        if len(te):
            # exp(log vol) - EPS: vuelve a vol anualizada (fracción).
            pred.iloc[te] = np.exp(model.predict(feats.iloc[te])) - _EPS
    return pred.where(pred > 0)
