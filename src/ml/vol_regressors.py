"""
Regresores de volatilidad solo-OHLCV y arnés de skill incremental sobre el HAR.

Idea: la volatilidad pasada ya explica casi todo lo explicable (HAR: skill
+0.30 a 5d y +0.19 a 21d contra el ingenuo). Aquí se añade UN regresor a la
vez al HAR, con los mismos folds purgados, y se mide cuánto skill suma.

Criterio de decisión (fijado ANTES de mirar resultados):
  - Un regresor se conserva solo si suma ≥ `MIN_DELTA_SKILL` (1 punto de skill,
    promedio de folds) a ese horizonte.
  - Se marca "robusto" si además el IC95% por bloques de fechas excluye 0.
  - Objetivo fijo: Parkinson (`targets.log_rv_fwd_*`); si cambia el estimador
    del objetivo, el skill deja de ser comparable.

Todos los regresores son causales: el valor en t usa solo datos hasta el
cierre de t. Convención de ventanas: incluyen t.

Registro: `REGRESSORS` (nombre → `Regressor`). Otro módulo (p. ej. calendario
y macro, M7b) añade los suyos con `register(...)` sin tocar este archivo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from src.ml.purged_cv import purged_walk_forward_date_splits
from src.ml.targets import HAR_COLS
from src.ml.vol_models import HARModel, naive_forecast, qlike

# Un epsilon sobre varianzas DIARIAS (≈ un movimiento de 0.3%): evita log(0)
# sin crear colas absurdas en ventanas cortas sin bajas/gaps.
EPS_VAR = 1e-5
MIN_DELTA_SKILL = 0.01
MIN_EMBARGO = 10


# ─── Registro ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Regressor:
    """
    name: nombre del ensayo. cols: columnas que produce.
    kind: "ticker" → fn(df OHLCV) → DataFrame por ticker;
          "universe" → fn(dict ticker→df) → DataFrame por fecha (igual para todos).
    """
    name: str
    cols: tuple[str, ...]
    kind: str
    fn: Callable


REGRESSORS: dict[str, Regressor] = {}


def register(reg: Regressor) -> Regressor:
    """Añade un regresor al registro (error si el nombre ya existe)."""
    if reg.kind not in ("ticker", "universe"):
        raise ValueError("kind debe ser 'ticker' o 'universe'")
    if reg.name in REGRESSORS:
        raise ValueError(f"regresor duplicado: {reg.name}")
    REGRESSORS[reg.name] = reg
    return reg


def compute_regressor(name: str, raws: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Calcula un regresor para todos los tickers → dict ticker → DataFrame (índice del ticker)."""
    reg = REGRESSORS[name]
    if reg.kind == "ticker":
        return {t: _finite(reg.fn(df).reindex(df.index)[list(reg.cols)])
                for t, df in raws.items()}
    uni = reg.fn(raws)
    return {t: _finite(uni.reindex(df.index)[list(reg.cols)]) for t, df in raws.items()}


def _finite(frame: pd.DataFrame) -> pd.DataFrame:
    """±inf (precios <= 0 en datos dudosos) → NaN: el arnés descarta esas filas."""
    return frame.replace([np.inf, -np.inf], np.nan)


# ─── Auxiliares ──────────────────────────────────────────────────────────────

def _log_mean(var: pd.Series, window: int) -> pd.Series:
    """log(media móvil de varianzas diarias + EPS). NaN si falta algún día de la ventana."""
    return np.log(var.rolling(window).mean().clip(lower=0) + EPS_VAR)


def _valid_ohlc(df: pd.DataFrame) -> pd.Series:
    """O y C dentro de [Low, High] y precios positivos (si no, el dato es dudoso)."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    return (l > 0) & (h >= l) & (o >= l) & (o <= h) & (c >= l) & (c <= h)


def _log_ret(df: pd.DataFrame) -> pd.Series:
    return np.log(df["Close"]).diff()


def _gap(df: pd.DataFrame) -> pd.Series:
    """Retorno nocturno log(Open_t / Close_{t-1})."""
    return np.log(df["Open"] / df["Close"].shift(1))


# ─── Regresores por ticker ───────────────────────────────────────────────────

def overnight(df: pd.DataFrame) -> pd.DataFrame:
    """Variación nocturna: log(gap²) de 1 día y de promedio 5d."""
    g2 = _gap(df) ** 2
    return pd.DataFrame({
        "log_gap2_1d": np.log(g2 + EPS_VAR),
        "log_gap2_5d": _log_mean(g2, 5),
    }, index=df.index)


def garman_klass_var(df: pd.DataFrame) -> pd.Series:
    """Varianza diaria Garman-Klass: ½·ln(H/L)² − (2ln2−1)·ln(C/O)²."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    v = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return v.where(_valid_ohlc(df))


def rogers_satchell_var(df: pd.DataFrame) -> pd.Series:
    """Varianza diaria Rogers-Satchell: ln(H/C)ln(H/O) + ln(L/C)ln(L/O)."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    v = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return v.where(_valid_ohlc(df))


def garman_klass(df: pd.DataFrame) -> pd.DataFrame:
    v = garman_klass_var(df)
    return pd.DataFrame({"log_gk_5d": _log_mean(v, 5), "log_gk_22d": _log_mean(v, 22)},
                        index=df.index)


def rogers_satchell(df: pd.DataFrame) -> pd.DataFrame:
    v = rogers_satchell_var(df)
    return pd.DataFrame({"log_rs_5d": _log_mean(v, 5), "log_rs_22d": _log_mean(v, 22)},
                        index=df.index)


def semivar(df: pd.DataFrame) -> pd.DataFrame:
    """Semivarianzas r²·1[r<0] (bajas) y r²·1[r>0] (alzas), en log."""
    r = _log_ret(df)
    dn = (r ** 2).where(r < 0, 0.0).where(r.notna())
    up = (r ** 2).where(r > 0, 0.0).where(r.notna())
    return pd.DataFrame({
        "log_semi_dn_5d": _log_mean(dn, 5),
        "log_semi_dn_22d": _log_mean(dn, 22),
        "log_semi_up_22d": _log_mean(up, 22),
    }, index=df.index)


def attention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Banderas de atención (continuas):
      vol_z60      z-score del log-volumen vs sus 60 días previos (incluye t).
      gap_sigma    log1p(|gap_t| / σ), σ = desv. de r en t-22..t-1.
      range_atr    (High−Low)_t / ATR14 de t-14..t-1 (recortado a [0, 10]).
    La escala usa datos hasta t-1, así que un shock en t se ve completo.
    """
    lv = np.log1p(df["Volume"].clip(lower=0))
    mu, sd = lv.rolling(60).mean(), lv.rolling(60).std()
    vol_z = ((lv - mu) / sd).where(sd > 0)

    r = _log_ret(df)
    sigma = r.rolling(22).std().shift(1)
    gap_sigma = np.log1p(_gap(df).abs() / sigma.where(sigma > 0))

    prev_c = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_c).abs(),
                    (df["Low"] - prev_c).abs()], axis=1).max(axis=1, skipna=False)
    atr = tr.rolling(14).mean().shift(1)
    range_atr = ((df["High"] - df["Low"]) / atr.where(atr > 0)).clip(0, 10)

    return pd.DataFrame({"vol_z60": vol_z, "gap_sigma": gap_sigma,
                         "range_atr": range_atr}, index=df.index)


# ─── Regresores de universo (cross-seccionales) ──────────────────────────────

def _returns_matrix(raws: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Retornos log diarios fecha × ticker sobre el calendario unión del universo."""
    return pd.DataFrame({t: _log_ret(df) for t, df in raws.items()}).sort_index()


def _rolling_z(x: pd.Series, window: int = 250, min_periods: int = 60) -> pd.Series:
    """z-score móvil causal (media y desviación de los `window` días hasta t)."""
    mu = x.rolling(window, min_periods=min_periods).mean()
    sd = x.rolling(window, min_periods=min_periods).std()
    return (x - mu) / sd.where(sd > 0)


def absorption_raw(raws: dict[str, pd.DataFrame], window: int = 60,
                   min_names: int = 15) -> pd.Series:
    """
    Razón de absorción: fracción de la varianza explicada por los k = N/5
    primeros autovalores de la correlación de los últimos `window` días.
    Usa solo los tickers con la ventana completa; NaN si hay < `min_names`.
    """
    R = _returns_matrix(raws)
    vals = R.to_numpy()
    out = np.full(len(R), np.nan)
    for i in range(window - 1, len(R)):
        w = vals[i - window + 1: i + 1]
        ok = ~np.isnan(w).any(axis=0)
        n = int(ok.sum())
        if n < min_names:
            continue
        sub = w[:, ok]
        sd = sub.std(axis=0)
        sub, sd = sub[:, sd > 0], sd[sd > 0]
        n = sub.shape[1]
        if n < min_names:
            continue
        ev = np.linalg.eigvalsh(np.corrcoef(sub, rowvar=False))
        k = max(1, n // 5)
        out[i] = ev[-k:].sum() / ev.sum()
    return pd.Series(out, index=R.index)


def dispersion_raw(raws: dict[str, pd.DataFrame], window: int = 60,
                   min_names: int = 10) -> pd.Series:
    """log de la dispersión cross-seccional diaria de retornos, promediada `window` días."""
    R = _returns_matrix(raws)
    disp = R.std(axis=1).where(R.notna().sum(axis=1) >= min_names)
    return np.log(disp.rolling(window).mean() + 1e-4)


def absorption(raws: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({"absorption_z": _rolling_z(absorption_raw(raws))})


def dispersion(raws: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({"dispersion_z": _rolling_z(dispersion_raw(raws))})


register(Regressor("overnight", ("log_gap2_1d", "log_gap2_5d"), "ticker", overnight))
register(Regressor("garman_klass", ("log_gk_5d", "log_gk_22d"), "ticker", garman_klass))
register(Regressor("rogers_satchell", ("log_rs_5d", "log_rs_22d"), "ticker", rogers_satchell))
register(Regressor("semivar", ("log_semi_dn_5d", "log_semi_dn_22d", "log_semi_up_22d"),
                   "ticker", semivar))
register(Regressor("attention", ("vol_z60", "gap_sigma", "range_atr"), "ticker", attention))
register(Regressor("absorption", ("absorption_z",), "universe", absorption))
register(Regressor("dispersion", ("dispersion_z",), "universe", dispersion))


# ─── Arnés ───────────────────────────────────────────────────────────────────

def verdict(delta_skill: float, ci_lo: float) -> str:
    """Criterio fijado de antemano: conserva si suma ≥ 1 punto de skill."""
    if delta_skill >= MIN_DELTA_SKILL:
        return "conserva (robusto)" if ci_lo > 0 else "conserva (no robusto)"
    return "descarta"


def _block_ids(dates: np.ndarray, block: int) -> np.ndarray:
    """Bloque (de `block` fechas únicas consecutivas) al que pertenece cada fila."""
    uniq, inv = np.unique(dates, return_inverse=True)
    return inv // block


def evaluate_regressor(panel: pd.DataFrame, target: str, h: int, reg_cols: list[str],
                       *, n_splits: int = 6, n_boot: int = 2000, seed: int = 0,
                       block: int | None = None) -> dict:
    """
    HAR + `reg_cols` contra HAR base, mismos folds purgados y misma muestra.

    `panel`: columnas date, HAR_COLS, `target`, `reg_cols`. Con `reg_cols=[]`
    evalúa solo el HAR (muestra completa). La muestra común excluye filas con
    NaN en los regresores, y el HAR base se evalúa sobre esas mismas filas.

    Devuelve skill (1 − MSE/MSE ingenuo) de ambos, Δskill promedio de folds,
    Δskill agrupado con IC95% por bootstrap de bloques de fechas (bloques de
    2h fechas, ≈ independientes), prueba Diebold-Mariano por bloques y ΔQLIKE
    (positivo = el regresor mejora).
    """
    block = block or 2 * h
    data = panel.dropna(subset=[target] + list(reg_cols)).reset_index(drop=True)
    cols_x = HAR_COLS + list(reg_cols)
    y = data[target].to_numpy()
    splits = purged_walk_forward_date_splits(
        data["date"], n_splits=n_splits, horizon=h, embargo=max(MIN_EMBARGO, h))

    sk_b, sk_x, ql_b, ql_x = [], [], [], []
    d_dates, e_b2, e_x2, e_n2 = [], [], [], []
    for tr, te in splits:
        X_tr, X_te = data.iloc[tr], data.iloc[te]
        base = HARModel().fit(X_tr, y[tr])
        p_b = base.predict(X_te)
        if reg_cols:
            lr = LinearRegression().fit(X_tr[cols_x].to_numpy(), y[tr])
            p_x = lr.predict(X_te[cols_x].to_numpy())
        else:
            p_x = p_b
        p_n = naive_forecast(X_te, h)
        yt = y[te]
        mse_n = np.mean((yt - p_n) ** 2)
        sk_b.append(1 - np.mean((yt - p_b) ** 2) / mse_n)
        sk_x.append(1 - np.mean((yt - p_x) ** 2) / mse_n)
        ql_b.append(qlike(yt, p_b))
        ql_x.append(qlike(yt, p_x))
        d_dates.append(X_te["date"].to_numpy())
        e_b2.append((yt - p_b) ** 2)
        e_x2.append((yt - p_x) ** 2)
        e_n2.append((yt - p_n) ** 2)

    dates = np.concatenate(d_dates)
    eb, ex, en = (np.concatenate(a) for a in (e_b2, e_x2, e_n2))
    mse_n_pool = en.mean()
    bid = _block_ids(dates, block)
    nb = bid.max() + 1
    S = np.bincount(bid, weights=eb - ex, minlength=nb)
    N = np.bincount(bid, minlength=nb).astype(float)
    delta_pool = S.sum() / N.sum() / mse_n_pool

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(n_boot, nb))
    boot = S[idx].sum(axis=1) / N[idx].sum(axis=1) / mse_n_pool
    lo, hi = np.percentile(boot, [2.5, 97.5])

    m = S / N                                   # diferencia media de pérdida por bloque
    sd = m.std(ddof=1) if nb > 1 else np.nan
    t = m.mean() / (sd / np.sqrt(nb)) if sd and sd > 0 else 0.0
    p = float(2 * stats.t.sf(abs(t), df=max(nb - 1, 1))) if sd and sd > 0 else 1.0

    return {
        "target": target, "horizon": h, "regressors": list(reg_cols),
        "n_obs": int(len(data)), "n_blocks": int(nb), "block_dates": int(block),
        "skill_base": float(np.mean(sk_b)), "skill_reg": float(np.mean(sk_x)),
        "delta_skill_fold_mean": float(np.mean(np.array(sk_x) - np.array(sk_b))),
        "delta_skill_fold_std": float(np.std(np.array(sk_x) - np.array(sk_b))),
        "delta_skill_pooled": float(delta_pool),
        "delta_ci_lo": float(lo), "delta_ci_hi": float(hi),
        "dm_t": float(t), "dm_p": p,
        "qlike_base": float(np.mean(ql_b)), "qlike_reg": float(np.mean(ql_x)),
        "delta_qlike": float(np.mean(ql_b) - np.mean(ql_x)),
        "fold_deltas": [float(a - b) for a, b in zip(sk_x, sk_b)],
    }
