"""
Nivel 2 / idea A: MOTOR DE RIESGO (no predice direccion). ¿El ML/estadistica mejora riesgo y construccion?

PRE-REGISTRO (fijado ANTES de correr; no se ajusta despues)
-----------------------------------------------------------
Universo: las 35 emisoras del panel; retornos diarios close-to-close (store), recortados a +-25 %.
Elegibles en cada rebalanceo: >=240 de los ultimos 252 retornos no nulos, dato ese dia y >=10 emisoras.
Rebalanceo MENSUAL (cierre del ultimo dia habil; pesos con info <= t; ejecucion al cierre t+1).
Costo 0.30 %/lado sobre el turnover real (|w_obj - w_derivado|), tambien la entrada inicial.
Solo largos, sin apalancamiento, resto en efectivo (retorno 0). Cap por nombre = max(0.10, 2/n).
Ventana de covarianza: 252 d. Sin parametros ajustados con datos de test.

Metricas (sobre retornos diarios netos de costos): vol anualizada, MDD, CVaR 5 % diario, Calmar
(retorno anualizado / |MDD|) y Sharpe neto (rf=0, x sqrt 252). Folds: los 6 de test por fechas
(purged_walk_forward_date_splits, n_splits=6, horizon=6, embargo=10; F1 = el mas antiguo, F6 = el mas
reciente). 2008 queda en TRAIN del panel: solo se reporta como estres para estrategias sin parametros
ajustados (pesos y EWMA), marcado "sin fold".

Baselines: (a) 1/N, (b) 1/N + vol targeting EWMA (span 21, objetivo 10 % anual, cap 1, exposicion
actualizada cada 5 d con rezago 1 d; usa src/ml/vol_target.ewma_vol ya existente; el costo del cambio de
exposicion es 0.30 %/unidad, no el 0.05 % del estudio M6), (c) min-var con covarianza muestral.
Auditoria de portfolio_optimizer_cov.py: es un optimizador media-varianza de UN solo corte, 10 tickers,
covarianza muestral ~2 anios, sin backtest, sin shrinkage, con rutas absolutas de otra maquina -> no se
reutiliza; aqui se construye un backtest walk-forward desde cero.

Configuraciones (todas registradas con log_trial, familias n2a_*):
  pesos: ew, invvol, mv_sample, mv_lw, mv_oas, mv_cc (constant-corr delta=0.5), hrp_sample, hrp_lw, rp_lw (ERC)
  escala (sobre ew, hrp_lw, mv_lw; base de combinacion fijada a priori): ewma, fhs5, fhs21, lgbm5, lgbm21
    * Exposicion = min(1, Q*/(-q_hat)), Q*_h = 1.645*0.10*sqrt(h/252) (equivale a 10 % de vol bajo normalidad).
    * fhs_h: q_hat = sigma_EWMA21*sqrt(h/252)* cuantil 5 % empirico EXPANDING (min 250) de los retornos
      h-dias estandarizados ya cerrados.
    * lgbm_h: LightGBM cuantil 5 % (num_leaves 4, 150 arboles, lr 0.03, min_child 80) sobre el retorno h-dias
      estandarizado, features solo de la serie del portafolio + amplitud; entrenado por fold con purga h+10 dias;
      prediccion recortada a <= -0.5.
  tilt (opcional): hrp_lw con tilt w*(1+k*z_ridge) (z recortado a +-2, k in {0.25, 0.5}); score Ridge-rank h21 de N1-B.

REGLA DE VEREDICTO (por comparacion y por metrica M en {MDD, CVaR, Sharpe}): POSITIVO solo si
  (i) mejora M al baseline relevante en >=4 de 6 folds, (ii) IC 90 % bootstrap por bloques (21 d, 1000 rep.) de la
  diferencia agrupada OOS excluye 0 a favor, (iii) mejora M en las ventanas de estres cubiertas (2020-02-15..06-30 y
  2022). En estres: MDD y CVaR dentro de la ventana; Sharpe -> retorno de la ventana. Baseline relevante: pesos vs 1/N
  y vs min-var muestral; escalas/combinaciones vs 1/N+EWMA. Diferencias de +-10 pts de MDD en un tramo = ruido.
  Test Diebold-Mariano (HAC Newey-West) sobre varianza realizada (r^2 bruto) para pesos y sobre pinball para cuantiles;
  cobertura: Kupiec y Christoffersen (CC) con observaciones no solapadas cada h dias.

Ejecucion: .venv/bin/python scripts/n2a_riesgo.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import OAS, LedoitWolf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import scripts.n1a_leadlag as N
from src.ml.purged_cv import purged_walk_forward_date_splits
from src.ml.trials import log_trial
from src.ml.vol_target import ewma_vol

SCR = N.SCRATCH
OUT = ROOT / "data" / "eval"
TRIALS = str(ROOT / "data" / "eval" / "trial_log.jsonl")
COST = 0.003
CLIP = 0.25
WIN, MIN_OBS, MIN_NAMES = 252, 240, 10
TARGET_VOL = 0.10
EXPO_EVERY = 5
BOOT_B, BOOT_BLOCK = 1000, 21
STRESS = {"2020": ("2020-02-15", "2020-06-30"), "2022": ("2022-01-01", "2022-12-31")}
STRESS_NOFIT = {"2008": ("2008-09-01", "2009-06-30")}
Z5 = 1.645


# ─── datos ────────────────────────────────────────────────────────────────────

def load_returns():
    panel = pd.read_parquet(SCR / "panel.parquet")
    pdt = pd.to_datetime(panel.index.values)
    pdates = pd.DatetimeIndex(sorted(set(pdt)))
    px = {}
    for tk in sorted(panel["ticker"].unique()):
        s = N.stock_series(tk)
        if s is not None:
            px[tk] = s["Close"]
    px = pd.DataFrame(px).sort_index()
    px = px[~px.index.duplicated()]
    px = px.loc["2006-01-01":pdates.max()]
    ret = px.pct_change(fill_method=None).clip(-CLIP, CLIP)
    ret = ret[ret.notna().sum(axis=1) >= MIN_NAMES]
    folds = []
    for tr, te in purged_walk_forward_date_splits(pdt, n_splits=6, horizon=6, embargo=10):
        t = pdt[te]
        folds.append((t.min(), t.max()))
    return ret, sorted(folds)


# ─── covarianzas y esquemas de pesos ──────────────────────────────────────────

def cov_est(X: np.ndarray, kind: str) -> np.ndarray:
    if kind == "sample":
        return np.cov(X, rowvar=False)
    if kind == "lw":
        return LedoitWolf().fit(X).covariance_
    if kind == "oas":
        return OAS().fit(X).covariance_
    if kind == "cc":
        S = np.cov(X, rowvar=False)
        sd = np.sqrt(np.diag(S))
        C = S / np.outer(sd, sd)
        n = len(S)
        rbar = (C.sum() - n) / (n * (n - 1))
        F = rbar * np.outer(sd, sd)
        np.fill_diagonal(F, sd ** 2)
        return 0.5 * F + 0.5 * S
    raise ValueError(kind)


def w_minvar(S: np.ndarray) -> np.ndarray:
    n = len(S)
    cap = max(0.10, 2.0 / n)
    A = S / np.mean(np.diag(S))
    r = minimize(lambda w: w @ A @ w, np.full(n, 1.0 / n), jac=lambda w: 2 * A @ w, method="SLSQP",
                 bounds=[(0, cap)] * n,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
                 options={"maxiter": 200, "ftol": 1e-12})
    w = np.clip(r.x, 0, cap)
    return w / w.sum()


def w_hrp(S: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(S))
    C = np.clip(S / np.outer(sd, sd), -1, 1)
    D = np.sqrt(np.clip((1 - C) / 2, 0, None))
    np.fill_diagonal(D, 0)
    order = [int(i) for i in leaves_list(linkage(squareform(D, checks=False), method="single"))]

    def cvar(idx):
        sub = S[np.ix_(idx, idx)]
        iv = 1.0 / np.diag(sub)
        iv /= iv.sum()
        return float(iv @ sub @ iv)

    w = np.ones(len(S))
    items = [order]
    while items:
        nxt = []
        for c in items:
            if len(c) > 1:
                m = len(c) // 2
                a, b = c[:m], c[m:]
                va, vb = cvar(a), cvar(b)
                al = 1 - va / (va + vb)
                w[a] *= al
                w[b] *= 1 - al
                nxt += [a, b]
        items = nxt
    return w / w.sum()


def w_erc(S: np.ndarray) -> np.ndarray:
    n = len(S)
    A = S / np.mean(np.diag(S))
    r = minimize(lambda y: 0.5 * y @ A @ y - np.log(y).sum() / n, np.full(n, 1.0 / n),
                 jac=lambda y: A @ y - 1.0 / (n * y), method="L-BFGS-B", bounds=[(1e-8, None)] * n)
    return r.x / r.x.sum()


def make_weights(kind: str, X: np.ndarray) -> np.ndarray:
    n = X.shape[1]
    if kind == "ew":
        return np.full(n, 1.0 / n)
    if kind == "invvol":
        iv = 1.0 / X.std(axis=0, ddof=1)
        return iv / iv.sum()
    est, how = {"mv_sample": ("sample", "mv"), "mv_lw": ("lw", "mv"), "mv_oas": ("oas", "mv"), "mv_cc": ("cc", "mv"),
                "hrp_sample": ("sample", "hrp"), "hrp_lw": ("lw", "hrp"), "rp_lw": ("lw", "rp")}[kind]
    S = cov_est(X, est)
    return {"mv": w_minvar, "hrp": w_hrp, "rp": w_erc}[how](S)


def schedule(ret: pd.DataFrame, kind: str) -> dict[int, np.ndarray]:
    """{posicion de decision p: pesos (n,)} en el ultimo dia habil de cada mes, con info <= p."""
    R = ret.to_numpy(dtype="float64")
    per = pd.Series(ret.index.to_period("M"))
    month_end = np.where(per.ne(per.shift(-1)).to_numpy())[0]
    sch = {}
    for p in month_end:
        if p < WIN:
            continue
        W = R[p - WIN + 1:p + 1]
        ok = (np.isfinite(W).sum(axis=0) >= MIN_OBS) & np.isfinite(W[-1])
        if ok.sum() < MIN_NAMES:
            continue
        X = np.nan_to_num(W[:, ok], nan=0.0)
        w = np.zeros(R.shape[1])
        w[ok] = make_weights(kind, X)
        sch[int(p)] = w
    return sch


def simulate(ret: pd.DataFrame, sch: dict[int, np.ndarray]) -> pd.DataFrame:
    """gross, cost (fraccion), turn por dia. Decision en p, ejecucion al cierre p+1."""
    R = np.nan_to_num(ret.to_numpy(dtype="float64"), nan=0.0)
    T, n = R.shape
    exec_map = {p + 1: w for p, w in sch.items() if p + 1 < T}
    wh = np.zeros(n)
    gross, cost, turn = np.zeros(T), np.zeros(T), np.zeros(T)
    for t in range(T):
        g = float(wh @ R[t])
        gross[t] = g
        if wh.sum() > 0:
            wh = wh * (1 + R[t]) / (1 + g)
        if t in exec_map:
            w = exec_map[t]
            tn = float(np.abs(w - wh).sum())
            turn[t], cost[t] = tn, COST * tn
            wh = w.copy()
    df = pd.DataFrame({"gross": gross, "cost": cost, "turn": turn}, index=ret.index)
    return df.iloc[min(exec_map) + 1:]


# ─── escala de exposicion ─────────────────────────────────────────────────────

def sample_exposure(raw: pd.Series) -> pd.Series:
    pos = np.arange(len(raw))
    return raw.where(pos % EXPO_EVERY == 0).ffill()


def ewma_exposure(g: pd.Series) -> pd.Series:
    sig = ewma_vol(g, 21).shift(1)
    return sample_exposure((TARGET_VOL / sig.where(sig > 0)).clip(upper=1.0))


def fwd_ret(g: pd.Series, h: int) -> pd.Series:
    return ((1 + g).rolling(h).apply(np.prod, raw=True) - 1).shift(-h)


def fhs_quantile(g: pd.Series, h: int) -> tuple[pd.Series, pd.Series]:
    sig = ewma_vol(g, 21)
    s = sig * np.sqrt(h / 252.0)
    z = (fwd_ret(g, h) / s).to_numpy()
    zq = np.full(len(g), np.nan)
    for t in range(len(g)):
        known = z[:max(t - h + 1, 0)]        # y_u cerrado si u+h <= t
        known = known[np.isfinite(known)]
        if len(known) >= 250:
            zq[t] = np.quantile(known, 0.05)
    return s * pd.Series(zq, index=g.index), s


def lgbm_features(g: pd.Series, breadth: pd.Series) -> pd.DataFrame:
    sig21, sig5, sig63 = ewma_vol(g, 21), ewma_vol(g, 5), ewma_vol(g, 63)
    nav = (1 + g).cumprod()
    d = sig21 / np.sqrt(252)
    f = pd.DataFrame({
        "lsig21": np.log(sig21), "r5": np.log(sig5 / sig21), "r63": np.log(sig63 / sig21),
        "mom5": g.rolling(5).sum() / d, "mom21": g.rolling(21).sum() / d, "mom63": g.rolling(63).sum() / d,
        "dd252": nav / nav.rolling(252, min_periods=60).max() - 1, "lastz": g / d,
        "breadth": breadth.reindex(g.index),
    })
    return f.replace([np.inf, -np.inf], np.nan)


def lgbm_quantile(g: pd.Series, h: int, folds, breadth: pd.Series) -> pd.Series:
    import lightgbm as lgb
    s = ewma_vol(g, 21) * np.sqrt(h / 252.0)
    z = fwd_ret(g, h) / s
    F = lgbm_features(g, breadth)
    fok = F.notna().all(axis=1).to_numpy()
    zv = z.to_numpy()
    idx = g.index
    q = pd.Series(np.nan, index=idx)
    for (a, b) in folds:
        te = np.where((idx >= a) & (idx <= b))[0]
        if not len(te):
            continue
        tr = np.arange(0, max(te[0] - h - 10, 0))
        tr = tr[fok[tr] & np.isfinite(zv[tr])]
        if len(tr) < 500:
            continue
        lo, hi = np.percentile(zv[tr], [0.5, 99.5])
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.05, num_leaves=4, n_estimators=150, learning_rate=0.03,
                              min_child_samples=80, subsample=0.7, subsample_freq=1, colsample_bytree=0.8,
                              n_jobs=1, verbose=-1, random_state=7)
        m.fit(F.iloc[tr].to_numpy(dtype="float32"), np.clip(zv[tr], lo, hi).astype("float32"))
        ok = te[fok[te]]
        q.iloc[ok] = np.minimum(m.predict(F.iloc[ok].to_numpy(dtype="float32")), -0.5) * s.iloc[ok].to_numpy()
    return q


def tail_exposure(qhat: pd.Series, h: int) -> pd.Series:
    qstar = Z5 * TARGET_VOL * np.sqrt(h / 252.0)
    raw = (qstar / (-qhat.where(qhat < 0)).shift(1)).clip(upper=1.0)
    return sample_exposure(raw)


def apply_exposure(base: pd.DataFrame, e: pd.Series) -> pd.DataFrame:
    e = e.reindex(base.index)
    ep = e.shift(1).fillna(e)
    net = e * (base["gross"] - base["cost"]) - COST * (e - ep).abs()
    return pd.DataFrame({"net": net, "e": e, "gross_scaled": e * base["gross"],
                         "turn": base["turn"] * e + (e - ep).abs()})


# ─── metricas ─────────────────────────────────────────────────────────────────

def mdd_of(x: np.ndarray) -> float:
    nav = np.cumprod(1 + x)
    return float((nav / np.maximum.accumulate(nav) - 1).min())


def cvar_of(x: np.ndarray, p: float = 0.05) -> float:
    k = max(int(np.ceil(p * len(x))), 1)
    return float(np.sort(x)[:k].mean())


def metrics(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return {}
    ann = float(np.prod(1 + x) ** (252 / len(x)) - 1)
    md = mdd_of(x)
    sd = x.std(ddof=1)
    return {"n": int(len(x)), "ret_ann": ann, "vol": float(sd * np.sqrt(252)), "mdd": md, "cvar5": cvar_of(x),
            "calmar": float(ann / abs(md)) if md < 0 else float("nan"),
            "sharpe": float(x.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan"), "total": float(np.prod(1 + x) - 1)}


def block_idx(n: int, B: int, L: int, rng) -> np.ndarray:
    nb = int(np.ceil(n / L))
    st = rng.integers(0, n, size=(B, nb))
    return ((st[:, :, None] + np.arange(L)[None, None, :]) % n).reshape(B, -1)[:, :n]


def boot_stats(x: np.ndarray, idx: np.ndarray) -> dict:
    out = {"mdd": [], "cvar5": [], "sharpe": []}
    k = max(int(np.ceil(0.05 * idx.shape[1])), 1)
    for a in range(0, idx.shape[0], 250):          # en lotes: poca RAM
        S = x[idx[a:a + 250]].astype("float64")
        nav = np.cumprod(1 + S, axis=1)
        out["mdd"].append((nav / np.maximum.accumulate(nav, axis=1) - 1).min(axis=1))
        out["cvar5"].append(np.sort(S, axis=1)[:, :k].mean(axis=1))
        out["sharpe"].append(S.mean(axis=1) / S.std(axis=1, ddof=1) * np.sqrt(252))
    return {m: np.concatenate(v) for m, v in out.items()}


def nw_var(d: np.ndarray, lag: int) -> float:
    n = len(d)
    e = d - d.mean()
    s = e @ e / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return s / n


def dm_test(la: np.ndarray, lb: np.ndarray, lag: int = 10) -> dict:
    d = np.asarray(la) - np.asarray(lb)
    d = d[np.isfinite(d)]
    t = float(d.mean() / np.sqrt(max(nw_var(d, lag), 1e-30)))
    return {"mean_diff": float(d.mean()), "t": t, "p": float(2 * (1 - stats.norm.cdf(abs(t))))}


def kupiec_christoffersen(hits: np.ndarray, p: float = 0.05) -> dict:
    n, x = len(hits), int(hits.sum())
    if n < 20 or x == 0 or x == n:
        return {"n": n, "hits": x, "rate": x / max(n, 1), "kupiec_p": float("nan"), "cc_p": float("nan")}
    ph = x / n
    ll = lambda q: (n - x) * np.log(1 - q) + x * np.log(q)
    pof = -2 * (ll(p) - ll(ph))
    a, b = hits[:-1], hits[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    pi01, pi11 = n01 / max(n00 + n01, 1), n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def l(pp, c0, c1):
        return (c0 * np.log(1 - pp) if c0 else 0.0) + (c1 * np.log(pp) if c1 and pp > 0 else 0.0)

    ind = -2 * (l(pi, n00 + n10, n01 + n11) - l(pi01, n00, n01) - l(pi11, n10, n11)) if 0 < pi01 < 1 else float("nan")
    cc = pof + ind
    return {"n": n, "hits": x, "rate": float(ph), "kupiec_p": float(1 - stats.chi2.cdf(pof, 1)),
            "cc_p": float(1 - stats.chi2.cdf(cc, 2)) if np.isfinite(cc) else float("nan")}


def pinball(y: np.ndarray, q: np.ndarray, tau: float = 0.05) -> np.ndarray:
    u = y - q
    return np.maximum(tau * u, (tau - 1) * u)


# ─── main ─────────────────────────────────────────────────────────────────────

BASES = ["ew", "invvol", "mv_sample", "mv_lw", "mv_oas", "mv_cc", "hrp_sample", "hrp_lw", "rp_lw"]
SCALE_BASES = ["ew", "hrp_lw", "mv_lw"]
SCALERS = ["ewma", "fhs5", "fhs21", "lgbm5", "lgbm21"]


def plain(b: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"net": b["gross"] - b["cost"], "e": 1.0, "gross_scaled": b["gross"], "turn": b["turn"]})


def main() -> None:
    ret, folds = load_returns()
    print(f"retornos {ret.shape}, {ret.index.min().date()}..{ret.index.max().date()}", flush=True)
    for i, (a, b) in enumerate(folds, 1):
        print(f"  F{i}: {a.date()}..{b.date()}", flush=True)
    t0, t1 = folds[0][0], folds[-1][1]
    neg = (ret.rolling(21, min_periods=15).sum() < 0).sum(axis=1)
    breadth = neg / ret.rolling(21, min_periods=15).sum().notna().sum(axis=1)

    base, S, fc = {}, {}, {}
    for k in BASES:
        base[k] = simulate(ret, schedule(ret, k))
        S[k] = plain(base[k])
        print(f"base {k}: {len(base[k])} dias, turn/anio={base[k]['turn'].sum() / (len(base[k]) / 252):.2f}", flush=True)
    for k in SCALE_BASES:
        g = base[k]["gross"]
        S[f"{k}|ewma"] = apply_exposure(base[k], ewma_exposure(g))
        for h in (5, 21):
            q, s = fhs_quantile(g, h)
            fc[(k, f"fhs{h}")] = q
            S[f"{k}|fhs{h}"] = apply_exposure(base[k], tail_exposure(q, h))
            q2 = lgbm_quantile(g, h, folds, breadth)
            fc[(k, f"lgbm{h}")] = q2
            fc[(k, f"norm{h}")] = -Z5 * s
            S[f"{k}|lgbm{h}"] = apply_exposure(base[k], tail_exposure(q2, h))
        print(f"escalas {k} listas", flush=True)

    # tilt Ridge-rank h21 (solo hay score en fechas OOS)
    tl = pd.read_parquet(SCR / "n1c_oos_rankaug_h21.parquet")
    tl["date"] = pd.to_datetime(tl["date"])
    by_date = {d: g for d, g in tl.groupby("date")}
    sch_h = schedule(ret, "hrp_lw")
    cols = list(ret.columns)
    for kk in (0.25, 0.5):
        sch_t = {}
        for p, w in sch_h.items():
            sub = by_date.get(ret.index[p])
            if sub is not None and len(sub) >= MIN_NAMES:
                z = ((sub.set_index("ticker")["score"] - sub["score"].mean()) / (sub["score"].std() + 1e-12)).clip(-2, 2)
                sc = pd.Series(0.0, index=cols)
                common = z.index.intersection(cols)
                sc.loc[common] = z.loc[common]
                w2 = w * np.clip(1 + kk * sc.to_numpy(), 0, None)
                if w2.sum() > 0:
                    w = w2 / w2.sum()
            sch_t[p] = w
        S[f"hrp_lw_tilt{kk}"] = plain(simulate(ret, sch_t))

    seg = lambda df, a, b: df.loc[a:b]
    R = {"meta": {"folds": [[str(a.date()), str(b.date())] for a, b in folds], "cost_side_pct": COST * 100,
                  "target_vol": TARGET_VOL, "n_names": int(ret.shape[1]), "boot": {"B": BOOT_B, "block": BOOT_BLOCK}},
         "configs": {}}
    for nm, df in S.items():
        d = seg(df, t0, t1).dropna(subset=["net"])
        c = {"pooled": metrics(d["net"].to_numpy()), "mean_exposure": float(d["e"].mean()),
             "turnover_anual": float(d["turn"].sum() / (len(d) / 252)),
             "por_fold": [metrics(seg(df, a, b)["net"].dropna().to_numpy()) for a, b in folds],
             "estres": {lab: metrics(seg(df, a, b)["net"].dropna().to_numpy()) for lab, (a, b) in STRESS.items()},
             "estres_sin_fold": {}}
        if "lgbm" not in nm and "tilt" not in nm:
            c["estres_sin_fold"] = {lab: metrics(seg(df, a, b)["net"].dropna().to_numpy()) for lab, (a, b) in STRESS_NOFIT.items()}
        R["configs"][nm] = c

    rng = np.random.default_rng(11)
    ref_idx = seg(S["ew"], t0, t1).index
    idx_b = block_idx(len(ref_idx), BOOT_B, BOOT_BLOCK, rng)
    cache = {}

    def boot(nm):
        if nm not in cache:
            cache[nm] = boot_stats(S[nm]["net"].reindex(ref_idx).fillna(0.0).to_numpy(), idx_b)
        return cache[nm]

    def compare(nm: str, bl: str) -> dict:
        A, B = R["configs"][nm], R["configs"][bl]
        out = {"strategy": nm, "baseline": bl, "metrics": {}}
        for M in ("mdd", "cvar5", "sharpe"):
            fw = int(sum(1 for fa, fb in zip(A["por_fold"], B["por_fold"]) if fa and fb and fa[M] > fb[M]))
            dd = boot(nm)[M] - boot(bl)[M]
            lo, hi = np.percentile(dd, [5, 95])
            key = {"mdd": "mdd", "cvar5": "cvar5", "sharpe": "total"}[M]
            st = {}
            for lab in STRESS:
                sa, sb = A["estres"][lab], B["estres"][lab]
                st[lab] = {"est": sa.get(key), "base": sb.get(key), "mejora": bool(sa and sb and sa[key] > sb[key])}
            out["metrics"][M] = {"folds_win": fw, "pooled_diff": float(A["pooled"][M] - B["pooled"][M]),
                                 "ci90": [float(lo), float(hi)], "ci_excluye0_favor": bool(lo > 0), "estres": st,
                                 "POSITIVO": bool(fw >= 4 and lo > 0 and all(v["mejora"] for v in st.values()))}
        out["positivo_en"] = [m for m, v in out["metrics"].items() if v["POSITIVO"]]
        return out

    comps = []
    for nm in BASES:
        for bl in ("ew", "mv_sample"):
            if nm != bl:
                comps.append(compare(nm, bl))
    for k in SCALE_BASES:
        for sc in SCALERS:
            if f"{k}|{sc}" != "ew|ewma":
                comps.append(compare(f"{k}|{sc}", "ew|ewma"))
    for kk in (0.25, 0.5):
        comps.append(compare(f"hrp_lw_tilt{kk}", "hrp_lw"))
    comps.append(compare("ew|ewma", "ew"))
    comps.append(compare("hrp_lw|ewma", "hrp_lw"))
    R["comparaciones"] = comps

    dm = {}
    pairs = [(nm, bl) for nm in BASES for bl in ("ew", "mv_sample") if nm != bl]
    pairs += [(f"{k}|{sc}", "ew|ewma") for k in SCALE_BASES for sc in SCALERS if f"{k}|{sc}" != "ew|ewma"]
    for nm, bl in pairs:
        a = seg(S[nm], t0, t1)["gross_scaled"]
        b = seg(S[bl], t0, t1)["gross_scaled"].reindex(a.index)
        dm[f"{nm} vs {bl}"] = dm_test(a.to_numpy() ** 2, b.to_numpy() ** 2)
    R["dm_varianza_realizada"] = dm

    qev = {}
    for k in SCALE_BASES[:2]:
        g = base[k]["gross"]
        for h in (5, 21):
            y = fwd_ret(g, h)
            for mname in (f"norm{h}", f"fhs{h}", f"lgbm{h}"):
                q = fc[(k, mname)]
                m = ((y.index >= t0) & (y.index <= t1) & y.notna().to_numpy() & q.notna().to_numpy())
                yy, qq, dts = y[m].to_numpy(), q[m].to_numpy(), y.index[m]
                pb = pinball(yy, qq)
                step = np.arange(0, len(yy), h)
                ev = {"pinball": float(pb.mean()), "cobertura_todo": float((yy < qq).mean()),
                      **{f"noover_{kk}": v for kk, v in kupiec_christoffersen((yy[step] < qq[step]).astype(int)).items()},
                      "por_fold": {}}
                for i, (a, b) in enumerate(folds, 1):
                    mm = (dts >= a) & (dts <= b)
                    ev["por_fold"][f"F{i}"] = {"pinball": float(pb[mm].mean()), "cobertura": float((yy[mm] < qq[mm]).mean())}
                if mname == f"norm{h}":
                    ref = pb
                else:
                    nref = qev[f"{k}|h{h}|norm{h}"]
                    ev["dm_vs_norm"] = dm_test(pb, ref, lag=2 * h)
                    ev["folds_mejor_pinball_que_norm"] = int(sum(ev["por_fold"][f"F{i}"]["pinball"] < nref["por_fold"][f"F{i}"]["pinball"] for i in range(1, 7)))
                qev[f"{k}|h{h}|{mname}"] = ev
    R["cuantiles"] = qev

    for nm, c in R["configs"].items():
        grp = "tilt" if "tilt" in nm else ("pesos" if "|" not in nm else ("escala" if nm.startswith("ew|") else "combo"))
        mt = {k: c["pooled"].get(k) for k in ("vol", "mdd", "cvar5", "calmar", "sharpe", "ret_ann")}
        mt["turnover_anual"] = c["turnover_anual"]
        log_trial(f"n2a_{grp}:{nm}", {"config": nm, "cost_side": COST, "rebal": "monthly", "win": WIN, "target_vol": TARGET_VOL, "folds": 6},
                  mt, n_obs=c["pooled"]["n"], path=TRIALS)
    for key, ev in qev.items():
        log_trial(f"n2a_cuantil:{key}", {"config": key, "tau": 0.05}, {"pinball": ev["pinball"], "cobertura": ev["cobertura_todo"]},
                  n_obs=int(ev["noover_n"]), path=TRIALS)

    (OUT / "n2a_riesgo.json").write_text(json.dumps(R, indent=1, default=lambda o: float(o)), encoding="utf-8")
    write_md(R)
    print("OK", flush=True)

NOTAS = """
### Lectura (regla pre-registrada aplicada; ver tablas 1-6)

- Ningun resultado es POSITIVO en MDD ni en Sharpe: el unico "positivo" es CVaR 5 % y viene de MENOR VOL/exposicion (los pesos de riesgo pagan Sharpe/Calmar; las escalas bajan retorno). No es mejora gratis.
- Shrinkage (LW/OAS/cc) = covarianza muestral: dif. de varianza realizada no significativa (DM p 0.24-0.50), 2/6 folds. HRP y risk parity tienen MAS varianza realizada que min-var muestral (DM p<0.01) y no mejoran su MDD/CVaR; frente a 1/N bajan vol pero Sharpe neto <= 1/N.
- Escala EWMA vs 1/N: MDD 6/6 folds (-34.5% -> -19.3%) pero Sharpe 0.44 -> 0.25 con costo 0.30%/lado (el hallazgo previo "mismo Sharpe" era con 0.05%). El IC bootstrap de MDD incluye 0 (el bootstrap por bloques rompe la dependencia de trayectoria: el MDD agrupado depende de un solo tramo, 2020).
- FHS 5/21d vs EWMA: MDD y CVaR mejoran 6/6 folds y en 2020/2022, pero con exposicion media menor (0.70/0.66 vs 0.77), Sharpe peor (6/6 folds) y Calmar menor: es mas conservador, no mas eficiente.
- LightGBM cuantil: pinball algo mejor en h5 (DM t -1.6, no significativo, 4-5/6 folds), peor en h21; no mejora el MDD vs EWMA y 2020 sigue siendo el unico tramo que decide el MDD agrupado. Cobertura Kupiec/CC aceptable en todos (no rechaza 5 %).
- Tilt Ridge-rank sobre HRP: turnover 1.75 -> 3.5/6.1 por anio, Sharpe y MDD sin mejora (2-3/6 folds).
- 2008 queda fuera de los folds (TRAIN del panel); solo se reporta para estrategias sin parametros ajustados.
"""


def f(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x * 100:.{nd}f}%" if pct else f"{x:.{nd}f}"


def write_md(R: dict) -> None:
    C = R["configs"]
    L = ["# N2-A motor de riesgo (OOS 6 folds, rebalanceo mensual, costo 0.30%/lado)", "",
         f"Folds (F1 antiguo .. F6 reciente): {R['meta']['folds']}. Objetivo vol {R['meta']['target_vol']:.0%}. Bootstrap por bloques {R['meta']['boot']}.",
         "Pre-registro completo en el docstring de scripts/n2a_riesgo.py.", "",
         "## 1. Metricas agrupadas OOS (neto de costos)", "",
         "| config | vol | MDD | CVaR5 d | Calmar | Sharpe | ret anual | exposicion | turnover/anio |", "|---|---|---|---|---|---|---|---|---|"]
    for nm, c in C.items():
        p = c["pooled"]
        L.append(f"| {nm} | {f(p['vol'], 1, True)} | {f(p['mdd'], 1, True)} | {f(p['cvar5'], 2, True)} | {f(p['calmar'])} | {f(p['sharpe'])} | "
                 f"{f(p['ret_ann'], 1, True)} | {f(c['mean_exposure'])} | {f(c['turnover_anual'])} |")
    key = ["ew", "mv_sample", "mv_lw", "hrp_lw", "rp_lw", "ew|ewma", "ew|fhs5", "ew|lgbm5", "hrp_lw|ewma", "hrp_lw|fhs5", "hrp_lw|lgbm5", "hrp_lw|lgbm21"]
    L += ["", "## 2. Por fold: MDD / Sharpe", "", "| config | " + " | ".join(f"F{i}" for i in range(1, 7)) + " |", "|---|" + "---|" * 6]
    for nm in key:
        if nm in C:
            L.append(f"| {nm} | " + " | ".join(f"{f(x.get('mdd'), 1, True)} / {f(x.get('sharpe'))}" for x in C[nm]["por_fold"]) + " |")
    L += ["", "## 3. Estres (MDD ventana / retorno ventana)", "", "| config | 2020 | 2022 | 2008 (sin fold, sin parametros) |", "|---|---|---|---|"]
    g = lambda d: f"{f(d.get('mdd'), 1, True)} / {f(d.get('total'), 1, True)}" if d else "-"
    for nm in key:
        if nm in C:
            L.append(f"| {nm} | {g(C[nm]['estres']['2020'])} | {g(C[nm]['estres']['2022'])} | {g(C[nm]['estres_sin_fold'].get('2008', {}))} |")
    L += ["", "## 4. Comparaciones vs baseline (folds que mejoran de 6 / diferencia agrupada [IC90 bootstrap]); POSITIVO segun regla", "",
          "| estrategia | baseline | MDD | CVaR | Sharpe | positivo en |", "|---|---|---|---|---|---|"]
    for c in R["comparaciones"]:
        m = c["metrics"]
        cell = lambda k, pct: f"{m[k]['folds_win']}/6 / {f(m[k]['pooled_diff'], 2, pct)} [{f(m[k]['ci90'][0], 2, pct)}, {f(m[k]['ci90'][1], 2, pct)}]"
        L.append(f"| {c['strategy']} | {c['baseline']} | {cell('mdd', True)} | {cell('cvar5', True)} | {cell('sharpe', False)} | {','.join(c['positivo_en']) or 'ninguno'} |")
    L += ["", "## 5. Diebold-Mariano sobre varianza realizada (r^2 bruto escalado; dif<0 => menos varianza que el baseline)", "", "| comparacion | dif media | t | p |", "|---|---|---|---|"]
    for k, v in R["dm_varianza_realizada"].items():
        L.append(f"| {k} | {v['mean_diff']:.2e} | {v['t']:.2f} | {v['p']:.3f} |")
    L += ["", "## 6. Pronostico de cola 5 % (agrupado OOS)", "", "| serie / h / modelo | pinball | cobertura (todo) | hits no solap. | Kupiec p | CC p | DM vs normal (t) | folds mejor que normal |", "|---|---|---|---|---|---|---|---|"]
    for k, v in R["cuantiles"].items():
        d = v.get("dm_vs_norm", {})
        L.append(f"| {k} | {v['pinball']:.5f} | {f(v['cobertura_todo'], 1, True)} | {v['noover_hits']}/{v['noover_n']} | {f(v['noover_kupiec_p'], 3)} | "
                 f"{f(v['noover_cc_p'], 3)} | {f(d.get('t'))} | {v.get('folds_mejor_pinball_que_norm', '-')} |")
    pos = [(c["strategy"], c["baseline"], c["positivo_en"]) for c in R["comparaciones"] if c["positivo_en"]]
    L += ["", "## 7. Veredicto", "", f"Comparaciones POSITIVAS segun la regla pre-registrada: {pos if pos else 'ninguna'}.", NOTAS]
    (OUT / "n2a_riesgo.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
