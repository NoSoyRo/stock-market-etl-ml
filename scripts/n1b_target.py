"""
Nivel 1 / idea B: cambiar el TARGET (cross-sectional rank / residual / demeaned) y features simples.

Reusa scripts/n0a (panel, splits purgados), n0b (liquidez), n1a (series de precio). No edita codigo existente.

SIN LEAKAGE
  * Entrada Open t+1, salida Close t+h+1 (misma convencion que el label dir_5 del repo; h=5 se verifica = ret_fwd).
  * Los MISMOS 6 folds de TEST (splits sobre el panel completo); la purga del train usa h+1 dias.
  * Features: las del panel (info hasta el cierre t) + rev1d (cierre t / cierre t-1) y calendario (conocido de antemano).
  * Targets solo ven retornos futuros en TRAIN (purgado). Residual: beta rodante 252d con datos hasta t-1.
    Demeaned por ticker: media EXPANDING del ret_h del ticker, rezagada h+1 dias (solo etiquetas ya cerradas).
  * Features ranqueadas por fecha (percentil cross-sectional, solo informacion de ese dia).

Ejecucion: .venv/bin/python scripts/n1b_target.py
"""
from __future__ import annotations

import gc
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import scripts.n0a_diagnostico as A
import scripts.n0b_diagnostico as B
import scripts.n1a_leadlag as N
from scripts.run_purged_logistic import C_REG, N_BINS_WOE, TOP_FEATURES, transform_woe_features
from src.ml.features import FEATURE_COLS
from src.ml.purged_cv import purged_walk_forward_date_splits
from src.ml.trial_stats import deflated_sharpe
from src.ml.trials import DEFAULT_PATH as TRIALS_PATH, count_trials, log_trial
from src.ml.woe import compute_all_woe

OUT = ROOT / "data" / "eval"
SCRATCH = N.SCRATCH
HORIZONS = (1, 5, 21)
COST = A.COST_SIDE
RIDGE_ALPHA = 1000.0
TEST_START = pd.Timestamp("2024-01-01")
BETA_WIN, BETA_MIN = 252, 120


# ─── datos ────────────────────────────────────────────────────────────────────

def build_extra(df: pd.DataFrame) -> pd.DataFrame:
    """Por (ticker, fecha): ret1d, r{h} (h=1,5,21) y beta rodante (t-1) contra mercado EW."""
    raws = {}
    for tk in df["ticker"].unique():
        s = N.stock_series(tk)
        if s is not None:
            raws[tk] = s
    daily = pd.DataFrame({tk: np.log(s["Close"]).diff() * 100.0 for tk, s in raws.items()})
    mkt_d = daily.mean(axis=1)
    frames = []
    for tk, s in raws.items():
        o, c = s["Open"], s["Close"]
        x = pd.DataFrame(index=s.index)
        x["ret1d"] = np.log(c).diff() * 100.0
        for h in HORIZONS:
            x[f"r{h}"] = (c.shift(-(h + 1)) / o.shift(-1) - 1.0) * 100.0
        r = daily[tk]
        beta = (r.rolling(BETA_WIN, min_periods=BETA_MIN).cov(mkt_d) /
                mkt_d.rolling(BETA_WIN, min_periods=BETA_MIN).var()).shift(1)
        x["beta"] = beta.reindex(x.index)
        x["ticker"] = tk
        x.index.name = "date"
        frames.append(x.reset_index())
    E = pd.concat(frames, ignore_index=True)
    E["date"] = pd.to_datetime(E["date"])
    for c in E.columns:
        if E[c].dtype == "float64":
            E[c] = E[c].astype("float64")
    return E


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("date")
    for h in HORIZONS:
        r = df[f"r{h}"]
        ok = g[f"r{h}"].transform("count") >= A.MIN_NAMES
        df[f"t_rank{h}"] = np.where(ok, g[f"r{h}"].rank(pct=True) - 0.5, np.nan).astype("float32")
        mh = g[f"r{h}"].transform("mean")                  # retorno de mercado EW de la misma ventana (solo en la etiqueta)
        df[f"t_resid{h}"] = np.where(ok, r - df["beta"].fillna(1.0) * mh, np.nan).astype("float32")
        em = df.groupby("ticker")[f"r{h}"].transform(lambda s: s.expanding(min_periods=60).mean().shift(h + 1))
        df[f"t_dm{h}"] = np.where(ok, r - em.fillna(0.0), np.nan).astype("float32")
        df[f"t_raw{h}"] = np.where(ok, r, np.nan).astype("float32")
    return df


def rank_features(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    R = df.groupby("date")[cols].rank(pct=True) - 0.5
    return R.fillna(0.0).to_numpy(dtype="float32")


def calendar_cols(df: pd.DataFrame) -> pd.DataFrame:
    cal = pd.Series(sorted(df["date"].unique()))
    ym = cal.dt.to_period("M")
    dtme = cal.groupby(ym).cumcount(ascending=False)      # dias habiles restantes en el mes (0 = ultimo)
    m = pd.DataFrame({"date": cal, "dtme": dtme.values})
    m["last5"] = (m["dtme"] < 5).astype("float32")
    return df.merge(m, on="date", how="left")


# ─── estadistica ──────────────────────────────────────────────────────────────

def ic_by_date(d: pd.DataFrame, min_names: int = A.MIN_NAMES) -> pd.Series:
    n = d.groupby("date")["score"].transform("size")
    d = d[n >= min_names]
    a = d.groupby("date")["score"].rank()
    b = d.groupby("date")["ret"].rank()
    a = a - a.groupby(d["date"]).transform("mean")
    b = b - b.groupby(d["date"]).transform("mean")
    t = pd.DataFrame({"date": d["date"].values, "ab": (a * b).values, "aa": (a * a).values, "bb": (b * b).values})
    s = t.groupby("date").sum()
    return (s["ab"] / np.sqrt(s["aa"] * s["bb"])).replace([np.inf, -np.inf], np.nan).dropna()


def alpha_rebal(d: pd.DataFrame, h: int, min_names: int = A.MIN_NAMES) -> pd.DataFrame:
    """Rebalanceo cada h fechas (por fold): largo tercil superior vs media de la celda; turnover real de la cartera."""
    rows = []
    for fo, z in d.groupby("fold"):
        dates = np.sort(z["date"].unique())[::h]
        grp = dict(list(z.groupby("date")))
        prev = None
        for dt in dates:
            w = grp.get(dt)
            if w is None or len(w) < min_names:
                continue
            r = w["score"].rank(pct=True)
            top = w[r > 2 / 3]
            ex = top["ret"].mean() - w["ret"].mean()
            cur = set(top["ticker"])
            turn = 1.0 if prev is None else 1.0 - len(cur & prev) / max(len(cur), 1)
            prev = cur
            if np.isfinite(ex):
                rows.append((dt, fo, ex, turn))
    return pd.DataFrame(rows, columns=["date", "fold", "excess", "turnover"])


def tstat(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 2 and v.std() > 0 else float("nan")


def summarize_alpha(a: pd.DataFrame) -> dict:
    if a.empty:
        return {}
    full = a["excess"] - 2 * COST                          # ida y vuelta completo por rebalanceo (como la linea base)
    turn = a["excess"] - 2 * COST * a["turnover"]          # costo solo sobre lo que cambia
    byf_full = full.groupby(a["fold"]).mean()
    byf_turn = turn.groupby(a["fold"]).mean()
    return {"n_rebal": int(len(a)), "excess_gross_pct": float(a["excess"].mean()),
            "alpha_net_full_pct": float(full.mean()), "alpha_net_full_t": tstat(full.values),
            "alpha_net_turn_pct": float(turn.mean()), "alpha_net_turn_t": tstat(turn.values),
            "turnover_mean": float(a["turnover"].mean()),
            "folds_pos_full": int((byf_full > 0).sum()), "folds_pos_turn": int((byf_turn > 0).sum()),
            "alpha_turn_by_fold": {int(k): float(v) for k, v in byf_turn.items()}}


def eval_oos(oos: pd.DataFrame, h: int, liq: pd.DataFrame) -> tuple[dict, np.ndarray]:
    lag = max(A.NW_LAG, h)
    ic = ic_by_date(oos)
    m, _, t = A.nw_t(ic.values, lag)
    a = alpha_rebal(oos, h)
    r = {"n_oos": int(len(oos)), "auc": A.safe_auc((oos["ret"] > 0).astype(int), oos["score"]), "ic": m, "t_nw": t,
         "nw_lag": lag, "n_dates_ic": int(len(ic)), **summarize_alpha(a)}
    ff = {}
    for fo, z in oos.groupby("fold"):
        icf = ic_by_date(z)
        ff[int(fo)] = float(icf.mean()) if len(icf) else None
    r["ic_by_fold"] = ff
    r["folds_ic_pos"] = int(sum(1 for v in ff.values() if v and v > 0))
    yrs = {}
    ad = pd.to_datetime(a["date"])
    nt = a["excess"] - 2 * COST * a["turnover"]
    for yv, v in ic.groupby(ic.index.year):
        mm, _, tt = A.nw_t(v.values, lag)
        sel = (ad.dt.year == yv)
        yrs[int(yv)] = {"ic": mm, "t": tt, "n": int(len(v)), "alpha_net_turn_pct": float(nt[sel].mean()) if sel.any() else None}
    post = ic[ic.index >= TEST_START]
    mm, _, tt = A.nw_t(post.values, lag)
    sel = ad >= TEST_START
    yrs["2024+"] = {"ic": mm, "t": tt, "n": int(len(post)), "alpha_net_turn_pct": float(nt[sel].mean()) if sel.any() else None}
    r["por_anio"] = yrs
    d = oos.merge(liq[["ticker", "date", "adv"]], on=["ticker", "date"], how="left")
    rk = d.groupby("date")["adv"].rank(pct=True)
    d["terc"] = pd.cut(rk, [0, 1 / 3, 2 / 3, 1.0001], labels=["T1_iliquido", "T2", "T3_liquido"])
    tc = {}
    for lab in ("T1_iliquido", "T2", "T3_liquido"):
        z = d[d["terc"] == lab]
        if len(z) < 200:
            continue
        icz = ic_by_date(z, B.MIN_CELL)
        mz, _, tz = A.nw_t(icz.values, lag)
        az = alpha_rebal(z, h, B.MIN_CELL)
        tc[lab] = {"ic": mz, "t": tz,
                   "alpha_net_full_pct": float((az["excess"] - 2 * COST).mean()) if len(az) else None,
                   "alpha_net_turn_pct": float((az["excess"] - 2 * COST * az["turnover"]).mean()) if len(az) else None}
    r["por_tercil"] = tc
    return r, nt.to_numpy() / 100.0


# ─── modelos ──────────────────────────────────────────────────────────────────

def folds_iter(df: pd.DataFrame, h: int):
    return purged_walk_forward_date_splits(pd.to_datetime(df["date"]), n_splits=A.N_SPLITS, horizon=h + 1, embargo=A.EMBARGO)


def ridge_oos(df, X, tcol, h) -> pd.DataFrame:
    tgt = df[tcol].to_numpy(dtype="float64")
    ret = df[f"r{h}"].to_numpy(dtype="float64")
    parts = []
    for fi, (tr, te) in enumerate(folds_iter(df, h)):
        tr = tr[np.isfinite(tgt[tr])]
        te = te[np.isfinite(ret[te])]
        lo, hi = np.nanpercentile(tgt[tr], [1, 99])
        m = Ridge(alpha=RIDGE_ALPHA).fit(X[tr], np.clip(tgt[tr], lo, hi))
        parts.append(pd.DataFrame({"date": df["date"].values[te], "ticker": df["ticker"].values[te],
                                   "ret": ret[te], "score": m.predict(X[te]), "fold": fi + 1}))
    return pd.concat(parts, ignore_index=True)


def woe_oos(df, cols, tcol, h) -> pd.DataFrame:
    """WoE+logistica; etiqueta = (tcol > 0); tcol=None => direccion ret_h > 0."""
    ret = df[f"r{h}"].to_numpy(dtype="float64")
    tgt = ret if tcol is None else df[tcol].to_numpy(dtype="float64")
    parts = []
    for fi, (tr, te) in enumerate(folds_iter(df, h)):
        tr = tr[np.isfinite(tgt[tr])]
        te = te[np.isfinite(ret[te])]
        ytr = (tgt[tr] > 0).astype(int)
        if len(np.unique(ytr)) < 2:
            continue
        Xtr, Xte = df.iloc[tr], df.iloc[te]
        t = Xtr[cols].copy()
        t["y"] = ytr
        woe = compute_all_woe(t, feature_cols=cols, target_col="y", n_bins=N_BINS_WOE)
        sel = sorted([f_ for f_, w in woe.items() if w.n_bins >= 2 and w.iv > 0 and np.isfinite(w.iv)],
                     key=lambda f_: woe[f_].iv, reverse=True)[:TOP_FEATURES]
        if not sel:
            continue
        lr = LogisticRegression(C=C_REG, max_iter=1000, random_state=42).fit(transform_woe_features(Xtr, sel, woe), ytr)
        p = lr.predict_proba(transform_woe_features(Xte, sel, woe))[:, 1]
        parts.append(pd.DataFrame({"date": Xte["date"].values, "ticker": Xte["ticker"].values,
                                   "ret": ret[te], "score": p, "fold": fi + 1}))
        del t, Xtr, Xte
        gc.collect()
    return pd.concat(parts, ignore_index=True)


def signal_oos(df, col, h) -> pd.DataFrame:
    ret = df[f"r{h}"].to_numpy(dtype="float64")
    parts = []
    for fi, (tr, te) in enumerate(folds_iter(df, h)):
        te = te[np.isfinite(ret[te])]
        parts.append(pd.DataFrame({"date": df["date"].values[te], "ticker": df["ticker"].values[te], "ret": ret[te],
                                   "score": df[col].to_numpy(dtype="float64")[te], "fold": fi + 1}))
    return pd.concat(parts, ignore_index=True)


# ─── calendario ───────────────────────────────────────────────────────────────

def nw_slope_t(y: np.ndarray, x: np.ndarray, lag: int) -> tuple[float, float]:
    X = np.column_stack([np.ones(len(x)), x])
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for k in range(1, lag + 1):
        G = Xe[k:].T @ Xe[:-k]
        S += (1 - k / (lag + 1)) * (G + G.T)
    V = XtXi @ S @ XtXi
    return float(beta[1]), float(beta[1] / np.sqrt(V[1, 1]))


def calendar_study(df: pd.DataFrame) -> dict:
    out = {}
    for h in HORIZONS:
        m = df.groupby("date").agg(r=(f"r{h}", "mean"), last5=("last5", "first"), dtme=("dtme", "first")).dropna().sort_index()
        b, t = nw_slope_t(m["r"].to_numpy(), m["last5"].to_numpy(), max(5, h))
        ent = m[m["dtme"] == 5]["r"]       # dtme=5 en t => entrada Open t+1 = primer dia de la ventana last5
        out[f"h{h}"] = {"mean_last5_pct": float(m.loc[m.last5 == 1, "r"].mean()), "mean_resto_pct": float(m.loc[m.last5 == 0, "r"].mean()),
                        "dif_pct": b, "t_hac": t, "n_last5": int((m.last5 == 1).sum()), "n_total": int(len(m)),
                        "entrada_dtme5_mean_pct": float(ent.mean()), "entrada_dtme5_t": tstat(ent.values),
                        "entrada_dtme5_neto_pct": float(ent.mean() - 2 * COST), "n_meses": int(len(ent))}
    return out


# ─── registro / salida ────────────────────────────────────────────────────────

def log(name, cfg, r, h):
    base = {"tanda": "n1b", "n_splits": A.N_SPLITS, "embargo": A.EMBARGO, "horizon_days": h, "label_span": h + 1,
            "entry": "next_open", "cost_side_pct": COST}
    base.update(cfg)
    met = {k: r.get(k) for k in ("auc", "ic", "t_nw", "alpha_net_full_pct", "alpha_net_turn_pct", "turnover_mean",
                                 "folds_pos_full", "folds_pos_turn")}
    met["ic_2024p"] = r["por_anio"]["2024+"]["ic"]
    log_trial(name, base, B.clean(met), int(r["n_oos"]), path=str(ROOT / TRIALS_PATH))


def f(x, nd=4):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def main() -> None:
    df = A.load_panel(SCRATCH / "panel.parquet")
    cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df[np.isfinite(df["ret_fwd"])].reset_index(drop=True)
    print(f"Panel {len(df):,} filas, {df['ticker'].nunique()} tickers, {len(cols)} features", flush=True)
    df = df.merge(build_extra(df), on=["date", "ticker"], how="left")
    gc.collect()
    chk = (df["r5"] - df["ret_fwd"]).abs()
    audit = {"r5_vs_ret_fwd_max_abs_dif": float(chk.max()), "r5_vs_ret_fwd_frac_gt_1e-3": float((chk > 1e-3).mean()),
             "n_nan_r1": int(df["r1"].isna().sum()), "n_nan_r5": int(df["r5"].isna().sum()), "n_nan_r21": int(df["r21"].isna().sum())}
    print("auditoria etiqueta:", audit, flush=True)
    df = add_targets(df)
    df = calendar_cols(df)
    df["rev1d"] = -df["ret1d"]
    X = rank_features(df, cols)
    rr = (df.groupby("date")["rev1d"].rank(pct=True) - 0.5).fillna(0.0).to_numpy(dtype="float32")
    X_aug = np.column_stack([X, rr, rr * df["last5"].to_numpy(dtype="float32")]).astype("float32")
    liq, _ = B.liquidity_panel(sorted(df["ticker"].unique()))
    R = {"meta": {"auditoria": audit, "ridge_alpha": RIDGE_ALPHA, "features": len(cols), "horizontes": list(HORIZONS),
                  "costo_por_lado_pct": COST, "convencion": "entrada Open t+1, salida Close t+h+1 (h=5 == ret_fwd del repo)",
                  "nw_lag": "max(5,h)", "cobertura_beta_pre_relleno": float(df["beta"].notna().mean())}, "cfg": {}}
    series = {}

    def run(nm, h, oos, cfg):
        r, sr = eval_oos(oos, h, liq)
        series[nm] = sr
        log(f"n1b_{nm}", cfg, r, h)
        R["cfg"][nm] = {"h": h, **B.clean(r)}
        print(f"  {nm}: AUC={f(r['auc'])} IC={f(r['ic'])} t={f(r['t_nw'], 2)} aF={f(r.get('alpha_net_full_pct'), 3)} "
              f"aT={f(r.get('alpha_net_turn_pct'), 3)} turn={f(r.get('turnover_mean'), 2)} "
              f"foldsT+={r.get('folds_pos_turn')}/6 IC24+={f(r['por_anio']['2024+']['ic'])} "
              f"T3={f(r['por_tercil'].get('T3_liquido', {}).get('ic'))}", flush=True)
        json.dump(R, open(OUT / "n1b_target.json", "w"), indent=2, ensure_ascii=False, default=str)
        gc.collect()

    for h in HORIZONS:
        print(f"\n=== h={h} ===", flush=True)
        for tg in ("rank", "resid", "dm", "raw"):
            run(f"ridge_{tg}_h{h}", h, ridge_oos(df, X, f"t_{tg}{h}", h),
                {"model": "ridge", "alpha": RIDGE_ALPHA, "target": tg, "features": "ranks40"})
        run(f"ridge_rank_aug_h{h}", h, ridge_oos(df, X_aug, f"t_rank{h}", h),
            {"model": "ridge", "alpha": RIDGE_ALPHA, "target": "rank", "features": "ranks40+rev1d+rev1d*last5"})
        for tg, tc in (("dir", None), ("rank", f"t_rank{h}"), ("resid", f"t_resid{h}"), ("dm", f"t_dm{h}")):
            run(f"woe_{tg}_h{h}", h, woe_oos(df, cols, tc, h),
                {"model": "woe_logistic", "C": C_REG, "n_bins": N_BINS_WOE, "top": TOP_FEATURES, "target": tg})
        run(f"rev1d_h{h}", h, signal_oos(df, "rev1d", h), {"model": "signal", "feature": "-ret1d"})

    R["calendario"] = B.clean(calendar_study(df))
    for h in HORIZONS:
        c = R["calendario"][f"h{h}"]
        print(f"  calendario h={h}: last5={f(c['mean_last5_pct'], 3)} resto={f(c['mean_resto_pct'], 3)} dif={f(c['dif_pct'], 3)} "
              f"t={f(c['t_hac'], 2)} | entrada dtme5: {f(c['entrada_dtme5_mean_pct'], 3)} t={f(c['entrada_dtme5_t'], 2)}", flush=True)
        log_trial(f"n1b_calendario_h{h}", {"tanda": "n1b", "model": "calendar_last5", "horizon_days": h, "entry": "next_open"},
                  B.clean({"dif_pct": c["dif_pct"], "t_hac": c["t_hac"], "entrada_dtme5_t": c["entrada_dtme5_t"]}), len(df),
                  path=str(ROOT / TRIALS_PATH))

    n_glob, n_n1b = count_trials(), count_trials("n1b_")
    R["dsr_meta"] = {"n_trials_global": n_glob, "n_trials_n1b": n_n1b}
    for nm, v in series.items():
        per = max(1, int(round(252 / R["cfg"][nm]["h"])))
        R["cfg"][nm]["dsr_global"] = B.clean(deflated_sharpe(v, n_glob, periods=per))
        R["cfg"][nm]["dsr_n1b"] = B.clean(deflated_sharpe(v, n_n1b, periods=per))
    R["veredicto"] = veredicto(R)
    json.dump(R, open(OUT / "n1b_target.json", "w"), indent=2, ensure_ascii=False, default=str)
    write_md(R)
    print("Guardado n1b_target.json/.md", flush=True)


def veredicto(R: dict) -> dict:
    out = {}
    for nm, r in R["cfg"].items():
        base = R["cfg"].get(f"woe_dir_h{r['h']}", {})
        d_ic = (r["ic"] or 0) - (base.get("ic") or 0)
        dsr = (r.get("dsr_global") or {}).get("dsr")
        crit_a = bool((r["t_nw"] or 0) > 2 and (r.get("folds_pos_turn") or 0) >= 4 and (r.get("alpha_net_turn_pct") or -1) > 0)
        crit_b = bool(dsr is not None and dsr > 0.95)
        out[nm] = {"t_gt_2": bool((r["t_nw"] or 0) > 2), "alfa_turn_gt_0": bool((r.get("alpha_net_turn_pct") or -1) > 0),
                   "folds_turn_mayoria": bool((r.get("folds_pos_turn") or 0) >= 4), "dIC_vs_woe_dir": d_ic,
                   "dIC_gt_0.03": bool(d_ic > 0.03), "dsr_gt_0.95": crit_b, "criterio_A": crit_a, "pasa": bool(crit_a or crit_b)}
    return out


def write_md(R: dict) -> None:
    L = ["# N1-B cambio de target (cross-sectional) y features simples", "",
         f"- Convencion: {R['meta']['convencion']}; mismos 6 folds de test; purga h+1 dias + embargo 10; NW lag {R['meta']['nw_lag']}.",
         f"- Auditoria etiqueta (r5 vs ret_fwd del panel): {R['meta']['auditoria']}",
         f"- Ridge alpha={R['meta']['ridge_alpha']} sobre 40 features convertidas a percentil por fecha; objetivo winsorizado 1/99 en train.",
         "- Targets: rank = percentil por fecha del ret_h; resid = ret_h - beta(t-1, 252d)*ret_h EW; dm = ret_h - media expanding del ticker (rezagada h+1); raw = ret_h.",
         "- IC y alfa se miden SIEMPRE contra ret_h crudo. Alfa = largo tercil superior vs media del universo, rebalanceo cada h dias; "
         "`neto full` = 2*0.30% por rebalanceo (como la linea base), `neto turn` = 0.30%*2*turnover real.",
         f"- DSR (sobre el exceso neto turn por rebalanceo): N global de ensayos unicos en trial_log = {R['dsr_meta']['n_trials_global']}; N familia n1b = {R['dsr_meta']['n_trials_n1b']}.", ""]
    for h in HORIZONS:
        L += [f"## Horizonte {h} dias", "",
              "| config | AUC | IC | t NW | alfa full % | alfa turn % | turnover | folds+ full/turn | IC 2024+ | IC T3 | dIC vs woe_dir | DSR glob |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for nm, r in R["cfg"].items():
            if r["h"] != h:
                continue
            v = R["veredicto"][nm]
            L.append(f"| {nm} | {f(r['auc'])} | {f(r['ic'])} | {f(r['t_nw'], 2)} | {f(r.get('alpha_net_full_pct'), 3)} | "
                     f"{f(r.get('alpha_net_turn_pct'), 3)} | {f(r.get('turnover_mean'), 2)} | {r.get('folds_pos_full')}/{r.get('folds_pos_turn')} | "
                     f"{f(r['por_anio']['2024+']['ic'])} | {f(r['por_tercil'].get('T3_liquido', {}).get('ic'))} | {f(v['dIC_vs_woe_dir'])} | "
                     f"{f((r.get('dsr_global') or {}).get('dsr'), 3)} |")
        L.append("")
    L += ["## Por tercil de liquidez (IC / t / alfa neto turn %)", "", "| config | T1 | T2 | T3 |", "|---|---|---|---|"]
    for nm, r in R["cfg"].items():
        c = r["por_tercil"]
        L.append(f"| {nm} | " + " | ".join(f"{f(c[k]['ic'])} / {f(c[k]['t'], 2)} / {f(c[k]['alpha_net_turn_pct'], 2)}" if k in c else "-"
                                          for k in ("T1_iliquido", "T2", "T3_liquido")) + " |")
    L += ["", "## IC por anio (mejor IC de cada horizonte; entre parentesis alfa neto turn %)", ""]
    best = {}
    for nm, r in R["cfg"].items():
        if r["h"] not in best or (r["ic"] or -9) > (R["cfg"][best[r["h"]]]["ic"] or -9):
            best[r["h"]] = nm
    names = [best[h] for h in HORIZONS]
    yrs = sorted({y for n in names for y in R["cfg"][n]["por_anio"] if y != "2024+"}, key=int) + ["2024+"]
    L += ["| anio | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for y in yrs:
        L.append(f"| {y} | " + " | ".join(f(R["cfg"][n]["por_anio"].get(y, {}).get("ic")) + " (" +
                                            f(R["cfg"][n]["por_anio"].get(y, {}).get("alpha_net_turn_pct"), 2) + ")" for n in names) + " |")
    L += ["", "## Calendario (serie de mercado EW; last5 es igual para todas las emisoras: no hay IC cross-sectional)", "",
          "| h | ret last5 % | ret resto % | dif % | t HAC | entrada dtme=5 bruto % | t | neto % | meses |", "|---|---|---|---|---|---|---|---|---|"]
    for h in HORIZONS:
        c = R["calendario"][f"h{h}"]
        L.append(f"| {h} | {f(c['mean_last5_pct'], 3)} | {f(c['mean_resto_pct'], 3)} | {f(c['dif_pct'], 3)} | {f(c['t_hac'], 2)} | "
                 f"{f(c['entrada_dtme5_mean_pct'], 3)} | {f(c['entrada_dtme5_t'], 2)} | {f(c['entrada_dtme5_neto_pct'], 3)} | {c['n_meses']} |")
    L += ["", "## Veredicto (t>2 y alfa neto turn>0 en >=4/6 folds, o DSR>0.95; hallazgo si dIC vs woe_dir > 0.03)", ""]
    for nm, v in R["veredicto"].items():
        L.append(f"- {nm}: t>2={v['t_gt_2']}, alfa>0={v['alfa_turn_gt_0']}, folds mayoria={v['folds_turn_mayoria']}, "
                 f"dIC={f(v['dIC_vs_woe_dir'])}, DSR>0.95={v['dsr_gt_0.95']}, pasa={v['pasa']}")
    (OUT / "n1b_target.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
