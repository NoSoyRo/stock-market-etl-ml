"""
Nivel 0 / tanda B: persistencia de predecibilidad, cortes pre-registrados, adversarial validation
temporal y monotonia por decil de score. Reusa scripts/n0a_diagnostico.py (panel, folds, WoE+LR,
IC con Newey-West, alfa neto 0.30%/lado). No modifica codigo existente.

Ejecucion:  .venv/bin/python scripts/n0b_diagnostico.py [--cache RUTA.parquet] [--perm 300]
Memoria baja: float32, sin paralelismo, LightGBM chico (n_jobs=1).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import scripts.n0a_diagnostico as A
from scripts.run_purged_logistic import (C_REG, EMBARGO, HORIZON, LABEL_SPAN, N_BINS_WOE,
                                         N_SPLITS, TOP_FEATURES, transform_woe_features)
from src.data.store import get as store_get
from src.ml.features import FEATURE_COLS
from src.ml.purged_cv import purged_walk_forward_date_splits
from src.ml.trials import DEFAULT_PATH as TRIALS_PATH, log_trial
from src.ml.woe import compute_all_woe

OUT_DIR = ROOT / "data" / "eval"
SCRATCH = Path("/tmp/claude-1000/-home-gomiten-Documents-stock-market-etl-ml/6d78f6e6-2a0a-441a-9d97-5e424d20393a/scratchpad")
COST_SIDE = A.COST_SIDE
MIN_CELL = 4            # emisoras minimas por fecha dentro de una celda
N_CELLS_PREREG = 8      # 3 liquidez + 3 tamano (no hay dato) + 2 volatilidad  => Bonferroni conservador
ALPHA_BONF = 0.05 / N_CELLS_PREREG
TEST_START = pd.Timestamp("2024-01-01")
MIN_TY = 60             # filas minimas por emisora-ano


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def log(name, cfg, metrics, n):
    base = {"model": "woe_logistic", "n_splits": N_SPLITS, "embargo": EMBARGO, "horizon": HORIZON,
            "label_span": LABEL_SPAN, "cost_side_pct": COST_SIDE, "tanda": "n0b"}
    base.update(cfg)
    log_trial(name, base, clean(metrics), int(n), path=str(ROOT / TRIALS_PATH))


# ─── estadistica ──────────────────────────────────────────────────────────────

def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 10 or n0 < 10:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ic_by_date(d: pd.DataFrame, min_names: int) -> pd.Series:
    out = {}
    for dt, z in d.groupby("date"):
        if len(z) < min_names:
            continue
        a, b = z["score"].rank().values, z["ret_fwd"].rank().values
        if a.std() > 0 and b.std() > 0:
            out[dt] = np.corrcoef(a, b)[0, 1]
    return pd.Series(out, dtype=float)


def alpha_net(d: pd.DataFrame, min_names: int) -> np.ndarray:
    """Exceso neto por rebalanceo (cada 5 fechas): tercil superior de score vs media de la celda - 2*costo."""
    dates = np.sort(d["date"].unique())
    v = []
    for dt in dates[::HORIZON]:
        z = d[d["date"] == dt]
        if len(z) < min_names:
            continue
        r = z["score"].rank(pct=True)
        top = z.loc[r > 2 / 3, "ret_fwd"].mean()
        if np.isfinite(top):
            v.append(top - z["ret_fwd"].mean() - 2 * COST_SIDE)
    return np.array(v)


def p_from_t(t: float) -> float:
    return float(2 * (1 - stats.norm.cdf(abs(t)))) if np.isfinite(t) else float("nan")


def cell_block(d: pd.DataFrame) -> dict:
    ic = ic_by_date(d, MIN_CELL)
    m, s, t = A.nw_t(ic.values)
    al = alpha_net(d, MIN_CELL)
    return {"n": int(len(d)), "n_dates": int(len(ic)), "ic_mean": m, "ic_t_nw5": t, "p": p_from_t(t),
            "alpha_net_pct": float(al.mean()) if len(al) else float("nan"), "n_rebal": int(len(al)),
            "alpha_t": float(al.mean() / (al.std(ddof=1) / np.sqrt(len(al)))) if len(al) > 2 and al.std() > 0 else float("nan")}


# ─── datos de liquidez / mercado (point-in-time) ──────────────────────────────

def corwin_schultz(h: pd.Series, l: pd.Series) -> pd.Series:
    hl = np.log(h / l) ** 2
    beta = hl + hl.shift(1)
    hh, ll = h.rolling(2).max(), l.rolling(2).min()
    gamma = np.log(hh / ll) ** 2
    k = 3 - 2 * np.sqrt(2)
    al = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(al) - 1) / (1 + np.exp(al))
    return s.clip(lower=0)


def liquidity_panel(tickers, win: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    """Por (ticker, fecha): ADV, %dias sin operacion y spread CS, con datos hasta t-1 (shift 1).
    Devuelve tambien el retorno diario igual-ponderado de mercado."""
    rows, rets = [], {}
    for tk in tickers:
        raw = store_get(tk)
        if raw is None or len(raw) < win + 5:
            continue
        raw = raw[~raw.index.duplicated()].sort_index()
        c = raw["Close"].astype("float64")
        r = c.pct_change()
        rets[tk] = r
        dv = c * raw["Volume"].astype("float64")
        nop = ((raw["Volume"] == 0) | (r == 0)).astype(float)
        cs = corwin_schultz(raw["High"].astype("float64"), raw["Low"].astype("float64"))
        x = pd.DataFrame({
            "adv": dv.rolling(win, min_periods=30).mean().shift(1),
            "nop": nop.rolling(win, min_periods=30).mean().shift(1),
            "cs": cs.rolling(win, min_periods=30).mean().shift(1)})
        x["ticker"] = tk
        x.index.name = "date"
        rows.append(x.reset_index())
    liq = pd.concat(rows, ignore_index=True)
    liq["date"] = pd.to_datetime(liq["date"])
    for c in ("adv", "nop", "cs"):
        liq[c] = liq[c].astype("float32")
    mkt = pd.DataFrame(rets).mean(axis=1)
    mkt.index = pd.to_datetime(mkt.index)
    return liq, mkt


# ─── modelo con ponderacion (variante de A.score_oos) ─────────────────────────

def score_oos_w(df: pd.DataFrame, cols: list[str], w: np.ndarray | None) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"])
    parts = []
    for fi, (tr, te) in enumerate(purged_walk_forward_date_splits(
            dates, n_splits=N_SPLITS, horizon=LABEL_SPAN, embargo=EMBARGO)):
        Xtr, Xte = df.iloc[tr], df.iloc[te]
        ytr = Xtr["y"].to_numpy(int)
        if len(np.unique(ytr)) < 2:
            continue
        t = Xtr[cols].copy()
        t["y"] = ytr
        woe = compute_all_woe(t, feature_cols=cols, target_col="y", n_bins=N_BINS_WOE)
        sel = sorted([f for f, ww in woe.items() if ww.n_bins >= 2 and ww.iv > 0 and np.isfinite(ww.iv)],
                     key=lambda f: woe[f].iv, reverse=True)[:TOP_FEATURES]
        if not sel:
            continue
        lr = LogisticRegression(C=C_REG, max_iter=1000, random_state=42).fit(
            transform_woe_features(Xtr, sel, woe), ytr,
            sample_weight=None if w is None else w[tr])
        p = lr.predict_proba(transform_woe_features(Xte, sel, woe))[:, 1]
        parts.append(pd.DataFrame({
            "date": Xte["date"].values, "ticker": Xte["ticker"].values, "y": Xte["y"].values,
            "ret_fwd": Xte["ret_fwd"].values.astype("float64"), "score": p, "fold": fi + 1}))
        del t, Xtr, Xte
        gc.collect()
    return pd.concat(parts, ignore_index=True)


def eval_short(oos: pd.DataFrame) -> dict:
    g = A.evaluate(oos)["global"]
    a = g.get("alpha", {})
    return {"n": int(len(oos)), "auc": g["auc_pooled"], "ic": g["ic_mean"], "t_nw5": g["ic_t_nw5"],
            "auc_date_median": g["auc_date_median"], "auc_ticker": g["auc_ticker"],
            "alpha_long_net_pct": a.get("long_excess_net_mean_pct"), "ls_net_pct": a.get("ls_net_mean_pct")}


# ─── 1. persistencia por emisora y ano ────────────────────────────────────────

def part1(oos: pd.DataFrame, liq: pd.DataFrame, n_perm: int, rng) -> dict:
    d = oos.copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year
    d = d.merge(liq, on=["ticker", "date"], how="left")
    recs = []
    for (tk, yr), z in d.groupby(["ticker", "year"]):
        if len(z) < MIN_TY:
            continue
        z = z.sort_values("date")
        y = z["y"].to_numpy(int)
        auc = fast_auc(y, z["score"].values)
        ic = stats.spearmanr(z["score"], z["ret_fwd"])[0]
        recs.append({"ticker": tk, "year": int(yr), "n": len(z), "auc": auc, "ic": float(ic),
                     "adv": float(np.log1p(z["adv"].mean())), "nop": float(z["nop"].mean()),
                     "cs": float(z["cs"].mean())})
    ty = pd.DataFrame(recs).dropna(subset=["auc"])
    out = {"n_ticker_year": int(len(ty)), "tickers": int(ty["ticker"].nunique()),
           "auc_media": float(ty["auc"].mean()), "auc_std_entre_celdas": float(ty["auc"].std()),
           "ic_media": float(ty["ic"].mean()), "ic_std_entre_celdas": float(ty["ic"].std())}

    def persistence(tab: pd.DataFrame, col: str) -> tuple[float, list]:
        cors, w = [], []
        for yr in sorted(tab["year"].unique()):
            a = tab[tab["year"] == yr].set_index("ticker")[col]
            b = tab[tab["year"] == yr + 1].set_index("ticker")[col]
            j = a.index.intersection(b.index)
            if len(j) >= 8:
                cors.append((yr, float(stats.spearmanr(a[j], b[j])[0]), len(j)))
        return (float(np.mean([c[1] for c in cors])) if cors else float("nan")), cors

    for col in ("auc", "ic"):
        m, per = persistence(ty, col)
        # permutacion 1: barajar emisoras del ano t+1 (baseline sin persistencia)
        piv = {yr: ty[ty["year"] == yr].set_index("ticker")[col] for yr in ty["year"].unique()}
        perm = []
        for _ in range(2000):
            cs = []
            for yr, _c, _n in per:
                a, b = piv[yr], piv[yr + 1]
                j = a.index.intersection(b.index)
                cs.append(stats.spearmanr(a[j].values, rng.permutation(b[j].values))[0])
            perm.append(np.mean(cs))
        perm = np.array(perm)
        # permutacion 2: desplazamiento circular del score contra la etiqueta dentro de emisora-ano
        # (preserva autocorrelacion, rompe la relacion score-retorno): ruido de estimacion puro
        sh_stats = []
        grp = [(tk, yr, z.sort_values("date")) for (tk, yr), z in d.groupby(["ticker", "year"]) if len(z) >= MIN_TY]
        pre = []
        for tk, yr, z in grp:
            pre.append((tk, yr, z["y"].to_numpy(int), z["score"].values, stats.rankdata(z["ret_fwd"].values)))
        for _ in range(n_perm):
            rec = []
            for tk, yr, y, s, rr in pre:
                k = int(rng.integers(HORIZON * 2, len(y) - HORIZON * 2))
                s2 = np.roll(s, k)
                if col == "auc":
                    v = fast_auc(y, s2)
                else:
                    v = float(np.corrcoef(stats.rankdata(s2), rr)[0, 1])
                rec.append((tk, yr, v))
            t2 = pd.DataFrame(rec, columns=["ticker", "year", col]).dropna()
            sh_stats.append(persistence(t2, col)[0])
        sh = np.array(sh_stats)
        sp_all = stats.spearmanr(*_pairs(ty, col)) if per else (float("nan"), float("nan"))
        out[f"persist_{col}"] = {
            "corr_media_por_par_de_anos": m, "pares": [{"ano_t": a, "rho": r, "n": n} for a, r, n in per],
            "rho_agrupado_zscore": float(sp_all[0]), "p_spearman_agrupado": float(sp_all[1]),
            "p_perm_emisoras": float((np.sum(perm >= m) + 1) / (len(perm) + 1)),
            "perm_emisoras_media": float(perm.mean()), "perm_emisoras_p95": float(np.percentile(perm, 95)),
            "p_perm_desplazamiento": float((np.sum(sh >= m) + 1) / (len(sh) + 1)),
            "perm_desplazamiento_media": float(sh.mean()), "perm_desplazamiento_p95": float(np.percentile(sh, 95)),
            "n_perm_desplazamiento": int(len(sh))}
    # predecibilidad ~ liquidez (se calcula siempre, se interpreta solo si hay persistencia)
    out["predecibilidad_vs_liquidez"] = {}
    for col in ("auc", "ic"):
        for liqc in ("adv", "nop", "cs"):
            z = ty.dropna(subset=[liqc])
            r, p = stats.spearmanr(z[col], z[liqc])
            out["predecibilidad_vs_liquidez"][f"{col}~{liqc}"] = {"rho": float(r), "p": float(p), "n": int(len(z))}
        # nivel por emisora (promedio de los anos) vs liquidez promedio
        g = ty.groupby("ticker")[[col, "adv", "nop", "cs"]].mean()
        for liqc in ("adv", "nop", "cs"):
            r, p = stats.spearmanr(g[col], g[liqc])
            out["predecibilidad_vs_liquidez"][f"{col}_por_emisora~{liqc}"] = {"rho": float(r), "p": float(p), "n": int(len(g))}
    # AUC por ano (todas las emisoras) para ver estabilidad temporal
    out["por_ano"] = {int(y): {"auc_media": float(g["auc"].mean()), "ic_media": float(g["ic"].mean()), "n_emisoras": int(len(g))}
                      for y, g in ty.groupby("year")}
    out["ty_table_head"] = ty.sort_values("auc", ascending=False).head(5).round(4).to_dict("records")
    return out


def _pairs(ty: pd.DataFrame, col: str):
    """Pares (valor_t, valor_t+1) agrupados, con z-score dentro de cada ano para quitar el efecto ano."""
    z = ty.copy()
    z[col] = z.groupby("year")[col].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
    xs, ys = [], []
    for yr in sorted(z["year"].unique()):
        a = z[z["year"] == yr].set_index("ticker")[col]
        b = z[z["year"] == yr + 1].set_index("ticker")[col]
        j = a.index.intersection(b.index)
        xs += list(a[j]); ys += list(b[j])
    return np.array(xs), np.array(ys)


# ─── 2. cortes pre-registrados ────────────────────────────────────────────────

def part2(oos: pd.DataFrame, liq: pd.DataFrame, mkt: pd.Series) -> dict:
    d = oos.merge(liq, on=["ticker", "date"], how="left")
    d["date"] = pd.to_datetime(d["date"])
    rv = mkt.rolling(20, min_periods=15).std().shift(1)
    med = rv.expanding(min_periods=250).median().shift(1)
    reg = pd.Series(np.where(rv > med, "alta", "baja"), index=rv.index).where(med.notna() & rv.notna())
    d["regimen"] = d["date"].map(reg)
    ranks = {}
    for nm, col, asc in (("liq_adv", "adv", True), ("liq_sinop", "nop", False), ("liq_cs", "cs", False)):
        # T3 = mas liquido: ADV alto, % sin operacion bajo, spread bajo
        r = d.groupby("date")[col].rank(pct=True)
        if not asc:
            r = 1 - r + 1e-9
        d[nm] = pd.cut(r, [0, 1 / 3, 2 / 3, 1.0001], labels=["T1_iliquido", "T2", "T3_liquido"])
    cells = {}
    axes = {"liquidez_ADV": ("liq_adv", ["T1_iliquido", "T2", "T3_liquido"]),
            "volatilidad_mercado": ("regimen", ["alta", "baja"])}
    for ax, (col, labs) in axes.items():
        for lab in labs:
            z = d[d[col] == lab]
            if len(z) == 0:
                continue
            name = f"{ax}:{lab}"
            r = cell_block(z)
            folds = {}
            for f, zf in z.groupby("fold"):
                folds[int(f)] = cell_block(zf)
            ic_pos = sum(1 for b in folds.values() if np.isfinite(b["ic_mean"]) and b["ic_mean"] > 0)
            al_pos = sum(1 for b in folds.values() if np.isfinite(b["alpha_net_pct"]) and b["alpha_net_pct"] > 0)
            r.update({"folds_ic_pos": ic_pos, "folds_alpha_pos": al_pos, "n_folds": len(folds),
                      "fold_ic": {k: b["ic_mean"] for k, b in folds.items()},
                      "fold_alpha": {k: b["alpha_net_pct"] for k, b in folds.items()}})
            r["pasa_bonferroni"] = bool(np.isfinite(r["p"]) and r["p"] < ALPHA_BONF and r["ic_mean"] > 0)
            r["replica_4de6"] = bool(ic_pos >= 4)
            r["alfa_neto_pos"] = bool(np.isfinite(r["alpha_net_pct"]) and r["alpha_net_pct"] > 0)
            r["cuenta"] = bool(r["pasa_bonferroni"] and r["replica_4de6"] and r["alfa_neto_pos"] and al_pos >= 4)
            cells[name] = r
            log(f"n0b_cortes:{name}", {"corte": name, "min_cell": MIN_CELL, "bonferroni_m": N_CELLS_PREREG},
                {k: r[k] for k in ("ic_mean", "ic_t_nw5", "p", "alpha_net_pct", "folds_ic_pos", "folds_alpha_pos", "cuenta")}, r["n"])
    # complementos informativos (NO cuentan en el conteo pre-registrado)
    comp = {}
    for nm, lab in (("liq_sinop", "T3_liquido"), ("liq_cs", "T3_liquido"), ("liq_sinop", "T1_iliquido"), ("liq_cs", "T1_iliquido")):
        comp[f"{nm}:{lab}"] = cell_block(d[d[nm] == lab])
    return {"celdas": cells, "complementarios_no_registrados": comp,
            "bonferroni_alpha": ALPHA_BONF, "n_celdas_preregistradas": N_CELLS_PREREG,
            "tamano": "SIN DATO: no hay capitalizacion ni acciones en circulacion en el repo (fundamentals.parquet solo tiene margenes); eje omitido, Bonferroni se mantiene con m=8",
            "regimen_def": "alta si vol realizada 20d del mercado igual-ponderado (hasta t-1) > mediana expansiva (hasta t-2); requiere 250 obs",
            "liquidez_def": "tercil por fecha del ADV 60d en pesos (Close*Volume, shift 1); T3 = mas liquido"}


# ─── 3. adversarial validation ────────────────────────────────────────────────

def adv_fit(df: pd.DataFrame, cols: list[str], seed: int = 0) -> tuple[float, np.ndarray, pd.Series]:
    import lightgbm as lgb
    X = df[cols].to_numpy(np.float32)
    lab = (df["date"] >= TEST_START).to_numpy(int)
    blk = (df["date"].dt.year * 100 + df["date"].dt.month).to_numpy()
    ub = np.unique(blk)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(ub, rng.integers(0, 5, len(ub))))
    f = np.array([fold_of[b] for b in blk])
    prm = dict(n_estimators=60, num_leaves=15, learning_rate=0.1, min_child_samples=100, subsample=0.7,
               subsample_freq=1, colsample_bytree=0.8, n_jobs=1, verbose=-1, random_state=seed, is_unbalance=True)
    oof = np.zeros(len(df), dtype=np.float32)
    for k in range(5):
        m = lgb.LGBMClassifier(**prm).fit(X[f != k], lab[f != k])
        oof[f == k] = m.predict_proba(X[f == k])[:, 1]
    auc = float(roc_auc_score(lab, oof))
    m = lgb.LGBMClassifier(**prm).fit(X, lab)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=cols)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    return auc, oof, imp


def part3(df: pd.DataFrame, cols: list[str]) -> dict:
    out = {"n_train_2008_2023": int((df["date"] < TEST_START).sum()), "n_test_2024_26": int((df["date"] >= TEST_START).sum())}
    auc, oof, imp = adv_fit(df, cols)
    out["auc_adversarial"] = auc
    out["importancia_top10"] = {k: float(v) for k, v in imp.head(10).items()}
    print(f"  AUC adversarial={auc:.4f}; top: {list(imp.head(5).index)}", flush=True)
    log("n0b_adversarial:clasificador", {"lgbm": "60 arboles, 15 hojas", "n_features": len(cols)},
        {"auc_adv": auc, "top5": list(imp.head(5).index)}, len(df))
    ref_oos = A.score_oos(df, cols)
    out["referencia_todas_features"] = eval_short(ref_oos)
    out["referencia_solo_2024+"] = eval_short(ref_oos[ref_oos["date"] >= TEST_START])
    out["variantes"] = {}
    del ref_oos
    gc.collect()
    # quitar top-k features que derivan
    for k in (1, 3, 5):
        drop = list(imp.index[:k])
        keep = [c for c in cols if c not in drop]
        auc_k, _, imp_k = adv_fit(df, keep)
        oos = A.score_oos(df, keep)
        r = {"quitadas": drop, "auc_adversarial_tras_quitar": auc_k, "global": eval_short(oos),
             "solo_2024+": eval_short(oos[oos["date"] >= TEST_START])}
        out["variantes"][f"drop_top{k}"] = r
        print(f"  drop_top{k}: advAUC={auc_k:.4f} AUC={r['global']['auc']:.4f} IC={r['global']['ic']:.4f}", flush=True)
        log(f"n0b_adversarial:drop_top{k}", {"drop": drop, "n_features": len(keep)},
            {"auc_adv_after": auc_k, **{f"g_{a}": b for a, b in r["global"].items()},
             **{f"t24_{a}": b for a, b in r["solo_2024+"].items()}}, len(oos))
        del oos
        gc.collect()
    # importance weighting: w = p/(1-p) del clasificador adversarial OOF, recortado y normalizado
    p = np.clip(oof.astype("float64"), 0.02, 0.98)
    # is_unbalance solo reescala las odds por una constante; al normalizar a media 1 no importa
    odds = p / (1 - p)
    w = np.clip(odds / np.median(odds), 0.1, 10.0)
    w = (w / w.mean()).astype("float64")
    out["pesos"] = {"min": float(w.min()), "p50": float(np.median(w)), "p95": float(np.percentile(w, 95)), "max": float(w.max()),
                    "ess_frac": float(w.sum() ** 2 / (w ** 2).sum() / len(w))}
    oos = score_oos_w(df, cols, w)
    r = {"global": eval_short(oos), "solo_2024+": eval_short(oos[oos["date"] >= TEST_START])}
    out["variantes"]["importance_weighting"] = r
    print(f"  IW: AUC={r['global']['auc']:.4f} IC={r['global']['ic']:.4f}", flush=True)
    log("n0b_adversarial:importance_weighting", {"clip": [0.1, 10.0], "w": "odds adversarial OOF"},
        {**{f"g_{a}": b for a, b in r["global"].items()}, **{f"t24_{a}": b for a, b in r["solo_2024+"].items()}}, len(oos))
    return out


# ─── 4. monotonia por decil ───────────────────────────────────────────────────

def part4(oos: pd.DataFrame) -> dict:
    d = oos.copy()
    d["date"] = pd.to_datetime(d["date"])
    n = d.groupby("date")["score"].transform("size")
    d = d[n >= A.MIN_NAMES].copy()
    d["exc"] = d["ret_fwd"] - d.groupby("date")["ret_fwd"].transform("mean")
    res = {}

    def summarize(dec_col: str, tag: str):
        g = d.groupby(dec_col)
        tab = pd.DataFrame({"n": g.size(), "exc_ret_pct": g["exc"].mean(), "ret_pct": g["ret_fwd"].mean(),
                            "hit_y": g["y"].mean(), "score_medio": g["score"].mean()})
        tab["ic_dentro"] = [float(stats.spearmanr(z["score"], z["ret_fwd"])[0]) if len(z) > 30 else np.nan for _, z in g]
        mono = stats.spearmanr(tab.index.values, tab["exc_ret_pct"].values)
        # d10 - d9 por fecha
        pv = d.pivot_table(index="date", columns=dec_col, values="exc", aggfunc="mean")
        diff = (pv[10] - pv[9]).dropna() if 10 in pv.columns and 9 in pv.columns else pd.Series(dtype=float)
        m, s, t = A.nw_t(diff.values)
        # por fold (d10 vs d9)
        fd = {}
        for f, z in d.groupby("fold"):
            a = z.loc[z[dec_col] == 10, "exc"].mean()
            b = z.loc[z[dec_col] == 9, "exc"].mean()
            fd[int(f)] = float(a - b)
        res[tag] = {"tabla": {int(k): {c: (None if not np.isfinite(v) else float(v)) for c, v in row.items()}
                              for k, row in tab.iterrows()},
                    "monotonia_spearman_decil_vs_exceso": {"rho": float(mono[0]), "p": float(mono[1])},
                    "d10_menos_d9": {"media_pct": m, "t_nw5": t, "n_fechas": int(len(diff)), "por_fold": fd,
                                     "folds_d10_gt_d9": int(sum(v > 0 for v in fd.values()))},
                    "d10_menos_d1_pct": float(tab["exc_ret_pct"].iloc[-1] - tab["exc_ret_pct"].iloc[0])}
        log(f"n0b_deciles:{tag}", {"decil": tag},
            {"mono_rho": float(mono[0]), "d10_d9_pct": m, "d10_d9_t": t, "d10_d1_pct": res[tag]["d10_menos_d1_pct"]}, len(d))

    # (a) decil cross-seccional por fecha
    r = d.groupby("date")["score"].rank(pct=True)
    d["dec_cs"] = np.minimum((r * 10).apply(np.ceil), 10).astype(int)
    summarize("dec_cs", "decil_seccional_por_fecha")
    # (b) decil del score dentro de cada fold (calibracion absoluta del tramo alto)
    d["dec_fold"] = d.groupby("fold")["score"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1).astype(int)
    summarize("dec_fold", "decil_pooled_por_fold")
    return res


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(SCRATCH / "panel.parquet"))
    ap.add_argument("--perm", type=int, default=300)
    ap.add_argument("--skip", default="", help="partes a omitir, ej: 3")
    args = ap.parse_args()
    skip = set(args.skip.split(",")) if args.skip else set()
    rng = np.random.default_rng(7)

    df = A.load_panel(Path(args.cache))
    cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df[np.isfinite(df["ret_fwd"])].reset_index(drop=True)
    print(f"Panel: {len(df):,} filas, {df['ticker'].nunique()} tickers, {len(cols)} features", flush=True)

    R = {"panel": {"filas": int(len(df)), "tickers": int(df["ticker"].nunique()), "n_features": len(cols)}}
    oos = A.score_oos(df, cols)
    R["base"] = eval_short(oos)
    print("Base:", {k: round(v, 4) if isinstance(v, float) else v for k, v in R["base"].items()}, flush=True)
    log("n0b_base:woe_logistic_base", {"cleaning": "base"}, R["base"], len(oos))
    oos.to_parquet(SCRATCH / "n0b_oos.parquet")

    liq, mkt = liquidity_panel(sorted(df["ticker"].unique()))
    R["cobertura_liquidez"] = {"adv_no_nulo": float(oos.merge(liq, on=["ticker", "date"], how="left")["adv"].notna().mean())}

    print("\n[1] persistencia", flush=True)
    R["p1_persistencia"] = part1(oos, liq, args.perm, rng)
    for col in ("auc", "ic"):
        p = R["p1_persistencia"][f"persist_{col}"]
        print(f"  {col}: rho_medio={p['corr_media_por_par_de_anos']:.3f} p_perm_emis={p['p_perm_emisoras']:.3f} "
              f"p_perm_desp={p['p_perm_desplazamiento']:.3f} rho_agr={p['rho_agrupado_zscore']:.3f} p={p['p_spearman_agrupado']:.3f}", flush=True)
        log(f"n0b_persistencia:{col}", {"min_ty": MIN_TY, "n_perm": args.perm},
            {k: v for k, v in p.items() if not isinstance(v, (list, dict))}, R["p1_persistencia"]["n_ticker_year"])

    print("\n[2] cortes", flush=True)
    R["p2_cortes"] = part2(oos, liq, mkt)
    for k, c in R["p2_cortes"]["celdas"].items():
        print(f"  {k}: IC={c['ic_mean']:.4f} t={c['ic_t_nw5']:.2f} p={c['p']:.4f} alfa={c['alpha_net_pct']:.3f}% "
              f"folds_ic+={c['folds_ic_pos']}/{c['n_folds']} folds_alfa+={c['folds_alpha_pos']} cuenta={c['cuenta']}", flush=True)

    print("\n[4] deciles", flush=True)
    R["p4_deciles"] = part4(oos)
    for k, v in R["p4_deciles"].items():
        print(f"  {k}: mono rho={v['monotonia_spearman_decil_vs_exceso']['rho']:.2f} d10-d9={v['d10_menos_d9']['media_pct']:.3f}% "
              f"t={v['d10_menos_d9']['t_nw5']:.2f} folds d10>d9={v['d10_menos_d9']['folds_d10_gt_d9']}", flush=True)
    del oos, liq
    gc.collect()

    if "3" not in skip:
        print("\n[3] adversarial", flush=True)
        R["p3_adversarial"] = part3(df, cols)

    R["trials_n0b"] = None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "n0b_diagnostico.json").write_text(json.dumps(clean(R), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_md(clean(R))
    print("Guardado data/eval/n0b_diagnostico.json y .md")


def f(x, nd=4):
    return "nan" if x is None else f"{x:.{nd}f}"


def write_md(R: dict) -> None:
    L = ["# N0-B diagnostico (WoE+Logistic, 6 folds)", "",
         f"Base OOS: AUC {f(R['base']['auc'])}, IC {f(R['base']['ic'])}, t NW {f(R['base']['t_nw5'], 2)}, "
         f"alfa largo neto {f(R['base']['alpha_long_net_pct'], 3)}%", ""]
    p1 = R["p1_persistencia"]
    L += ["## 1. Persistencia por emisora-ano", "",
          f"{p1['n_ticker_year']} celdas emisora-ano ({p1['tickers']} emisoras). AUC medio {f(p1['auc_media'])} "
          f"(sd entre celdas {f(p1['auc_std_entre_celdas'])}); IC medio {f(p1['ic_media'])}.", "",
          "| metrica | rho medio t->t+1 | p perm. emisoras | p perm. desplaz. (ruido) | rho agrupado | p |", "|---|---|---|---|---|---|"]
    for c in ("auc", "ic"):
        p = p1[f"persist_{c}"]
        L.append(f"| {c} | {f(p['corr_media_por_par_de_anos'], 3)} | {f(p['p_perm_emisoras'], 3)} | "
                 f"{f(p['p_perm_desplazamiento'], 3)} | {f(p['rho_agrupado_zscore'], 3)} | {f(p['p_spearman_agrupado'], 3)} |")
    L += ["", "Predecibilidad vs liquidez (Spearman):", "", "| par | rho | p | n |", "|---|---|---|---|"]
    for k, v in p1["predecibilidad_vs_liquidez"].items():
        L.append(f"| {k} | {f(v['rho'], 3)} | {f(v['p'], 3)} | {v['n']} |")
    L += ["", "Por ano: " + "; ".join(f"{y}: AUC {f(v['auc_media'], 3)} IC {f(v['ic_media'], 3)}" for y, v in p1["por_ano"].items()), ""]
    p2 = R["p2_cortes"]
    L += ["## 2. Cortes pre-registrados", "", f"Bonferroni alfa = {p2['bonferroni_alpha']:.4f} (m={p2['n_celdas_preregistradas']}). Tamano: {p2['tamano']}", "",
          "| celda | n | IC | t NW | p | alfa neto % | folds IC+ | folds alfa+ | cuenta |", "|---|---|---|---|---|---|---|---|---|"]
    for k, c in p2["celdas"].items():
        L.append(f"| {k} | {c['n']} | {f(c['ic_mean'])} | {f(c['ic_t_nw5'], 2)} | {f(c['p'], 4)} | {f(c['alpha_net_pct'], 3)} | "
                 f"{c['folds_ic_pos']}/{c['n_folds']} | {c['folds_alpha_pos']}/{c['n_folds']} | {c['cuenta']} |")
    L += ["", "Complementarios (no cuentan): " + "; ".join(f"{k}: IC {f(v['ic_mean'])} t {f(v['ic_t_nw5'], 2)} alfa {f(v['alpha_net_pct'], 3)}%"
                                                      for k, v in p2["complementarios_no_registrados"].items()), ""]
    if "p3_adversarial" in R:
        p3 = R["p3_adversarial"]
        L += ["## 3. Adversarial validation (2008-2023 vs 2024-2026)", "",
              f"AUC adversarial {f(p3['auc_adversarial'])}. Top features (ganancia): " +
              ", ".join(f"{k} {v:.2f}" for k, v in list(p3["importancia_top10"].items())[:6]), "",
              "| variante | AUC | IC | t NW | alfa largo neto % | AUC 2024+ | IC 2024+ | t 2024+ |", "|---|---|---|---|---|---|---|---|"]
        rf, r24 = p3["referencia_todas_features"], p3["referencia_solo_2024+"]
        L.append(f"| todas | {f(rf['auc'])} | {f(rf['ic'])} | {f(rf['t_nw5'], 2)} | {f(rf['alpha_long_net_pct'], 3)} | {f(r24['auc'])} | {f(r24['ic'])} | {f(r24['t_nw5'], 2)} |")
        for k, v in p3["variantes"].items():
            g, t = v["global"], v["solo_2024+"]
            extra = f" (adv {f(v['auc_adversarial_tras_quitar'], 3)})" if "auc_adversarial_tras_quitar" in v else ""
            L.append(f"| {k}{extra} | {f(g['auc'])} | {f(g['ic'])} | {f(g['t_nw5'], 2)} | {f(g['alpha_long_net_pct'], 3)} | {f(t['auc'])} | {f(t['ic'])} | {f(t['t_nw5'], 2)} |")
        L += ["", "Quitadas: " + "; ".join(f"{k}: {v['quitadas']}" for k, v in p3["variantes"].items() if "quitadas" in v), ""]
    L += ["## 4. Monotonia por decil de score", ""]
    for tag, v in R["p4_deciles"].items():
        L += [f"### {tag}", "", "| decil | n | exceso % | ret % | hit | IC dentro |", "|---|---|---|---|---|---|"]
        for k, r in v["tabla"].items():
            L.append(f"| {k} | {r['n']:.0f} | {f(r['exc_ret_pct'], 3)} | {f(r['ret_pct'], 3)} | {f(r['hit_y'], 3)} | {f(r['ic_dentro'], 3)} |")
        m = v["monotonia_spearman_decil_vs_exceso"]
        d = v["d10_menos_d9"]
        L += ["", f"Monotonia rho {f(m['rho'], 2)} (p {f(m['p'], 3)}); d10-d9 = {f(d['media_pct'], 3)}% (t NW {f(d['t_nw5'], 2)}, "
                  f"d10>d9 en {d['folds_d10_gt_d9']}/6 folds); d10-d1 = {f(v['d10_menos_d1_pct'], 3)}%", ""]
    (OUT_DIR / "n0b_diagnostico.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
