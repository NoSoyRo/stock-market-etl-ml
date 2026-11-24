"""
Nivel 0 / tanda A: diagnostico del baseline WoE+Logistic (AUC ~0.52).

Reusa build_dataset, purged_walk_forward_date_splits (6 folds por fechas, embargo 10,
label_span 6) y el modelo WoE+LR de run_purged_logistic. No modifica codigo existente.

Para cada configuracion de limpieza (base, A, B, C) se reentrena el modelo por fold
(train y test filtrados igual) y se reporta: AUC pooled, rank IC por fecha (Newey-West),
AUC demeaned por fecha y por ticker, y alfa neto de costos (0.30%/lado).

Ejecucion:  .venv/bin/python scripts/n0a_diagnostico.py [--cache RUTA.parquet]
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

from scripts.run_purged_cv import build_dataset
from scripts.run_purged_logistic import (
    C_REG, EMBARGO, HORIZON, LABEL_SPAN, N_BINS_WOE, N_SPLITS, TOP_FEATURES,
    transform_woe_features,
)
from src.data.store import get as store_get
from src.ml.features import FEATURE_COLS
from src.ml.power import required_weeks
from src.ml.purged_cv import purged_walk_forward_date_splits
from src.ml.trials import DEFAULT_PATH as TRIALS_PATH, log_trial
from src.ml.woe import compute_all_woe

MIN_NAMES = 8          # minimo de emisoras por fecha para IC / demeaning cross-sectional
NW_LAG = 5             # horizonte 5d => retornos solapados => lag >= 5
COST_SIDE = 0.30       # % por lado
WINSOR = 25.0          # % sobre ret_fwd
ZERO_FRAC_MAX = 0.05
OUT_DIR = ROOT / "data" / "eval"


# ─── utilidades estadisticas ──────────────────────────────────────────────────

def nw_t(x: np.ndarray, lag: int = NW_LAG) -> tuple[float, float, float]:
    """media, std, t-stat Newey-West (Bartlett) de la media de x."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lag + 3:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    e = x - m
    s = e @ e / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    se = np.sqrt(max(s, 1e-18) / n)
    return float(m), float(x.std(ddof=1)), float(m / se)


def daily_rank_ic(d: pd.DataFrame) -> pd.Series:
    """Spearman por fecha entre score y ret_fwd (solo fechas con >= MIN_NAMES)."""
    g = d.groupby("date", sort=True)
    n = g["score"].transform("size")
    d = d[n >= MIN_NAMES]
    r_s = d.groupby("date")["score"].rank()
    r_r = d.groupby("date")["ret_fwd"].rank()
    tmp = pd.DataFrame({"date": d["date"].values, "a": r_s.values, "b": r_r.values})
    out = tmp.groupby("date").apply(lambda z: np.corrcoef(z["a"], z["b"])[0, 1] if z["a"].std() > 0 and z["b"].std() > 0 else np.nan)
    return out.dropna()


def safe_auc(y, s) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def demeaned_auc(d: pd.DataFrame) -> dict:
    """AUC con etiqueta/score relativos (a) a la mediana de cada fecha, (b) a cada ticker."""
    g = d.groupby("date")
    n = g["score"].transform("size")
    dd = d[n >= MIN_NAMES]
    ymed = dd["ret_fwd"] - dd.groupby("date")["ret_fwd"].transform("median")
    smed = dd["score"] - dd.groupby("date")["score"].transform("median")
    keep = ymed != 0
    auc_date = safe_auc((ymed[keep] > 0).astype(int), smed[keep])
    # terciles cross-sectional: top vs bottom tercil de retorno por fecha, score relativo
    rk = dd.groupby("date")["ret_fwd"].rank(pct=True)
    mask = (rk <= 1 / 3) | (rk > 2 / 3)
    auc_terc = safe_auc((rk[mask] > 2 / 3).astype(int), smed[mask])
    # por ticker: etiqueta = ret > mediana del ticker en el fold; score - media del score del ticker
    yt = d["ret_fwd"] - d.groupby("ticker")["ret_fwd"].transform("median")
    st = d["score"] - d.groupby("ticker")["score"].transform("mean")
    k2 = yt != 0
    auc_tk = safe_auc((yt[k2] > 0).astype(int), st[k2])
    return {"auc_date_median": auc_date, "auc_date_terciles": auc_terc, "auc_ticker": auc_tk}


def net_alpha(d: pd.DataFrame) -> dict:
    """Rebalanceo no solapado cada 5 fechas. Largo tercil superior vs promedio igual-ponderado.
    alfa_neto = ret(top tercil) - ret(universo) - costo ida y vuelta (2*0.30%).
    Tambien spread L/S (top - bottom) neto de 4 patas * 0.30%."""
    dates = np.sort(d["date"].unique())
    rows = []
    for dt in dates[::HORIZON]:
        z = d[d["date"] == dt]
        if len(z) < MIN_NAMES:
            continue
        r = z["score"].rank(pct=True)
        top = z.loc[r > 2 / 3, "ret_fwd"].mean()
        bot = z.loc[r <= 1 / 3, "ret_fwd"].mean()
        rows.append((top - z["ret_fwd"].mean(), top - bot))
    if not rows:
        return {}
    a = np.array(rows)
    long_net = a[:, 0] - 2 * COST_SIDE
    ls_net = a[:, 1] - 4 * COST_SIDE
    out = {"n_rebal": len(a)}
    for nm, v in (("long_excess_gross", a[:, 0]), ("long_excess_net", long_net),
                  ("ls_gross", a[:, 1]), ("ls_net", ls_net)):
        out[nm + "_mean_pct"] = float(v.mean())
        out[nm + "_t"] = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 2 and v.std() > 0 else float("nan")
    return out


# ─── modelo por fold ──────────────────────────────────────────────────────────

def score_oos(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Devuelve DataFrame OOS (date, ticker, y, ret_fwd, score, fold) con los mismos splits."""
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
        sel = sorted([f for f, w in woe.items() if w.n_bins >= 2 and w.iv > 0 and np.isfinite(w.iv)],
                     key=lambda f: woe[f].iv, reverse=True)[:TOP_FEATURES]
        if not sel:
            continue
        lr = LogisticRegression(C=C_REG, max_iter=1000, random_state=42).fit(
            transform_woe_features(Xtr, sel, woe), ytr)
        p = lr.predict_proba(transform_woe_features(Xte, sel, woe))[:, 1]
        parts.append(pd.DataFrame({
            "date": Xte["date"].values, "ticker": Xte["ticker"].values,
            "y": Xte["y"].values, "ret_fwd": Xte["ret_fwd"].values.astype("float64"),
            "score": p, "fold": fi + 1}))
        del t, Xtr, Xte
        gc.collect()
    return pd.concat(parts, ignore_index=True)


def evaluate(oos: pd.DataFrame) -> dict:
    res = {"n_oos": int(len(oos)), "n_tickers_oos": int(oos["ticker"].nunique()),
           "folds": {}, "global": {}}

    def block(d: pd.DataFrame) -> dict:
        ic = daily_rank_ic(d)
        m, s, t = nw_t(ic.values)
        _, _, t10 = nw_t(ic.values, 10)
        b = {"n": int(len(d)), "n_dates": int(len(ic)), "auc_pooled": safe_auc(d["y"], d["score"]),
             "ic_mean": m, "ic_std": s, "ic_t_nw5": t, "ic_t_nw10": t10,
             "pct_ic_pos": float((ic > 0).mean()) if len(ic) else float("nan")}
        b.update(demeaned_auc(d))
        return b

    for f, d in oos.groupby("fold"):
        res["folds"][int(f)] = block(d)
    res["global"] = block(oos)
    res["global"]["alpha"] = net_alpha(oos)
    ic = daily_rank_ic(oos)
    res["_ic_series_len"] = int(len(ic))
    res["global"]["ic_sigma_daily"] = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")
    return res


# ─── main ─────────────────────────────────────────────────────────────────────

def load_panel(cache: Path) -> pd.DataFrame:
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = build_dataset()
        c = df.select_dtypes("float64").columns
        df[c] = df[c].astype("float32")
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    return df


def zero_return_fraction(tickers) -> dict:
    out = {}
    for tk in tickers:
        raw = store_get(tk)
        if raw is None or len(raw) < 2:
            continue
        r = raw["Close"].pct_change().dropna()
        out[tk] = float((r == 0).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data" / "eval" / "_n0a_panel_cache.parquet"))
    args = ap.parse_args()

    df = load_panel(Path(args.cache))
    cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df[np.isfinite(df["ret_fwd"])].reset_index(drop=True)
    print(f"Panel: {len(df):,} filas, {df['ticker'].nunique()} tickers, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")

    # 1. historia usable
    cnt = df.groupby("ticker").size()
    hist = {"n_tickers_panel": int(len(cnt)),
            "n_mx": int(sum(t.endswith(".MX") for t in cnt.index)),
            "n_gt50": int((cnt > 50).sum()), "n_gt250": int((cnt > 250).sum()),
            "n_gt1000": int((cnt > 1000).sum()),
            "rows_per_ticker_quantiles": {str(q): float(cnt.quantile(q)) for q in (0, .25, .5, .75, 1)},
            "tickers_non_mx": sorted(t for t in cnt.index if not t.endswith(".MX"))[:20]}
    ntk_date = df.groupby("date")["ticker"].nunique()
    hist["tickers_por_fecha"] = {"mediana": float(ntk_date.median()), "max": int(ntk_date.max()),
                                 "fechas_con_>=8": int((ntk_date >= MIN_NAMES).sum()),
                                 "n_fechas": int(len(ntk_date))}
    print("Historia:", json.dumps(hist, ensure_ascii=False))

    # mascara de calidad
    q = pd.read_parquet(ROOT / "data" / "quality" / "quality_mask.parquet",
                        columns=["ticker", "date", "valid"])
    q["date"] = pd.to_datetime(q["date"])
    q["ticker"] = q["ticker"].astype(str)
    # el panel usa tickers sin sufijo (ALSEA); la mascara usa ALSEA.MX
    mask_tk = set(q["ticker"])
    tk_map = {t: (t + ".MX" if (t + ".MX") in mask_tk else t) for t in df["ticker"].unique()}
    unmapped = sorted(t for t, m in tk_map.items() if m not in mask_tk)
    q["ticker"] = q["ticker"].str.replace(r"\.MX$", "", regex=True)
    q = q[q["ticker"].isin(set(df["ticker"]))]
    df = df.merge(q, on=["ticker", "date"], how="left")
    cov = float(df["valid"].notna().mean())
    # filas sin entrada en la mascara (tickers sin mascara o fechas posteriores a su fin): se conservan
    sin_mask = float(df["valid"].isna().mean())
    df["valid"] = df["valid"].fillna(True).astype(bool)
    print(f"Tickers sin mascara: {unmapped}; filas sin mascara conservadas: {sin_mask:.1%}")
    del q
    print(f"Cobertura de quality_mask en el panel: {cov:.1%}; valid={df['valid'].mean():.1%}")

    zf = zero_return_fraction(sorted(df["ticker"].unique()))
    bad_tk = {t for t, f in zf.items() if f > ZERO_FRAC_MAX}
    print(f"Tickers con >{ZERO_FRAC_MAX:.0%} de dias con retorno 0: {len(bad_tk)}")

    configs = {
        "base": lambda d: d,
        "A_valid": lambda d: d[d["valid"]],
        "B_valid_winsor25": lambda d: _winsor(d[d["valid"]]),
        "C_valid_winsor25_sin_ceros": lambda d: _winsor(d[d["valid"] & ~d["ticker"].isin(bad_tk)]),
    }

    results: dict = {"historia": hist, "quality_mask_coverage": cov, "filas_sin_mascara_conservadas": sin_mask, "tickers_sin_mascara": unmapped,
                     "tickers_excluidos_ceros": sorted(bad_tk),
                     "zero_frac_por_ticker_top10": dict(sorted(zf.items(), key=lambda kv: -kv[1])[:10]),
                     "configs": {}}
    sigma_for_power = None
    n_weeks = None

    for name, fn in configs.items():
        d = fn(df).reset_index(drop=True)
        print(f"\n=== {name}: {len(d):,} filas, {d['ticker'].nunique()} tickers ===")
        oos = score_oos(d, cols)
        r = evaluate(oos)
        r["n_rows_config"] = int(len(d))
        r["n_tickers_config"] = int(d["ticker"].nunique())
        results["configs"][name] = r
        g = r["global"]
        print(f"  AUC={g['auc_pooled']:.4f} IC={g['ic_mean']:.4f} t_nw5={g['ic_t_nw5']:.2f} "
              f"AUCdate={g['auc_date_median']:.4f} AUCtk={g['auc_ticker']:.4f} "
              f"alfa_long_net={g.get('alpha', {}).get('long_excess_net_mean_pct', float('nan')):.3f}%")
        for f, b in r["folds"].items():
            print(f"   fold {f}: AUC={b['auc_pooled']:.4f} IC={b['ic_mean']:.4f} t={b['ic_t_nw5']:.2f} "
                  f"AUCdate={b['auc_date_median']:.4f} AUCtk={b['auc_ticker']:.4f} ndates={b['n_dates']}")
        # power con el IC por fecha de esta config
        ic = daily_rank_ic(oos)
        n_nonov = max(len(ic) // HORIZON, 1)       # periodos no solapados de 5d
        sig = float(ic.std(ddof=1))
        mde_t2 = (2.0 + stats.norm.ppf(0.8)) * sig / np.sqrt(n_nonov)
        mde_80 = (stats.norm.ppf(0.95) + stats.norm.ppf(0.8)) * sig / np.sqrt(n_nonov)
        r["power"] = {"sigma_ic": sig, "n_fechas_ic": int(len(ic)), "n_periodos_no_solapados": n_nonov,
                      "se_ic_no_solapado": sig / np.sqrt(n_nonov),
                      "mde_ic_alpha05_pot80": float(mde_80), "mde_ic_t2_pot80": float(mde_t2),
                      "semanas_para_ic_0.02": required_weeks(0.02, sig),
                      "semanas_para_ic_0.05": required_weeks(0.05, sig)}
        print(f"  power: sigma_IC={sig:.3f} n_nonov={n_nonov} MDE(t>2,pot80)={mde_t2:.4f} MDE(a.05,pot80)={mde_80:.4f}")

        trial_metrics = {"auc_pooled": g["auc_pooled"], "ic_mean": g["ic_mean"], "ic_t_nw5": g["ic_t_nw5"],
                         "auc_date_median": g["auc_date_median"], "auc_ticker": g["auc_ticker"],
                         "alpha_long_net_pct": g.get("alpha", {}).get("long_excess_net_mean_pct"),
                         "alpha_ls_net_pct": g.get("alpha", {}).get("ls_net_mean_pct"),
                         "mde_ic": float(mde_t2)}
        log_trial(f"n0a_diagnostico:{name}",
                  {"model": "woe_logistic", "C": C_REG, "n_splits": N_SPLITS, "embargo": EMBARGO,
                   "horizon": HORIZON, "label_span": LABEL_SPAN, "entry": "next_open",
                   "n_features": TOP_FEATURES, "n_bins": N_BINS_WOE, "cleaning": name,
                   "winsor_pct": WINSOR if "winsor" in name else None,
                   "zero_frac_max": ZERO_FRAC_MAX if name.startswith("C") else None,
                   "min_names": MIN_NAMES, "nw_lag": NW_LAG, "cost_side_pct": COST_SIDE},
                  {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                   for k, v in trial_metrics.items()},
                  int(len(oos)), path=str(ROOT / TRIALS_PATH))
        del oos, d
        gc.collect()

    # veredicto
    verd = {}
    for name, r in results["configs"].items():
        g = r["global"]
        a = g.get("alpha", {})
        ic_ok = bool(g["ic_t_nw5"] > 2)
        alpha_ok = bool(a.get("long_excess_net_mean_pct", -1) > 0 or a.get("ls_net_mean_pct", -1) > 0)
        verd[name] = {"ic_t_gt_2": ic_ok, "alpha_neto_gt_0": alpha_ok,
                      "pasa_criterio_IC_y_alfa": ic_ok and alpha_ok,
                      "ic_mean_vs_mde": [g["ic_mean"], r["power"]["mde_ic_t2_pot80"]],
                      "clase": ("senal_detectada" if ic_ok else
                                "no_concluyente" if abs(g["ic_mean"]) < r["power"]["mde_ic_t2_pot80"]
                                else "no_hay_senal")}
    results["veredicto"] = verd
    results["notas"] = [
        "DSR no calculado aqui (criterio alterno DSR>0.95 se evalua en tandas posteriores).",
        f"IC por fecha solo con >= {MIN_NAMES} emisoras; t Newey-West lag {NW_LAG} (y 10).",
        "Retorno cero medido en retornos diarios Close->Close del store por ticker.",
        "Cada config reentrena WoE+LR por fold con train y test filtrados igual.",
        "Alfa: rebalanceo no solapado cada 5 fechas; largo tercil superior vs promedio universo, "
        "costo 2*0.30%; L/S 4*0.30%. Sin impacto de mercado ni borrow.",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "n0a_diagnostico.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_md(results)
    print("\nGuardado data/eval/n0a_diagnostico.json y .md")


def _winsor(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["ret_fwd"] = d["ret_fwd"].clip(-WINSOR, WINSOR)
    d["y"] = (d["ret_fwd"] > 0).astype(int)
    return d


def write_md(R: dict) -> None:
    L = ["# N0-A diagnostico (WoE+Logistic, 6 folds por fechas)", ""]
    h = R["historia"]
    L += ["## 1. Historia usable (panel del CV)",
          f"- Tickers en panel: {h['n_tickers_panel']} (.MX: {h['n_mx']}); >50 filas: {h['n_gt50']}; "
          f">250: {h['n_gt250']}; >1000: {h['n_gt1000']}",
          f"- Emisoras por fecha: mediana {h['tickers_por_fecha']['mediana']}, max {h['tickers_por_fecha']['max']}; "
          f"fechas con >= {MIN_NAMES}: {h['tickers_por_fecha']['fechas_con_>=8']}/{h['tickers_por_fecha']['n_fechas']}",
          f"- Cobertura quality_mask: {R['quality_mask_coverage']:.1%}; excluidos por >5% ceros: "
          f"{len(R['tickers_excluidos_ceros'])}", "",
          "## 2-5. Resultados globales por configuracion", "",
          "| config | filas | tick. | AUC | IC medio | t NW5 | AUC dem.fecha | AUC terc. | AUC dem.ticker | alfa largo neto % | L/S neto % |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in R["configs"].items():
        g, a = r["global"], r["global"].get("alpha", {})
        L.append(f"| {n} | {r['n_rows_config']} | {r['n_tickers_config']} | {g['auc_pooled']:.4f} | "
                 f"{g['ic_mean']:.4f} | {g['ic_t_nw5']:.2f} | {g['auc_date_median']:.4f} | "
                 f"{g['auc_date_terciles']:.4f} | {g['auc_ticker']:.4f} | "
                 f"{a.get('long_excess_net_mean_pct', float('nan')):.3f} | {a.get('ls_net_mean_pct', float('nan')):.3f} |")
    L += ["", "## Por fold (config base)", "", "| fold | AUC | IC | t NW5 | AUC dem.fecha | AUC dem.ticker | fechas |", "|---|---|---|---|---|---|---|"]
    for f, b in R["configs"]["base"]["folds"].items():
        L.append(f"| {f} | {b['auc_pooled']:.4f} | {b['ic_mean']:.4f} | {b['ic_t_nw5']:.2f} | "
                 f"{b['auc_date_median']:.4f} | {b['auc_ticker']:.4f} | {b['n_dates']} |")
    L += ["", "## 6. Poder / MDE", "", "| config | sigma IC | periodos 5d no solapados | MDE IC (t>2, pot 80%) | MDE (a=.05, pot 80%) |", "|---|---|---|---|---|"]
    for n, r in R["configs"].items():
        p = r["power"]
        L.append(f"| {n} | {p['sigma_ic']:.3f} | {p['n_periodos_no_solapados']} | "
                 f"{p['mde_ic_t2_pot80']:.4f} | {p['mde_ic_alpha05_pot80']:.4f} |")
    L += ["", "## Veredicto", ""]
    for n, v in R["veredicto"].items():
        L.append(f"- {n}: t>2={v['ic_t_gt_2']}, alfa neto>0={v['alpha_neto_gt_0']}, "
                 f"pasa={v['pasa_criterio_IC_y_alfa']}, clase={v['clase']}")
    L += ["", "## Notas"] + [f"- {x}" for x in R["notas"]]
    (OUT_DIR / "n0a_diagnostico.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
