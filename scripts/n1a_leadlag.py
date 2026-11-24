"""
Nivel 1 / idea A: lead-lag US -> Mexico (BMV).

Reusa scripts/n0a_diagnostico.py (panel, 6 folds con purga, WoE+LR, IC con Newey-West, alfa neto
0.30%/lado) y scripts/n0b_diagnostico.py (liquidez). No modifica codigo existente.

ALINEACION SIN LEAKAGE (calendario BMV = fechas del panel)
  * Fila (ticker, t): features con informacion hasta el CIERRE del dia t. Etiqueta ret_fwd = Close[t+6]/Open[t+1]-1.
  * Cierre NY (16:00 ET) del dia t coincide con el cierre BMV (15:00 CT) y ocurre ANTES del Open t+1 de la BMV
    => el retorno US del dia t se conoce antes de la entrada (Open t+1). No hay fuga.
  * Retorno US para la fila t = acumulado US en (t_prev_BMV, t]: si BMV cerro un dia que NY operaba, ese retorno
    se acumula en el siguiente dia BMV (informacion pasada). Si NY cerro y BMV opera, aporta 0.
  * Betas: rodantes 252 dias (min 120) con datos hasta t-1 (shift 1), emisora vs factor, misma fecha.
  * Residual: r_i,t(cierre t) - beta_{t-1} * F_t; solo usa precios hasta el cierre t.
  * Gap ADR->local: ADR_t - (local_t - USDMXN_t) en log%, cierres del dia t (todo hasta t).
  * Ninguna feature usa retornos MX de t+1 o posteriores (auditoria automatica de betas dentro del script).

Ejecucion: .venv/bin/python scripts/n1a_leadlag.py [--only corr|model]
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import scripts.n0a_diagnostico as A
import scripts.n0b_diagnostico as B
from src.data.bmv_catalog import BMV_YFINANCE
from src.data.store import get as store_get
from src.ml.features import FEATURE_COLS
from src.ml.trials import DEFAULT_PATH as TRIALS_PATH, log_trial

OUT_DIR = ROOT / "data" / "eval"
SCRATCH = Path("/tmp/claude-1000/-home-gomiten-Documents-stock-market-etl-ml/6d78f6e6-2a0a-441a-9d97-5e424d20393a/scratchpad")
US_EXTRA = SCRATCH / "us"          # EWW, HG=F, ^TNX, ^VIX descargados con yfinance (datos publicos)
BETA_WIN, BETA_MIN = 252, 120
TEST_START = pd.Timestamp("2024-01-01")
ADR = {"AMXL": "AMX", "FEMSAUBD": "FMX"}   # panel -> ADR en NYSE

# factor -> (fuente, tipo)  tipo: 'lret' retorno log %, 'diff' cambio de nivel (puntos)
FACTORS = {"spy": ("SPY", "lret"), "ewz": ("EWZ", "lret"), "eem": ("EEM", "lret"), "xle": ("XLE", "lret"),
           "hg": ("HG=F", "lret"), "tnx": ("^TNX", "diff"), "mxn": ("MXN=X", "lret"), "vix": ("^VIX", "lret"),
           "eww": ("EWW", "lret")}
BETA_FACTORS = ["spy", "ewz", "eem", "xle", "hg", "mxn", "eww"]
LEVEL_NEW_MKT = [f"us_{k}" for k in FACTORS]


def log(name, cfg, metrics, n):
    base = {"model": "woe_logistic", "n_splits": A.N_SPLITS, "embargo": A.EMBARGO, "horizon": A.HORIZON,
            "label_span": A.LABEL_SPAN, "cost_side_pct": A.COST_SIDE, "tanda": "n1a", "entry": "next_open"}
    base.update(cfg)
    log_trial(name, base, B.clean(metrics), int(n), path=str(ROOT / TRIALS_PATH))


# ─── datos US ─────────────────────────────────────────────────────────────────

def load_us(src: str) -> pd.DataFrame | None:
    p = US_EXTRA / f"{src}.parquet"
    d = pd.read_parquet(p) if p.exists() else store_get(src)
    if d is None or len(d) < 50:
        return None
    d = d[~d.index.duplicated()].sort_index()
    d.index = pd.to_datetime(d.index)
    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    return d


def factor_cum(src: str, kind: str) -> pd.Series | None:
    d = load_us(src)
    if d is None:
        return None
    c = d["Close"].astype("float64")
    if kind == "lret":
        return np.log(c).diff().fillna(0).cumsum() * 100.0
    return c.diff().fillna(0).cumsum()


def bmv_factor_returns(cal: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """F_t = acumulado US en (t_prev, t] para cada fecha BMV t (asof: ultima fecha US <= t)."""
    out, missing = {}, []
    prev = pd.DatetimeIndex(pd.Series(cal).shift(1).fillna(cal[0]))
    for k, (src, kind) in FACTORS.items():
        cs = factor_cum(src, kind)
        if cs is None:
            missing.append(k)
            continue
        a = cs.reindex(cs.index.union(cal)).ffill()
        out[k] = pd.Series(a.reindex(cal).to_numpy() - a.reindex(prev).to_numpy(), index=cal).fillna(0.0)
    return pd.DataFrame(out), {"faltantes": missing}


# ─── features por emisora ─────────────────────────────────────────────────────

def stock_series(tk: str) -> pd.DataFrame | None:
    raw = store_get(BMV_YFINANCE.get(tk, tk))
    if raw is None or len(raw) < 150:
        return None
    raw = raw[~raw.index.duplicated()].sort_index()
    raw.index = pd.to_datetime(raw.index)
    return raw[["Open", "Close"]].astype("float64")


def build_new_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cal = pd.DatetimeIndex(sorted(df["date"].unique()))
    F, meta = bmv_factor_returns(cal)
    meta["factores_usados"] = list(F.columns)
    adr_ret = {}
    for tk, adr in ADR.items():
        cs = factor_cum(adr, "lret")
        if cs is not None:
            adr_ret[tk] = cs.reindex(cs.index.union(cal)).ffill().reindex(cal).diff().fillna(0.0)
    meta["adr_disponibles"] = sorted(adr_ret)
    frames, audit = [], {}
    for tk in df["ticker"].unique():
        s = stock_series(tk)
        if s is None:
            continue
        r = (np.log(s["Close"]).diff() * 100.0).reindex(cal)       # retorno MX con cierre t
        x = pd.DataFrame(index=cal)
        for k in F.columns:
            x[f"us_{k}"] = F[k].astype("float32")
        for k in BETA_FACTORS:
            if k not in F:
                continue
            beta = (r.rolling(BETA_WIN, min_periods=BETA_MIN).cov(F[k]) /
                    F[k].rolling(BETA_WIN, min_periods=BETA_MIN).var()).shift(1)   # datos hasta t-1
            x[f"bx_{k}"] = beta * F[k]
            if k in ("spy", "ewz"):
                x[f"res_{k}"] = r - beta * F[k]
            if k == "spy" and len(audit) < 3:
                r2 = r.copy()
                r2.iloc[-200:] = r2.iloc[-200:] * 3 + 5.0
                b2 = (r2.rolling(BETA_WIN, min_periods=BETA_MIN).cov(F[k]) /
                      F[k].rolling(BETA_WIN, min_periods=BETA_MIN).var()).shift(1)
                cut = len(cal) - 200
                audit[tk] = bool(np.allclose(beta.iloc[:cut + 1].to_numpy(), b2.iloc[:cut + 1].to_numpy(), equal_nan=True))
        x["adr_gap"] = (adr_ret[tk] - (r - F["mxn"])) if (tk in adr_ret and "mxn" in F) else 0.0
        x["ticker"] = tk
        x.index.name = "date"
        frames.append(x.reset_index())
    N = pd.concat(frames, ignore_index=True)
    N["date"] = pd.to_datetime(N["date"])
    for c in N.columns:
        if N[c].dtype == "float64":
            N[c] = N[c].astype("float32")
    meta["auditoria_beta_sin_futuro"] = audit
    return N, meta


# ─── (a) correlacion cruzada rezagada con FDR ─────────────────────────────────

def bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    m = len(p)
    o = np.argsort(p)
    ok = p[o] <= q * (np.arange(1, m + 1) / m)
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    rej = np.zeros(m, bool)
    rej[o[:k]] = True
    return rej


def corr_targets(tk: str, dates: pd.DatetimeIndex) -> pd.DataFrame | None:
    s = stock_series(tk)
    if s is None:
        return None
    o, c = s["Open"], s["Close"]
    t = pd.DataFrame({
        "cc0": np.log(c).diff() * 100.0,                         # control: mismo dia
        "gap": (o.shift(-1) / c - 1) * 100.0,                   # cierre t -> open t+1
        "oc1": (c.shift(-1) / o.shift(-1) - 1) * 100.0,         # open t+1 -> close t+1
    })
    return t.reindex(dates)


def part_corr(df: pd.DataFrame) -> dict:
    facs = [c for c in df.columns if c.startswith("us_")]
    rows = []
    for tk, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        tg = corr_targets(tk, pd.DatetimeIndex(g["date"]))
        if tg is None:
            continue
        g = pd.concat([g, tg.reset_index(drop=True)], axis=1)
        for tgt in ("ret_fwd", "gap", "oc1", "cc0"):
            for per, msk in (("full", np.ones(len(g), bool)), ("2024+", (g["date"] >= TEST_START).to_numpy())):
                for f in facs:
                    z = g.loc[msk, [f, tgt]].dropna()
                    if len(z) < 120 or z[f].std() == 0 or z[tgt].std() == 0:
                        continue
                    a = (z[f] - z[f].mean()) / z[f].std()
                    b = (z[tgt] - z[tgt].mean()) / z[tgt].std()
                    m, _, t = A.nw_t((a * b).to_numpy(), A.NW_LAG if tgt == "ret_fwd" else 1)
                    rows.append({"ticker": tk, "factor": f[3:], "target": tgt, "periodo": per, "n": len(z),
                                 "corr": m, "t_nw": t, "p": B.p_from_t(t)})
    R = pd.DataFrame(rows)
    out = {"n_tests_por_familia": {}, "familias": {}}
    for (tgt, per), z in R.groupby(["target", "periodo"]):
        z = z.reset_index(drop=True)
        z["bh_reject"] = bh(z["p"].to_numpy())
        fam = f"{tgt}|{per}"
        out["n_tests_por_familia"][fam] = int(len(z))
        top = z.reindex(z["t_nw"].abs().sort_values(ascending=False).index).head(8)
        by_f = z.groupby("factor").agg(corr_media=("corr", "mean"), t_med=("t_nw", "mean"),
                                       n_bh=("bh_reject", "sum"), n=("corr", "size")).round(4)
        out["familias"][fam] = {
            "n_rechazos_BH_q05": int(z["bh_reject"].sum()),
            "n_p_lt_05_sin_corregir": int((z["p"] < 0.05).sum()),
            "esperados_por_azar_05": round(0.05 * len(z), 1),
            "mediana_corr": float(z["corr"].median()),
            "top8": top[["ticker", "factor", "corr", "t_nw", "p", "bh_reject"]].round(4).to_dict("records"),
            "por_factor": by_f.reset_index().to_dict("records")}
    return out


# ─── (b) modelo con features nuevas ───────────────────────────────────────────

def tercile_blocks(oos: pd.DataFrame, liq: pd.DataFrame) -> dict:
    d = oos.merge(liq[["ticker", "date", "adv"]], on=["ticker", "date"], how="left")
    r = d.groupby("date")["adv"].rank(pct=True)
    d["terc"] = pd.cut(r, [0, 1 / 3, 2 / 3, 1.0001], labels=["T1_iliquido", "T2", "T3_liquido"])
    out = {}
    for lab in ("T1_iliquido", "T2", "T3_liquido"):
        z = d[d["terc"] == lab]
        if len(z):
            b = B.cell_block(z)
            out[lab] = {k: b[k] for k in ("n", "ic_mean", "ic_t_nw5", "alpha_net_pct")}
    return out


def year_blocks(ic: pd.Series) -> dict:
    out = {}
    for y, v in ic.groupby(ic.index.year):
        m, _, t = A.nw_t(v.values)
        out[int(y)] = {"ic": m, "t": t, "n_fechas": int(len(v))}
    post = ic[ic.index >= TEST_START]
    m, _, t = A.nw_t(post.values)
    out["2024+"] = {"ic": m, "t": t, "n_fechas": int(len(post))}
    return out


def fold_alphas(oos: pd.DataFrame) -> dict:
    res = {}
    for fo, z in oos.groupby("fold"):
        a = A.net_alpha(z.reset_index(drop=True))
        ic = A.daily_rank_ic(z)
        res[int(fo)] = {"alpha_long_net_pct": a.get("long_excess_net_mean_pct"),
                        "ic": float(ic.mean()) if len(ic) else None}
    return res


def run_config(name, df, cols, top, liq, base_ic):
    A.TOP_FEATURES = top   # score_oos lee la constante del modulo n0a (sin editar el archivo)
    oos = A.score_oos(df, cols)
    ev = A.evaluate(oos)
    g = ev["global"]
    ic = A.daily_rank_ic(oos)
    a = g.get("alpha", {})
    fa = fold_alphas(oos)
    r = {"n_features_entrada": len(cols), "top_features": top, "n_oos": int(len(oos)),
         "auc": g["auc_pooled"], "ic": g["ic_mean"], "t_nw5": g["ic_t_nw5"], "t_nw10": g["ic_t_nw10"],
         "auc_date_median": g["auc_date_median"], "alpha_long_net_pct": a.get("long_excess_net_mean_pct"),
         "ls_net_pct": a.get("ls_net_mean_pct"), "folds": fa,
         "folds_alpha_pos": int(sum(1 for v in fa.values() if (v["alpha_long_net_pct"] or -1) > 0)),
         "folds_ic_pos": int(sum(1 for v in fa.values() if (v["ic"] or -1) > 0)),
         "por_tercil_liquidez": tercile_blocks(oos, liq), "por_anio": year_blocks(ic)}
    if base_ic is not None:
        j = pd.concat([ic, base_ic], axis=1, join="inner")
        m, _, t = A.nw_t((j.iloc[:, 0] - j.iloc[:, 1]).to_numpy())
        r["delta_ic_vs_base_ctrl"] = {"media": m, "t_nw5": t, "n_fechas": int(len(j))}
    return r, oos, ic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--cache", default=str(SCRATCH / "panel.parquet"))
    args = ap.parse_args()

    df = A.load_panel(Path(args.cache))
    cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df[np.isfinite(df["ret_fwd"])].reset_index(drop=True)
    print(f"Panel {len(df):,} filas, {df['ticker'].nunique()} tickers, {len(cols)} features base", flush=True)

    N, meta = build_new_features(df)
    df = df.merge(N, on=["date", "ticker"], how="left")
    del N
    gc.collect()
    feat_mkt = [c for c in LEVEL_NEW_MKT if c in df.columns]
    feat_beta = [c for c in df.columns if c.startswith(("bx_", "res_"))]
    feat_adr = ["adr_gap"]
    allnew = feat_mkt + feat_beta + feat_adr
    meta["cobertura_pre_relleno"] = {c: float(df[c].notna().mean()) for c in allnew}
    df[allnew] = df[allnew].fillna(0.0).astype("float32")   # NaN (primeros ~120 dias de beta) -> 0 neutro
    meta["features_nuevas"] = {"mkt_igual_para_todas_las_emisoras": feat_mkt, "exposicion_beta": feat_beta, "adr": feat_adr}
    print("meta:", json.dumps(B.clean(meta), ensure_ascii=False)[:700], flush=True)

    p = OUT_DIR / "n1a_leadlag.json"
    R = json.loads(p.read_text(encoding="utf-8")) if (args.only == "model" and p.exists()) else {}
    R["meta"] = B.clean(meta)
    R["panel"] = {"filas": int(len(df)), "tickers": int(df["ticker"].nunique())}
    R["descargas"] = {k: (US_EXTRA / f"{k}.parquet").exists() for k in ("EWW", "HG=F", "^TNX", "^VIX")}

    if args.only in ("", "corr"):
        print("\n[a] correlacion cruzada + FDR", flush=True)
        R["a_corr"] = B.clean(part_corr(df[["date", "ticker", "ret_fwd"] + [c for c in df.columns if c.startswith("us_")]]))
        for fam, v in R["a_corr"]["familias"].items():
            print(f"  {fam}: BH={v['n_rechazos_BH_q05']} p<.05={v['n_p_lt_05_sin_corregir']} azar={v['esperados_por_azar_05']} medcorr={v['mediana_corr']:.4f}", flush=True)
            log(f"n1a_xcorr:{fam}", {"familia": fam, "n_tests": R["a_corr"]["n_tests_por_familia"][fam],
                                      "factores": list(FACTORS), "fdr": "BH q=0.05", "nw_lag": A.NW_LAG},
                {"bh_rechazos": v["n_rechazos_BH_q05"], "p05": v["n_p_lt_05_sin_corregir"],
                 "mediana_corr": v["mediana_corr"]}, int(len(df)))
        p.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    if args.only in ("", "model"):
        print("\n[b] modelo", flush=True)
        liq, _ = B.liquidity_panel(sorted(df["ticker"].unique()))
        variants = [
            ("base_top15", cols, 15), ("base_top25", cols, 25),
            ("mkt_top15", cols + feat_mkt, 15), ("beta_top15", cols + feat_beta, 15),
            ("adr_top15", cols + feat_adr, 15), ("all_top15", cols + allnew, 15),
            ("all_top25", cols + allnew, 25), ("solo_nuevas_top15", allnew, 15)]
        R["b_modelo"] = {}
        ics = {}
        for nm, cs_, top in variants:
            ctrl = "base_top15" if top == 15 else "base_top25"
            r, oos, ic = run_config(nm, df, cs_, top, liq, ics.get(ctrl) if nm != ctrl else None)
            ics[nm] = ic
            R["b_modelo"][nm] = B.clean(r)
            print(f"  {nm}: AUC={r['auc']:.4f} IC={r['ic']:.4f} t={r['t_nw5']:.2f} alfa={r['alpha_long_net_pct']:.3f}% "
                  f"folds_alfa+={r['folds_alpha_pos']}/6 IC2024+={r['por_anio']['2024+']['ic']:.4f} "
                  f"dIC={r.get('delta_ic_vs_base_ctrl', {}).get('media', float('nan')):.4f}", flush=True)
            log(f"n1a_modelo:{nm}", {"variant": nm, "n_features_entrada": len(cs_), "top_features": top,
                                      "C": A.C_REG, "n_bins": A.N_BINS_WOE, "nw_lag": A.NW_LAG, "beta_win": BETA_WIN},
                {"auc": r["auc"], "ic": r["ic"], "t_nw5": r["t_nw5"], "alpha_long_net_pct": r["alpha_long_net_pct"],
                 "ic_2024p": r["por_anio"]["2024+"]["ic"], "folds_alpha_pos": r["folds_alpha_pos"]}, len(oos))
            del oos
            gc.collect()
            p.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        R["veredicto"] = veredicto(R)

    p.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_md(R)
    print("Guardado n1a_leadlag.json/.md", flush=True)


def veredicto(R: dict) -> dict:
    out = {}
    for nm, r in R["b_modelo"].items():
        ctrl = R["b_modelo"]["base_top25" if nm.endswith("25") else "base_top15"]
        d_ic = r["ic"] - ctrl["ic"]
        out[nm] = {"t_nw5_gt_2": bool(r["t_nw5"] > 2), "alfa_neto_gt_0": bool((r["alpha_long_net_pct"] or -1) > 0),
                   "alfa_pos_mayoria_folds": bool(r["folds_alpha_pos"] >= 4),
                   "delta_ic_vs_control": d_ic, "delta_ic_gt_0.03": bool(d_ic > 0.03),
                   "pasa_criterio": bool(r["t_nw5"] > 2 and r["folds_alpha_pos"] >= 4 and d_ic > 0.03)}
    return out


def f(x, nd=4):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def write_md(R: dict) -> None:
    m = R["meta"]
    L = ["# N1-A lead-lag US -> Mexico", "", "## Datos y alineacion",
         f"- Descargas yfinance: {R['descargas']}", f"- Factores usados: {m.get('factores_usados')}; faltantes: {m.get('faltantes')}",
         f"- ADR disponibles: {m.get('adr_disponibles')} (solo AMXL->AMX, FEMSAUBD->FMX; el resto adr_gap=0)",
         f"- Auditoria beta sin futuro (corromper retornos MX recientes no cambia betas previos): {m.get('auditoria_beta_sin_futuro')}",
         "- Retorno US de la fila t = acumulado US en (t_prev_BMV, t]; BMV cierra 15:00 CT = 16:00 ET, asi que el cierre NY de t se conoce antes del Open t+1 (entrada).",
         "- Betas rodantes 252d con datos hasta t-1; residual y gap ADR usan solo cierres del dia t.",
         "- Las features `us_*` son iguales para todas las emisoras de una fecha: no pueden mover el IC transversal por si solas (solo AUC pooled).", ""]
    if "a_corr" in R:
        L += ["## (a) Correlacion cruzada por emisora (FDR Benjamini-Hochberg q=0.05)", "",
              "corr(factor US del dia t, objetivo); t Newey-West (lag 5 en ret_fwd, 1 en los demas). In-sample, diagnostico.", "",
              "| objetivo/periodo | tests | rechazos BH | p<.05 sin corregir | esperados por azar | mediana corr |", "|---|---|---|---|---|---|"]
        for fam, v in R["a_corr"]["familias"].items():
            L.append(f"| {fam} | {R['a_corr']['n_tests_por_familia'][fam]} | {v['n_rechazos_BH_q05']} | {v['n_p_lt_05_sin_corregir']} | {v['esperados_por_azar_05']} | {f(v['mediana_corr'])} |")
        L += ["", "Top 8 por |t| en ret_fwd (completo):", "", "| emisora | factor | corr | t | p | BH |", "|---|---|---|---|---|---|"]
        for r in R["a_corr"]["familias"].get("ret_fwd|full", {}).get("top8", []):
            L.append(f"| {r['ticker']} | {r['factor']} | {f(r['corr'])} | {f(r['t_nw'], 2)} | {f(r['p'])} | {r['bh_reject']} |")
        L.append("")
    if "b_modelo" in R:
        L += ["## (b) Modelo base + features nuevas (mismos 6 folds, WoE+Logistica)", "",
              "| variante | AUC | IC | t NW5 | alfa largo neto % | folds alfa+ | IC 2024+ | t 2024+ | dIC vs control | t dIC |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for nm, r in R["b_modelo"].items():
            dd = r.get("delta_ic_vs_base_ctrl", {})
            y = r["por_anio"]["2024+"]
            L.append(f"| {nm} | {f(r['auc'])} | {f(r['ic'])} | {f(r['t_nw5'], 2)} | {f(r['alpha_long_net_pct'], 3)} | {r['folds_alpha_pos']}/6 | {f(y['ic'])} | {f(y['t'], 2)} | {f(dd.get('media'))} | {f(dd.get('t_nw5'), 2)} |")
        L += ["", "Control: top15 -> base_top15; top25 -> base_top25.", "", "### Por tercil de liquidez (ADV 60d): IC / t / alfa neto %", "",
              "| variante | T1 iliquido | T2 | T3 liquido |", "|---|---|---|---|"]
        for nm, r in R["b_modelo"].items():
            c = r["por_tercil_liquidez"]
            L.append(f"| {nm} | " + " | ".join(f"{f(c[k]['ic_mean'])} / {f(c[k]['ic_t_nw5'], 2)} / {f(c[k]['alpha_net_pct'], 2)}" for k in ("T1_iliquido", "T2", "T3_liquido") if k in c) + " |")
        L += ["", "### IC por anio", "", "| anio | base_top15 | beta_top15 | all_top15 |", "|---|---|---|---|"]
        yrs = sorted(k for k in R["b_modelo"]["base_top15"]["por_anio"] if k != "2024+") + ["2024+"]
        for y in yrs:
            L.append(f"| {y} | " + " | ".join(f(R["b_modelo"][n]["por_anio"].get(y, {}).get("ic")) for n in ("base_top15", "beta_top15", "all_top15")) + " |")
        L += ["", "## Veredicto (criterio: t NW5>2, alfa neto>0 en >=4/6 folds, dIC>0.03)", ""]
        for nm, v in R.get("veredicto", {}).items():
            L.append(f"- {nm}: t>2={v['t_nw5_gt_2']}, alfa>0={v['alfa_neto_gt_0']}, folds alfa+ mayoria={v['alfa_pos_mayoria_folds']}, dIC={f(v['delta_ic_vs_control'])} (>0.03: {v['delta_ic_gt_0.03']}), pasa={v['pasa_criterio']}")
        L += ["", "DSR no calculado en esta tanda."]
    (OUT_DIR / "n1a_leadlag.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
