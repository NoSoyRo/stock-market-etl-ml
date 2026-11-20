"""
Nivel 3 / PRUEBA B: features OHLCV de microestructura y memoria como feature del Ridge-rank y como GATE.

PASO 1 (cobertura de eventos, comprobada ANTES de elegir; ver data/eval/n3_prueba.md):
  * data/news/history/articles.parquet: 909 titulares, solo WALMEX con >=100/ano (2018-2020); 0 para el resto del panel.
  * data/news/archive (PIT): 556 titulares, 304 de 2026 (la historia previa son titulares viejos descargados tarde, sin PIT).
  * data/microstructure/hechos: flujo por casa (no son "hechos relevantes"), solo desde 2025-06-30.
  * data/eps: 65 emisoras US, ninguna del panel de 35 mexicanas.  data/calendars/events.csv: banxico/fomc/inpc (macro, sin
    emisora; mismo dia para todas, sin informacion cross-sectional).
  => NO hay >=5 anos con timestamp para >=15 emisoras: se ejecuta la PRUEBA B (no la A).

HIPOTESIS: proxies OHLCV de liquidez/volatilidad/memoria (Corwin-Schultz, Amihud, Garman-Klass, Parkinson, razon gap/rango,
Hurst, entropia de permutacion) anaden IC al Ridge-rank (rank_aug de N1-B) y/o definen un regimen donde el IC es material.

PRE-REGISTRO (fijado ANTES de correr; no se ajusta despues)
  Features (todas con datos <= t, ventana hacia atras; rank cross-sectional por fecha en [-0.5, 0.5]):
    cs60 (Corwin-Schultz medio 60d), amihud60 (log media 60d de |r|/volumen en MXN), gk60 (sqrt media 60d Garman-Klass),
    pk60 (sqrt media 60d Parkinson), gaprng60 (media 60d de |ln(O_t/C_t-1)|/ln(H/L)),
    hurst100 (0.5*pendiente log-var(k-retornos) vs log k, k=1,2,4,8,16, ventana 100d),
    pe100 (entropia de permutacion normalizada, m=4, tau=1, sobre cierres, ventana 100d).
  Modelos (Ridge alpha=1000, target rank h, 6 folds purgados por fechas, identicos a N1-B; entrada Open t+1):
    base = ranks40+rev1d+rev1d*last5 (rank_aug)   | micro = base+{cs,amihud,gk,pk,gaprng}
    mem  = base+{hurst,pe}                         | all   = base+7 features        ; h in {5, 21}
  (i) DELTA vs base. "Material" = TODAS: dIC medio >= +0.005, t(NW) del dIC pareado por fecha > 2, dIC>0 en >=4/6 folds.
      Alfa: portafolio largo tercil superior, rebalanceo cada h, costo por emisora de n2b (c_1x, sin filtro) y 0.30% fijo;
      reporta alfa neto, folds+, 2024+, tercil liquido (T3).
  (ii) GATE por terciles de hurst100 / pe100: (a) transversal (tercil de la emisora en la fecha) y (b) de mercado (media
      transversal del feature, tercil por cuantiles EXPANDING con shift 1, >=250 fechas). Score = 'all' y base.
      IC condicional por tercil. "Regimen util" = TODAS: IC >= 0.05, t(NW) >= 2.9 (Bonferroni 0.05/12 celdas por h),
      IC>0 en >=5/6 folds, IC 2024+ > 0. Alfa neto (costo por emisora) dentro del tercil como evidencia adicional.
  Metricas generales: IC por fecha con t(NW, lag=max(5,h)), alfa neto %/rebalanceo, folds con alfa+, 2024+, T3, DSR con N global
  real del log (trial_stats.deflated_sharpe, serie neta de la cartera tercil con costo por emisora).
  CRITERIO GLOBAL: exito = t(NW) del IC > 2 Y alfa neto > 0 en >=4/6 folds (costo por emisora), O DSR > 0.95; hallazgo solo si
  ademas el efecto incremental vs base es "material" segun (i) o (ii). Cada configuracion se registra con log_trial (n3_*).

Ejecucion: .venv/bin/python scripts/n3_prueba.py   (un solo proceso, float32, sin paralelismo)
"""
from __future__ import annotations

import gc
import json
import math
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
import scripts.n1a_leadlag as N
import scripts.n1b_target as T
import scripts.n2b_costos as C
from src.data.store import get as store_get
from src.ml.features import FEATURE_COLS
from src.ml.trial_stats import deflated_sharpe
from src.ml.trials import DEFAULT_PATH as TRIALS_PATH, count_trials, log_trial

OUT = ROOT / "data" / "eval"
SCRATCH = T.SCRATCH
HS = (5, 21)
FEATS_MICRO = ["cs60", "amihud60", "gk60", "pk60", "gaprng60"]
FEATS_MEM = ["hurst100", "pe100"]
ALLF = FEATS_MICRO + FEATS_MEM
D_IC_MIN, D_T_MIN, GATE_IC_MIN, GATE_T_MIN = 0.005, 2.0, 0.05, 2.9
PERM_M, PERM_W, HURST_W = 4, 100, 100


# ─── features OHLCV ───────────────────────────────────────────────────────────

def perm_entropy(c: pd.Series, w: int = PERM_W, m: int = PERM_M) -> pd.Series:
    v = c.to_numpy("float64")
    n = len(v)
    out = np.full(n, np.nan, dtype="float32")
    if n < w + m:
        return pd.Series(out, index=c.index)
    win = np.lib.stride_tricks.sliding_window_view(v, m)                 # (n-m+1, m)
    ok = np.isfinite(win).all(axis=1)
    rk = np.argsort(np.argsort(np.where(ok[:, None], win, 0.0), axis=1, kind="stable"), axis=1)
    code = (rk * (m ** np.arange(m))).sum(axis=1)
    uniq = np.unique(code[ok])
    idx = np.searchsorted(uniq, code)
    oh = np.zeros((len(code), len(uniq)), dtype="float32")
    oh[np.arange(len(code))[ok], idx[ok]] = 1.0
    oh_s = pd.DataFrame(oh).rolling(w - m + 1, min_periods=w - m + 1).mean().to_numpy()
    okr = pd.Series(ok.astype("float32")).rolling(w - m + 1).min().to_numpy() > 0
    p = np.clip(oh_s, 1e-12, None)
    ent = -(np.where(oh_s > 0, p * np.log(p), 0.0)).sum(axis=1) / np.log(float(math.factorial(m)))
    ent = np.where(okr, ent, np.nan)
    out[m - 1:] = ent.astype("float32")                                  # patron termina en t
    return pd.Series(out, index=c.index)


def hurst(c: pd.Series, w: int = HURST_W, ks=(1, 2, 4, 8, 16)) -> pd.Series:
    lp = np.log(c.where(c > 0))
    lx = np.log(np.array(ks, float))
    xc = lx - lx.mean()
    lv = np.column_stack([np.log(lp.diff(k).rolling(w, min_periods=int(w * 0.8)).var().replace(0, np.nan)).to_numpy()
                          for k in ks])
    return pd.Series((lv * xc).sum(axis=1) / (xc ** 2).sum() / 2.0, index=c.index).astype("float32")


def ohlcv_features(tk: str) -> pd.DataFrame | None:
    raw = store_get(N.BMV_YFINANCE.get(tk, tk))
    if raw is None or len(raw) < 150:
        return None
    raw = raw[~raw.index.duplicated()].sort_index()
    raw.index = pd.to_datetime(raw.index)
    o, h, l, c, v = (raw[k].astype("float64") for k in ("Open", "High", "Low", "Close", "Volume"))
    r = c.pct_change()
    dv = c * v
    lhl = np.log(h / l)
    gk = 0.5 * lhl ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    pk = lhl ** 2 / (4 * np.log(2))
    gap = np.log(o / c.shift(1)).abs()
    am = (r.abs() / dv.where(dv > 0)).replace([np.inf, -np.inf], np.nan)
    x = pd.DataFrame(index=raw.index)
    x["cs60"] = B.corwin_schultz(h, l).rolling(60, min_periods=30).mean()
    x["amihud60"] = np.log(am.rolling(60, min_periods=30).mean() + 1e-12)
    x["gk60"] = np.sqrt(gk.clip(lower=0).rolling(60, min_periods=30).mean())
    x["pk60"] = np.sqrt(pk.rolling(60, min_periods=30).mean())
    x["gaprng60"] = (gap / lhl.where(lhl > 0)).replace([np.inf, -np.inf], np.nan).rolling(60, min_periods=30).mean()
    x["hurst100"] = hurst(c)
    x["pe100"] = perm_entropy(c)
    x = x.astype("float32")
    x["ticker"] = tk
    x.index.name = "date"
    return x.reset_index()


def build_micro(tickers) -> pd.DataFrame:
    fr = [f for f in (ohlcv_features(t) for t in tickers) if f is not None]
    return pd.concat(fr, ignore_index=True)


# ─── estadistica ──────────────────────────────────────────────────────────────

def ic_stats(oos: pd.DataFrame, h: int, min_names: int = 6) -> dict:
    ic = T.ic_by_date(oos, min_names)
    if len(ic) < 20:
        return {"ic": None, "t": None, "n": int(len(ic)), "by_fold": {}, "ic24": None}
    lag = max(A.NW_LAG, h)
    m, _, t = A.nw_t(ic.values, lag)
    byf = {}
    for fo, z in oos.groupby("fold"):
        i = T.ic_by_date(z, min_names)
        byf[int(fo)] = float(i.mean()) if len(i) else None
    post = ic[ic.index >= T.TEST_START]
    m24 = float(post.mean()) if len(post) > 10 else None
    return {"ic": float(m), "t": float(t), "n": int(len(ic)), "by_fold": byf,
            "folds_pos": int(sum(1 for v in byf.values() if v and v > 0)), "ic24": m24, "_series": ic}


def clean_ic(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "_series"}


def ridge_scores(df, X, h):
    return T.ridge_oos(df, X, f"t_rank{h}", h)


def attach_gates(oos: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    return oos.merge(g, on=["date", "ticker"], how="left")


def alpha_block(d: pd.DataFrame, h: int, fixed: float | None, ccol="c_1x", min_names=A.MIN_NAMES, t3=False):
    a = C.run(d, h, "tercil", ccol, None, min_names=min_names, fixed=fixed, t3=t3)
    r, sr = C.summarize(a, d, h)
    if not r:
        return {}, np.array([])
    keep = ("n_rebal", "gross_pct", "cost_pct", "alpha_net_pct", "alpha_net_t", "turnover", "folds_pos", "n_folds",
            "alpha_net_2024p_pct", "ic", "t_ic_nw", "alpha_by_fold", "sharpe_net")
    return {k: r.get(k) for k in keep}, sr


def f(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main() -> None:
    df = A.load_panel(SCRATCH / "panel.parquet")
    cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df[np.isfinite(df["ret_fwd"])].reset_index(drop=True)
    df = df.merge(T.build_extra(df), on=["date", "ticker"], how="left")
    gc.collect()
    df = T.add_targets(df)
    df = T.calendar_cols(df)
    df["rev1d"] = -df["ret1d"]
    tick = sorted(df["ticker"].unique())
    print(f"Panel {len(df):,} filas, {len(tick)} tickers", flush=True)

    mic = build_micro(tick)
    df = df.merge(mic, on=["date", "ticker"], how="left")
    cov = {c: float(df[c].notna().mean()) for c in ALLF}
    print("cobertura features:", cov, flush=True)
    # features cross-sectional ranks (NaN -> 0 = mediana) + gates
    Xb = T.rank_features(df, cols)
    rr = (df.groupby("date")["rev1d"].rank(pct=True) - 0.5).fillna(0.0).to_numpy("float32")
    Xbase = np.column_stack([Xb, rr, rr * df["last5"].to_numpy("float32")]).astype("float32")
    del Xb
    FR = {c: T.rank_features(df, [c])[:, 0] for c in ALLF}
    Xs = {"base": Xbase,
          "micro": np.column_stack([Xbase] + [FR[c] for c in FEATS_MICRO]).astype("float32"),
          "mem": np.column_stack([Xbase] + [FR[c] for c in FEATS_MEM]).astype("float32"),
          "all": np.column_stack([Xbase] + [FR[c] for c in ALLF]).astype("float32")}
    # gates: tercil transversal y regimen de mercado (cuantiles expanding, shift 1)
    gate = df[["date", "ticker"]].copy()
    for v in ("hurst100", "pe100"):
        rk = df.groupby("date")[v].rank(pct=True)
        gate[f"x_{v}"] = pd.cut(rk, [0, 1 / 3, 2 / 3, 1.0001], labels=["T1", "T2", "T3"]).astype(object)
        mk = df.groupby("date")[v].mean()
        q1 = mk.expanding(250).quantile(1 / 3).shift(1)
        q2 = mk.expanding(250).quantile(2 / 3).shift(1)
        reg = pd.Series(np.where(mk <= q1, "T1", np.where(mk <= q2, "T2", "T3")), index=mk.index).where(q1.notna())
        gate[f"m_{v}"] = df["date"].map(reg).values
    liq_cols = ["date", "ticker", "adv", "c_1x"]
    cost = C.cost_panel(tick)
    cost["c_1x"] = C.side_cost(cost)
    cost = cost[["date", "ticker", "adv", "c_1x"]]
    meta_df = df[["date", "ticker"]].copy()
    n0 = count_trials()
    R = {"meta": {"prueba": "B", "cobertura_features": cov, "n_trials_inicio": n0, "ridge_alpha": T.RIDGE_ALPHA,
                  "umbrales": {"dIC_min": D_IC_MIN, "dIC_t_min": D_T_MIN, "gate_ic_min": GATE_IC_MIN, "gate_t_min": GATE_T_MIN},
                  "cobertura_eventos": "ver docstring: insuficiente para Prueba A"},
         "cfg": {}, "gate": {}}
    series, scores = {}, {}

    def logt(name, cfg, res, n):
        log_trial(f"n3_{name}", {"tanda": "n3", **cfg}, B.clean(res), int(n), path=str(ROOT / TRIALS_PATH))

    for h in HS:
        print(f"\n=== h={h} ===", flush=True)
        base_ic = None
        for nm in ("base", "micro", "mem", "all"):
            oos = ridge_scores(df, Xs[nm], h)
            scores[(nm, h)] = oos
            st = ic_stats(oos, h)
            d = C.prep(attach_gates(oos, gate), cost, h)
            rv, sr = alpha_block(d, h, None)
            rf, _ = alpha_block(d, h, C.FIXED)
            rv3, _ = alpha_block(d, h, None, min_names=B.MIN_CELL, t3=True)
            rf3, _ = alpha_block(d, h, C.FIXED, min_names=B.MIN_CELL, t3=True)
            ent = {"h": h, "modelo": nm, "ic": clean_ic(st), "var": rv, "fix030": rf, "var_T3": rv3, "fix030_T3": rf3}
            if nm == "base":
                base_ic = st
            else:
                a, b = base_ic["_series"].align(st["_series"], join="inner")
                dd = (b - a).values
                m, _, t = A.nw_t(dd, max(A.NW_LAG, h))
                dfold = {fo: (None if st["by_fold"].get(fo) is None or base_ic["by_fold"].get(fo) is None
                              else st["by_fold"][fo] - base_ic["by_fold"][fo]) for fo in base_ic["by_fold"]}
                npos = sum(1 for v in dfold.values() if v is not None and v > 0)
                bv = R["cfg"][f"base_h{h}"]["var"]
                ent["delta"] = {"dIC": float(m), "t_nw": float(t), "dIC_by_fold": dfold, "folds_dIC_pos": npos,
                                "d_alpha_var_pct": (rv.get("alpha_net_pct") or 0) - (bv.get("alpha_net_pct") or 0),
                                "material": bool(m >= D_IC_MIN and t > D_T_MIN and npos >= 4)}
            R["cfg"][f"{nm}_h{h}"] = ent
            series[f"{nm}_h{h}"] = sr
            print(f"  {nm}: IC={f(st['ic'],4)} t={f(st['t'],2)} f+={st.get('folds_pos')}/6 IC24={f(st['ic24'],4)} | "
                  f"var aN={f(rv.get('alpha_net_pct'))} f+={rv.get('folds_pos')} 24+={f(rv.get('alpha_net_2024p_pct'))} T3={f(rv3.get('alpha_net_pct'))} | "
                  f"fix aN={f(rf.get('alpha_net_pct'))} f+={rf.get('folds_pos')} T3={f(rf3.get('alpha_net_pct'))}"
                  + (f" | dIC={f(ent['delta']['dIC'],4)} t={f(ent['delta']['t_nw'],2)} npos={ent['delta']['folds_dIC_pos']}" if "delta" in ent else ""), flush=True)
            for cn, rr_ in (("c1x", rv), ("fix030", rf)):
                logt(f"modelo_{nm}_h{h}_{cn}", {"h": h, "modelo": nm, "costo": cn, "ridge_alpha": T.RIDGE_ALPHA, "n_splits": A.N_SPLITS,
                                                "embargo": A.EMBARGO, "entry": "next_open"},
                     {"ic": st["ic"], "t_ic": st["t"], "alpha_net_pct": rr_.get("alpha_net_pct"), "folds_pos": rr_.get("folds_pos")}, len(oos))
            json.dump(B.clean(R), open(OUT / "n3_prueba.json", "w"), indent=2, ensure_ascii=False, default=str)
            gc.collect()

        # (ii) gates
        for nm in ("all", "base"):
            oos = attach_gates(scores[(nm, h)], gate)
            for v in ("hurst100", "pe100"):
                for kind in ("x", "m"):
                    gcol = f"{kind}_{v}"
                    for lab in ("T1", "T2", "T3"):
                        z = oos[oos[gcol] == lab]
                        mn = B.MIN_CELL if kind == "x" else 6
                        st = ic_stats(z, h, mn) if len(z) > 300 else {"ic": None, "t": None, "by_fold": {}, "n": 0}
                        rv = {}
                        if len(z) > 300:
                            dz = C.prep(z.drop(columns=["x_hurst100", "x_pe100", "m_hurst100", "m_pe100"]), cost, h)
                            rv, _ = alpha_block(dz, h, None, min_names=mn)
                        ok = bool(st["ic"] is not None and st["ic"] >= GATE_IC_MIN and st["t"] is not None and st["t"] >= GATE_T_MIN
                                  and st.get("folds_pos", 0) >= 5 and st.get("ic24") is not None and st["ic24"] > 0)
                        key = f"{nm}_h{h}_{gcol}_{lab}"
                        R["gate"][key] = {"h": h, "score": nm, "var": v, "tipo": "transversal" if kind == "x" else "mercado", "tercil": lab,
                                          "n_filas": int(len(z)), "ic": clean_ic(st), "alpha": rv, "regimen_util": ok}
                        logt(f"gate_{key}", {"h": h, "score": nm, "var": v, "tipo": kind, "tercil": lab, "ridge_alpha": T.RIDGE_ALPHA,
                                             "entry": "next_open"},
                             {"ic": st["ic"], "t_ic": st["t"], "alpha_net_pct": rv.get("alpha_net_pct")}, len(z))
                        print(f"  gate {key}: n={len(z)} IC={f(st['ic'],4)} t={f(st['t'],2)} f+={st.get('folds_pos')} "
                              f"IC24={f(st.get('ic24'),4)} aN={f(rv.get('alpha_net_pct'))} f+={rv.get('folds_pos')} util={ok}", flush=True)
                        del z
                    gc.collect()
            json.dump(B.clean(R), open(OUT / "n3_prueba.json", "w"), indent=2, ensure_ascii=False, default=str)
            del oos
            gc.collect()

    n_glob = count_trials()
    R["meta"]["n_trials_final"] = n_glob
    for k, ent in R["cfg"].items():
        per = max(1, int(round(252 / ent["h"])))
        ent["dsr_global"] = B.clean(deflated_sharpe(series[k], n_glob, periods=per))
    json.dump(B.clean(R), open(OUT / "n3_prueba.json", "w"), indent=2, ensure_ascii=False, default=str)
    write_md(R)
    print("guardado; N global =", n_glob, flush=True)


def write_md(R: dict) -> None:
    m = R["meta"]
    L = ["# N3 Prueba B: features OHLCV de microestructura/memoria", "",
         "Cobertura de eventos (Paso 1): insuficiente para Prueba A (ver docstring de scripts/n3_prueba.py): noticias por emisora solo "
         "WALMEX 2018-2020 (~900 titulares); archivo PIT con 304 de 556 titulares en 2026; flujo (hechos) desde 2025-06-30 y no es de "
         "hechos relevantes; EPS solo emisoras US; events.csv solo macro (banxico/fomc/inpc).", "",
         f"- Umbrales pre-registrados: {m['umbrales']}. N global de ensayos: inicio {m['n_trials_inicio']}, final {m.get('n_trials_final')}.",
         f"- Cobertura de features (fraccion no-NaN del panel): {({k: round(v, 3) for k, v in m['cobertura_features'].items()})}", "",
         "## (i) Modelos: IC y alfa neto (tercil superior, rebalanceo cada h)", "",
         "| h | modelo | IC (t NW) | folds IC+ | dIC (t) | dIC folds+ | material | aN var % | f+ | 2024+ | T3 var | aN fix030 % | f+ | T3 fix | DSR |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, e in R["cfg"].items():
        i, v, x = e["ic"], e["var"], e["fix030"]
        dl = e.get("delta", {})
        L.append(f"| {e['h']} | {e['modelo']} | {f(i['ic'],4)} ({f(i['t'],2)}) | {i.get('folds_pos')}/6 | "
                 f"{f(dl.get('dIC'),4)} ({f(dl.get('t_nw'),2)}) | {dl.get('folds_dIC_pos','-')} | {dl.get('material','-')} | "
                 f"{f(v.get('alpha_net_pct'))} | {v.get('folds_pos')}/6 | {f(v.get('alpha_net_2024p_pct'))} | {f(e['var_T3'].get('alpha_net_pct'))} | "
                 f"{f(x.get('alpha_net_pct'))} | {x.get('folds_pos')}/6 | {f(e['fix030_T3'].get('alpha_net_pct'))} | {f((e.get('dsr_global') or {}).get('dsr'))} |")
    L += ["", "## (ii) Gate por terciles de Hurst / entropia de permutacion", "",
          "| h | score | variable | tipo | tercil | n | IC (t) | folds IC+ | IC 2024+ | aN % (costo emisora) | f+ | util |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, g in R["gate"].items():
        i = g["ic"]
        L.append(f"| {g['h']} | {g['score']} | {g['var']} | {g['tipo']} | {g['tercil']} | {g['n_filas']} | {f(i.get('ic'),4)} ({f(i.get('t'),2)}) | "
                 f"{i.get('folds_pos','-')} | {f(i.get('ic24'),4)} | {f(g['alpha'].get('alpha_net_pct'))} | {g['alpha'].get('folds_pos','-')} | {g['regimen_util']} |")
    (OUT / "n3_prueba.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
