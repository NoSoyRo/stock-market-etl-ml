"""
Nivel 2 / idea B: COSTO POR EMISORA (en lugar de 0.30 %/lado constante) y portafolio consciente de costos.

PRE-REGISTRO (fijado ANTES de correr; no se ajusta despues)
-----------------------------------------------------------
Hipotesis: el costo real (spread + impacto) es muy heterogeneo; operando solo emisoras baratas y rotando poco,
la senal Ridge-rank de N1-B (IC~0.03, T3~0.06) podria dar alfa neto > 0.
Senal: scores OOS Ridge rank_aug de N1-B/N1-C (cache n1c_oos_rankaug_h{5,21}.parquet; 6 folds por fechas, F6 = el mas
antiguo en la etiqueta del cache, 2017-07..2019-02; F1 = el mas reciente). NO se reentrena nada.

Costo por lado por (emisora, t), solo datos <= t-1 (shift 1):
  S      = mediana-promedio de {Corwin-Schultz 60d, Roll 60d}  (spread completo, en fraccion)
  imp    = 0.5 * Amihud_60d * Q,   Q = 1 000 000 MXN por nombre y por operacion (supuesto de tamano retail/pequeno)
  c_side = max(PISO, COM + MULT * (S/2 + imp)),  tope 1.0 %
  COM = 0.10 % (comision+derechos+IVA aprox., supuesto), PISO = 0.15 %, MULT = 1 (principal; sensibilidad 0.5 y 2;
  COM 0.20 %). Tambien se calculan %dias con ret 0 / volumen 0 y ADV (diagnostico y filtro de liquidez; no entran al costo).
  Los parametros COM/PISO/Q/MULT son SUPUESTOS, no estimados de ejecuciones reales; no hay datos de ejecucion.
Filtro de universo: elegible si c_side < UMBRAL; UMBRAL = cuantil 67 % (variante q67) o 50 % (q50) de c_side agrupado en
fechas ANTERIORES al primer dia OOS (2017-07-21). Sin filtro = universo completo.

Prediccion de liquidez (paso 2): objetivos = log ADV y spread CS medios en t+1..t+h (h=5,21); baselines EWMA(span 20) y
mediana 20d; modelo = Ridge (log-features: ewma20, med20, media 60/250d, ultimo valor, %sin operacion, nivel de mercado),
pooled, entrenamiento expanding con fechas < inicio de fold - (h+5) dias. Metrica: MAE y skill = 1 - MAE_mod/MAE_base
(se reporta vs EWMA y vs mediana), en las fechas no solapadas (cada h) del test.

Portafolio (paso 3), rebalanceo cada h in {5, 21} fechas por fold, igual-ponderado, solo largos, alfa = exceso del
portafolio sobre la media del universo elegible (como N1-C, el benchmark no paga costo; es un alfa RELATIVO):
  tercil     : tercil superior del score
  top10_b10 / top5_b5 : top-k con buffer/histeresis (se conserva mientras rank < k+b)
  ca10 / ca5 : top-k que penaliza turnover costoso: u_i = E_i + (c_i si ya se tiene, -c_i si no), con
               E_i = 0.03 * sigma_h * z_i (IC0 = 0.03 de 1B; sigma_h = desv. cross-section del ret_h en las 250 fechas previas;
               z_i = score estandarizado por fecha); se eligen los k mayores u_i.
  costo neto = suma por compra/venta de |peso| * c_i (c_i por emisora, o 0.30 % fijo en baseline). Rebalanceo inicial: solo compra.
Baseline: misma senal, costo fijo 0.30 %/lado, universo completo (tercil, top10_b10, top5_b5).

Metricas: alfa neto %/rebalanceo (con t), Sharpe neto anualizado (media/desv de los excesos netos por rebalanceo * sqrt(252/h);
es Sharpe RELATIVO al universo), turnover, folds con alfa+ (criterio >= 4/6), alfa 2024+, tercil liquido (ADV, top 1/3 por fecha),
IC por fecha (t Newey-West) del score en el universo elegible, DSR con N global del log (src/ml/trial_stats.deflated_sharpe).
CRITERIO GLOBAL: exito = t(NW) del IC > 2 Y alfa neto > 0 en la mayoria (>= 4/6) de folds, O DSR > 0.95.
Cada configuracion se registra con log_trial (familias n2b_*).

Ejecucion: .venv/bin/python scripts/n2b_costos.py
"""
from __future__ import annotations

import gc
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import scripts.n0a_diagnostico as A
import scripts.n0b_diagnostico as B
import scripts.n1b_target as T
from src.data.store import get as store_get
from src.ml.trial_stats import deflated_sharpe
from src.ml.trials import DEFAULT_PATH as TRIALS_PATH, count_trials, log_trial

OUT = ROOT / "data" / "eval"
SCRATCH = T.SCRATCH
WIN = 60
Q_MXN = 1_000_000.0
COM, FLOOR, CAP = 0.10, 0.15, 1.0     # % por lado
FIXED = 0.30
IC0 = 0.03
SIG_WIN, MIN_PREV = 250, 60
OOS_START = pd.Timestamp("2017-07-21")
PORTS = {"tercil": ("topk", None, None), "top10_b10": ("topk", 10, 10), "top5_b5": ("topk", 5, 5),
         "ca10": ("ca", 10, 10), "ca5": ("ca", 5, 5)}


# ─── 1. proxies de costo point-in-time ────────────────────────────────────────

def daily_proxies(tk: str) -> pd.DataFrame | None:
    raw = store_get(tk)
    if raw is None or len(raw) < WIN + 5:
        return None
    raw = raw[~raw.index.duplicated()].sort_index()
    raw.index = pd.to_datetime(raw.index)
    c = raw["Close"].astype("float64")
    r = c.pct_change()
    dv = c * raw["Volume"].astype("float64")
    cs = B.corwin_schultz(raw["High"].astype("float64"), raw["Low"].astype("float64"))
    cov = r.rolling(WIN, min_periods=40).cov(r.shift(1))
    roll = 2 * np.sqrt((-cov).clip(lower=0))
    am = (r.abs() / dv.where(dv > 0)).replace([np.inf, -np.inf], np.nan)
    nop = ((raw["Volume"] == 0) | (r == 0)).astype(float)
    return pd.DataFrame({"r": r, "dv": dv, "cs_d": cs, "roll": roll, "am_d": am, "nop_d": nop})


def cost_panel(tickers) -> pd.DataFrame:
    """Por (ticker, date): proxies con datos hasta t-1 (shift 1) y componentes del costo (fracciones, no %)."""
    rows = []
    for tk in tickers:
        d = daily_proxies(tk)
        if d is None:
            continue
        x = pd.DataFrame({
            "adv": d["dv"].rolling(WIN, min_periods=30).mean(),
            "cs": d["cs_d"].rolling(WIN, min_periods=30).mean(),
            "roll": d["roll"],
            "amihud": d["am_d"].rolling(WIN, min_periods=30).mean(),
            "nop": d["nop_d"].rolling(WIN, min_periods=30).mean()}).shift(1)
        x["ticker"] = tk
        x.index.name = "date"
        rows.append(x.reset_index())
    p = pd.concat(rows, ignore_index=True)
    p["S"] = p[["cs", "roll"]].mean(axis=1)                      # spread completo (fraccion)
    p["half"] = p["S"] / 2
    p["imp"] = 0.5 * p["amihud"] * Q_MXN
    return p


def side_cost(p: pd.DataFrame, mult: float = 1.0, com: float = COM, floor: float = FLOOR) -> pd.Series:
    """Costo por lado en % (NaN si faltan proxies)."""
    var = (p["half"] + p["imp"]) * 100.0
    c = (com + mult * var).clip(upper=CAP)
    return np.maximum(c, floor).where(var.notna())


# ─── 2. prediccion de liquidez ────────────────────────────────────────────────

def liquidity_prediction(tickers, folds: dict, hs=(5, 21)) -> dict:
    series = {}
    for tk in tickers:
        d = daily_proxies(tk)
        if d is not None:
            series[tk] = d
    res = {}
    for tgt in ("adv", "spread"):
        for h in hs:
            frames = []
            for tk, d in series.items():
                raw = d["dv"] if tgt == "adv" else d["cs_d"] * 100.0
                v = np.log(raw.clip(lower=1.0)) if tgt == "adv" else np.log(raw + 0.05)   # %-puntos + 0.05
                f = pd.DataFrame(index=d.index)
                f["y"] = v.rolling(h).mean().shift(-h)
                f["ew20"] = v.ewm(span=20, min_periods=10).mean()
                f["med20"] = v.rolling(20, min_periods=10).median()
                f["m60"] = v.rolling(60, min_periods=30).mean()
                f["m250"] = v.rolling(250, min_periods=120).mean()
                f["last"] = v
                f["nop60"] = d["nop_d"].rolling(60, min_periods=30).mean()
                f["ticker"] = tk
                frames.append(f.reset_index().rename(columns={"index": "date", "Date": "date"}))
            P = pd.concat(frames, ignore_index=True)
            P["date"] = pd.to_datetime(P["date"])
            P["mkt"] = P.groupby("date")["ew20"].transform("mean")
            P = P.dropna().reset_index(drop=True)
            feats = ["ew20", "med20", "m60", "m250", "last", "nop60", "mkt"]
            rows = []
            for fo, (a, b) in folds.items():
                tr = P[P["date"] < a - pd.Timedelta(days=int((h + 5) * 1.5))]
                te = P[(P["date"] >= a) & (P["date"] <= b)]
                dts = np.sort(te["date"].unique())[::h]
                te = te[te["date"].isin(set(dts))]
                if len(tr) < 2000 or te.empty:
                    continue
                # el modelo aprende el residuo respecto al EWMA (baseline)
                m = Ridge(alpha=10.0).fit(tr[feats].to_numpy("float32"), (tr["y"] - tr["ew20"]).to_numpy("float32"))
                pred = te["ew20"].to_numpy() + m.predict(te[feats].to_numpy("float32"))
                y = te["y"].to_numpy()
                rows.append({"fold": fo, "n": len(te), "mae_model": float(np.mean(np.abs(y - pred))),
                             "mae_ewma": float(np.mean(np.abs(y - te["ew20"]))),
                             "mae_med": float(np.mean(np.abs(y - te["med20"])))})
                del tr, te
            R = pd.DataFrame(rows)
            w = R["n"]
            agg = {k: float(np.average(R[k], weights=w)) for k in ("mae_model", "mae_ewma", "mae_med")}
            agg["skill_vs_ewma"] = 1 - agg["mae_model"] / agg["mae_ewma"]
            agg["skill_vs_med"] = 1 - agg["mae_model"] / agg["mae_med"]
            agg["folds_mejora_vs_ewma"] = int((R["mae_model"] < R["mae_ewma"]).sum())
            agg["n_folds"] = int(len(R))
            agg["n"] = int(w.sum())
            res[f"{tgt}_h{h}"] = agg
            del P, frames
            gc.collect()
    return res


# ─── 3. portafolio consciente de costos ───────────────────────────────────────

def prep(oos: pd.DataFrame, cost: pd.DataFrame, h: int) -> dict:
    """oos + costo punto-a-tiempo (varios multiplicadores se agregan despues con add_cost)."""
    d = oos.merge(cost, on=["date", "ticker"], how="left")
    d = d.sort_values(["date", "ticker"]).reset_index(drop=True)
    d["z"] = (d["score"] - d.groupby("date")["score"].transform("mean")) / d.groupby("date")["score"].transform("std")
    # sigma_h cross-section previa (250 fechas) para el modelo de alfa esperado
    sd = d.groupby("date")["ret"].std()
    sig = sd.rolling(SIG_WIN, min_periods=MIN_PREV).mean().shift(h + 1)
    d["sig"] = d["date"].map(sig)
    d["rk"] = d.groupby("date")["score"].rank(pct=True)
    d["adv_rk"] = d.groupby("date")["adv"].rank(pct=True)
    return d


def run(d: pd.DataFrame, h: int, port: str, ccol: str | None, thr: float | None, min_names: int = A.MIN_NAMES,
        fixed: float | None = None, t3: bool = False) -> pd.DataFrame:
    """ccol: columna de costo por lado (%) por emisora; fixed: costo constante (anula ccol); thr: filtro c<thr."""
    kind, k0, b0 = PORTS[port]
    x = d
    if t3:
        x = x[x["adv_rk"] > 2 / 3]
    if thr is not None:
        x = x[x[ccol] < thr]
    rows = []
    for fo, zf in x.groupby("fold"):
        fdates = np.sort(zf["date"].unique())[::h]
        grp = {dt: g for dt, g in zf.groupby("date")}
        prev: set = set()
        first = True
        for dt in fdates:
            w = grp[dt]
            if len(w) < min_names:
                continue
            cmap = (pd.Series(fixed, index=w["ticker"]) if fixed is not None
                    else w.set_index("ticker")[ccol]).fillna(FIXED if fixed is None else fixed)
            if port == "tercil":
                cur = set(w.loc[w["rk"] > 2 / 3, "ticker"])
            else:
                k = int(min(k0, max(1, len(w) // 3)))
                if kind == "topk":
                    order = w.sort_values("score", ascending=False)["ticker"].tolist()
                    rank = {t_: j for j, t_ in enumerate(order)}
                    new = set(order[:k])
                    if prev:
                        keep = sorted([t_ for t_ in prev if t_ in rank and rank[t_] < k + b0], key=lambda t_: rank[t_])[:k]
                        fill = [t_ for t_ in order if t_ not in keep]
                        new = set(keep) | set(fill[: k - len(keep)])
                    cur = new
                else:                                           # ca: penaliza turnover costoso
                    sg = w["sig"].iloc[0]
                    if not np.isfinite(sg):
                        continue
                    E = IC0 * sg * w["z"].to_numpy()
                    c = cmap.reindex(w["ticker"]).to_numpy()
                    held = np.array([t_ in prev for t_ in w["ticker"]])
                    u = E + np.where(held, c, -c)
                    cur = set(w["ticker"].to_numpy()[np.argsort(-u)[:k]])
            if not cur:
                continue
            buys, sells = cur - prev, prev - cur
            wb, ws = 1.0 / len(cur), 1.0 / max(len(prev), 1)
            cost = wb * sum(cmap.get(t_, FIXED) for t_ in buys) + ws * sum(cmap.get(t_, FIXED) for t_ in sells)
            turn = (len(buys) / len(cur)) if not first else 1.0
            sel = w[w["ticker"].isin(cur)]
            ex = sel["ret"].mean() - w["ret"].mean()
            if np.isfinite(ex):
                rows.append((dt, fo, ex, turn, cost, len(cur), len(w), float(cmap.reindex(sel["ticker"]).mean())))
            prev, first = cur, False
    return pd.DataFrame(rows, columns=["date", "fold", "excess", "turnover", "cost", "n_inv", "n_univ", "c_hold"])


def summarize(a: pd.DataFrame, d: pd.DataFrame, h: int) -> tuple[dict, np.ndarray]:
    if a.empty:
        return {}, np.array([])
    net = a["excess"] - a["cost"]
    byf = net.groupby(a["fold"]).mean()
    post = (pd.to_datetime(a["date"]) >= T.TEST_START).values
    sdv = net.std(ddof=1)
    out = {"n_rebal": int(len(a)), "gross_pct": float(a["excess"].mean()), "cost_pct": float(a["cost"].mean()),
           "alpha_net_pct": float(net.mean()), "alpha_net_t": T.tstat(net.values),
           "sharpe_net": float(net.mean() / sdv * np.sqrt(252 / h)) if sdv > 0 else None,
           "turnover": float(a["turnover"].mean()), "n_inv": float(a["n_inv"].mean()), "n_univ": float(a["n_univ"].mean()),
           "c_hold_pct": float(a["c_hold"].mean()), "folds_pos": int((byf > 0).sum()), "n_folds": int(len(byf)),
           "alpha_by_fold": {int(k): float(v) for k, v in byf.items()},
           "alpha_net_2024p_pct": float(net[post].mean()) if post.any() else None}
    sub = d[d["date"].isin(set(a["date"]))]
    ic = T.ic_by_date(sub, 6)
    if len(ic) > 5:
        m, _, t = A.nw_t(ic.values, max(A.NW_LAG, h))
        out.update({"ic": m, "t_ic_nw": t})
    return out, net.to_numpy() / 100.0


def main() -> None:
    tick = sorted(pd.read_parquet(SCRATCH / "n1c_oos_rankaug_h5.parquet")["ticker"].unique())
    print("tickers", len(tick), flush=True)
    cost = cost_panel(tick)
    cost["c_1x"] = side_cost(cost)
    pre = cost[(cost["date"] < OOS_START) & (cost["date"] >= "2008-01-01")]["c_1x"].dropna()
    thr = {"q67": float(pre.quantile(0.67)), "q50": float(pre.quantile(0.50))}
    print("umbrales (pre-OOS)", thr, "N pre", len(pre), flush=True)
    for nm, (mu, co) in {"c_05x": (0.5, COM), "c_2x": (2.0, COM), "c_com20": (1.0, 0.20)}.items():
        cost[nm] = side_cost(cost, mu, co)
    # umbrales por escenario (cuantiles pre-OOS del propio escenario: igual regla, solo cambia la escala)
    thr_sc = {nm: {q: float(cost[(cost["date"] < OOS_START) & (cost["date"] >= "2008-01-01")][nm].dropna().quantile(v))
                   for q, v in (("q67", 0.67), ("q50", 0.50))} for nm in ("c_1x", "c_05x", "c_2x", "c_com20")}
    oos_ = cost[cost["date"] >= OOS_START]
    # tabla por emisora (ultimos 252 fechas hasta el final)
    last = cost[cost["date"] >= cost["date"].max() - pd.Timedelta(days=365)].groupby("ticker").agg(
        adv_mxn_M=("adv", lambda s: float(s.mean() / 1e6)), cs_pct=("cs", lambda s: float(s.mean() * 100)),
        roll_pct=("roll", lambda s: float(s.mean() * 100)), amihud_x1e9=("amihud", lambda s: float(s.mean() * 1e9)),
        nop_pct=("nop", lambda s: float(s.mean() * 100)), c_side_pct=("c_1x", "mean")).sort_values("c_side_pct")
    R = {"meta": {"Q_MXN": Q_MXN, "COM_pct": COM, "FLOOR_pct": FLOOR, "CAP_pct": CAP, "FIXED_pct": FIXED, "IC0": IC0,
                  "umbral_pre_oos_pct": thr, "umbral_por_escenario": thr_sc, "oos_start": str(OOS_START.date()),
                  "n_trials_global_inicio": count_trials(),
                  "dist_costo_oos_pct": {q: float(oos_["c_1x"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                  "frac_en_piso_oos": float((oos_["c_1x"] <= FLOOR + 1e-9).mean()),
                  "corr_spearman_proxies": {f"{a}~{b}": float(cost[[a, b]].dropna().corr("spearman").iloc[0, 1])
                                            for a, b in (("cs", "roll"), ("cs", "amihud"), ("nop", "cs"), ("adv", "cs"))}},
         "tabla_emisoras": last.round(4).reset_index().to_dict("records"), "cfg": {}}
    print("dist costo OOS", R["meta"]["dist_costo_oos_pct"], "piso", R["meta"]["frac_en_piso_oos"], flush=True)
    json.dump(B.clean(R), open(OUT / "n2b_costos.json", "w"), indent=2, ensure_ascii=False, default=str)

    # paso 2
    folds = {}
    o5 = pd.read_parquet(SCRATCH / "n1c_oos_rankaug_h5.parquet")
    for fo, g in o5.groupby("fold"):
        folds[int(fo)] = (g["date"].min(), g["date"].max())
    R["liquidez_pred"] = B.clean(liquidity_prediction(tick, folds))
    print("pred liquidez", json.dumps(R["liquidez_pred"], indent=1), flush=True)
    json.dump(B.clean(R), open(OUT / "n2b_costos.json", "w"), indent=2, ensure_ascii=False, default=str)
    del o5
    gc.collect()

    # paso 3
    series, results = {}, R["cfg"]
    for h in (5, 21):
        oos = pd.read_parquet(SCRATCH / f"n1c_oos_rankaug_h{h}.parquet")
        d = prep(oos, cost[["date", "ticker", "adv", "c_1x", "c_05x", "c_2x", "c_com20"]], h)
        del oos
        # costo faltante (sin proxies): se trata como no elegible si hay filtro; sin filtro se usa el fijo 0.30
        jobs = []   # (nombre, port, ccol, filtro, fixed)
        for port in PORTS:
            if port in ("tercil", "top10_b10", "top5_b5"):
                jobs.append((f"base_fixed030_full_{port}", port, "c_1x", None, FIXED, "base"))
            for flt in ("none", "q67", "q50"):
                jobs.append((f"c1x_{flt}_{port}", port, "c_1x", flt, None, "c_1x"))
        for sc in ("c_05x", "c_2x", "c_com20"):
            for flt in ("none", "q67"):
                for port in ("top10_b10", "ca10"):
                    jobs.append((f"{sc}_{flt}_{port}", port, sc, flt, None, sc))
        for nm, port, ccol, flt, fixed, sc in jobs:
            if sc == "base":
                tv = None
            else:
                tv = None if flt == "none" else thr_sc[ccol][flt]
            dd = d if flt != "none" or sc == "base" else d
            a = run(dd, h, port, ccol, tv, fixed=fixed)
            r, sr = summarize(a, dd, h)
            if not r:
                continue
            if not (fixed is not None):
                a3 = run(dd, h, port, ccol, tv, min_names=B.MIN_CELL, fixed=fixed, t3=True)
                if len(a3):
                    r3, _ = summarize(a3, dd, h)
                    r["T3"] = {k: r3.get(k) for k in ("alpha_net_pct", "alpha_net_t", "turnover", "cost_pct", "folds_pos")}
            else:
                a3 = run(dd, h, port, ccol, tv, min_names=B.MIN_CELL, fixed=fixed, t3=True)
                if len(a3):
                    r3, _ = summarize(a3, dd, h)
                    r["T3"] = {k: r3.get(k) for k in ("alpha_net_pct", "alpha_net_t", "turnover", "cost_pct", "folds_pos")}
            full = f"h{h}_{nm}"
            series[full] = sr
            results[full] = {"h": h, "port": port, "escenario": sc, "filtro": flt, "umbral_pct": tv, **B.clean(r)}
            cfg = {"tanda": "n2b", "h": h, "port": port, "escenario": sc, "filtro": flt, "umbral": tv, "fixed": fixed,
                   "Q": Q_MXN, "com": COM, "floor": FLOOR, "ic0": IC0, "n_splits": A.N_SPLITS, "embargo": A.EMBARGO, "entry": "next_open"}
            log_trial(f"n2b_{full}", cfg, B.clean({k: r.get(k) for k in ("alpha_net_pct", "cost_pct", "turnover", "folds_pos", "sharpe_net")}),
                      int(r["n_rebal"]), path=str(ROOT / TRIALS_PATH))
            print(f"{full:36s} gross={r['gross_pct']:.3f} cost={r['cost_pct']:.3f} aN={r['alpha_net_pct']:.3f} t={r['alpha_net_t']:.2f} "
                  f"SR={r['sharpe_net'] if r['sharpe_net'] is None else round(r['sharpe_net'], 2)} turn={r['turnover']:.2f} "
                  f"n={r['n_inv']:.1f}/{r['n_univ']:.1f} f+={r['folds_pos']}/{r['n_folds']} 24+={r['alpha_net_2024p_pct']} "
                  f"IC={r.get('ic')} t={r.get('t_ic_nw')} T3={r.get('T3', {}).get('alpha_net_pct')}", flush=True)
            json.dump(B.clean(R), open(OUT / "n2b_costos.json", "w"), indent=2, ensure_ascii=False, default=str)
            gc.collect()
        del d
        gc.collect()
    n_glob = count_trials()
    R["meta"]["n_trials_global_final"] = n_glob
    for nm, v in series.items():
        per = max(1, int(round(252 / results[nm]["h"])))
        results[nm]["dsr_global"] = B.clean(deflated_sharpe(v, n_glob, periods=per))
    json.dump(B.clean(R), open(OUT / "n2b_costos.json", "w"), indent=2, ensure_ascii=False, default=str)
    write_md(R)
    print("guardado; N global =", n_glob, flush=True)


def f(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_md(R: dict) -> None:
    m = R["meta"]
    L = ["# N2-B costo por emisora y portafolio consciente de costos", "",
         f"- Supuestos de costo (NO estimados de ejecuciones reales): Q={m['Q_MXN']:.0f} MXN/operacion, comision {m['COM_pct']}%, piso {m['FLOOR_pct']}%, tope {m['CAP_pct']}%; costo lado = max(piso, com + mult*(S/2 + 0.5*Amihud*Q)).",
         f"- Umbral pre-OOS (costo/lado %): {m['umbral_pre_oos_pct']}. Distribucion del costo en OOS (pct 5/25/50/75/95): {m['dist_costo_oos_pct']}; fraccion en el piso: {f(m['frac_en_piso_oos'], 2)}.",
         f"- Correlaciones de Spearman entre proxies: {m['corr_spearman_proxies']}",
         f"- N global de ensayos: inicio {m['n_trials_global_inicio']}, final {m.get('n_trials_global_final')}.",
         "- Alfa = exceso del portafolio largo vs media del universo elegible (relativo; benchmark sin costo). Folds: etiqueta del cache (F6 = mas antiguo).", "",
         "## Costo por emisora (ultimos 12 meses, escenario 1x)", "", "| ticker | ADV MXN M | CS % | Roll % | Amihud e-9 | sin op % | costo/lado % |", "|---|---|---|---|---|---|---|"]
    for r in R["tabla_emisoras"]:
        L.append(f"| {r['ticker']} | {f(r['adv_mxn_M'], 1)} | {f(r['cs_pct'], 2)} | {f(r['roll_pct'], 2)} | {f(r['amihud_x1e9'], 2)} | {f(r['nop_pct'], 1)} | {f(r['c_side_pct'], 3)} |")
    L += ["", "## Prediccion de liquidez (MAE en log; skill = 1 - MAE_modelo/MAE_baseline)", "",
          "| objetivo | n | MAE modelo | MAE EWMA20 | MAE med20 | skill vs EWMA | skill vs med | folds mejora |", "|---|---|---|---|---|---|---|---|"]
    for k, v in R["liquidez_pred"].items():
        L.append(f"| {k} | {v['n']} | {f(v['mae_model'], 4)} | {f(v['mae_ewma'], 4)} | {f(v['mae_med'], 4)} | {f(v['skill_vs_ewma'], 4)} | {f(v['skill_vs_med'], 4)} | {v['folds_mejora_vs_ewma']}/{v['n_folds']} |")
    for h in (5, 21):
        L += ["", f"## Portafolio h={h}", "",
              "| config | bruto % | costo % | alfa neto % | t | Sharpe neto | turnover | n inv/univ | folds+ | 2024+ % | IC (t NW) | T3 neto % | DSR |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, v in R["cfg"].items():
            if v["h"] != h:
                continue
            L.append(f"| {k[len(f'h{h}_'):]} | {f(v['gross_pct'])} | {f(v['cost_pct'])} | {f(v['alpha_net_pct'])} | {f(v['alpha_net_t'], 2)} | {f(v.get('sharpe_net'), 2)} | "
                     f"{f(v['turnover'], 2)} | {f(v['n_inv'], 1)}/{f(v['n_univ'], 1)} | {v['folds_pos']}/{v['n_folds']} | {f(v.get('alpha_net_2024p_pct'))} | "
                     f"{f(v.get('ic'), 4)} ({f(v.get('t_ic_nw'), 2)}) | {f(v.get('T3', {}).get('alpha_net_pct'))} | {f((v.get('dsr_global') or {}).get('dsr'))} |")
    (OUT / "n2b_costos.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
