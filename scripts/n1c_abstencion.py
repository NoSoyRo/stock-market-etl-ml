"""
Nivel 1 / idea C1: abstencion / seleccion por confianza sobre el mejor modelo de 1B (Ridge ranks, target rank / resid).

Reusa scripts/n1b_target (panel, targets, ridge_oos, ic_by_date, estadistica). No edita codigo existente.
src/ml/meta_model.py es OTRA cosa (meta-labeling supervisado sobre triple-barrier, clasificador de 2o nivel): no se duplica.

SIN LEAKAGE (todas las reglas usan solo informacion previa al rebalanceo en la fecha i):
  * Dispersion/score: umbrales = cuantiles de scores de las 250 fechas OOS anteriores (el score no necesita etiqueta;
    se exigen >= 60 fechas previas, antes de eso se abstiene).
  * Conformal temporal: etiquetas solo de fechas <= i-(h+1) (ya cerradas), ventana de 250 fechas.
  * Ninguna regla mira el retorno del test.

Reglas
  base_tercil     : largo tercil superior, sin abstencion (replica 1B)
  gate_disp_q{q}  : opera solo si la dispersion cross-sectional del score hoy esta en el percentil >= q de su ventana previa
  name_conf_q{q}  : dentro del tercil superior, solo nombres cuyo score centrado supera el cuantil q de la ventana previa
  conf_a{a}_d{d}  : conformal mondrian: en la ventana previa, por decil de score, cuantil a de la etiqueta rank; se selecciona el nombre
                    si el cuantil inferior del decil supera al cuantil inferior no condicional en delta (heuristica, documentada)
  top{k}_b{b}     : cartera concentrada top-k con buffer/histeresis (se mantiene el nombre mientras su rank <= k+b)
Alfa = exceso de la cartera sobre media del universo (celda), rebalanceo cada h fechas por fold; neto = 2*0.30%*turnover.

Ejecucion: .venv/bin/python scripts/n1c_abstencion.py
"""
from __future__ import annotations

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
import scripts.n1b_target as T
from src.ml.features import FEATURE_COLS
from src.ml.trial_stats import deflated_sharpe
from src.ml.trials import DEFAULT_PATH as TRIALS_PATH, count_trials, log_trial

OUT = ROOT / "data" / "eval"
SCRATCH = T.SCRATCH
COST = A.COST_SIDE
WIN, MIN_PREV = 250, 60
BASES = (("rankaug", 5), ("rankaug", 21), ("resid", 5), ("resid", 21))


# ─── estrategia ───────────────────────────────────────────────────────────────

def run_strategy(oos: pd.DataFrame, h: int, rule: str, par: dict, min_names: int) -> pd.DataFrame:
    """Filas por rebalanceo: date, fold, excess (bruto %), turnover, n_inv, traded."""
    d = oos.sort_values(["date", "ticker"]).reset_index(drop=True)
    d["z"] = d["score"] - d.groupby("date")["score"].transform("mean")
    dates = np.sort(d["date"].unique())
    idx_of = {dt: i for i, dt in enumerate(dates)}
    grp = {dt: g for dt, g in d.groupby("date")}
    disp = np.array([grp[dt]["score"].std() for dt in dates])
    zs = [grp[dt]["z"].to_numpy() for dt in dates]
    ys = [grp[dt]["tgt"].to_numpy() for dt in dates]
    rows = []
    for fo, zf in d.groupby("fold"):
        fdates = np.sort(zf["date"].unique())[::h]
        prev = None
        for dt in fdates:
            w = grp[dt]
            if len(w) < min_names:
                continue
            i = idx_of[dt]
            r = w["score"].rank(pct=True)
            cand = w[r > 2 / 3]
            sel = cand
            if rule == "gate_disp":
                past = disp[max(0, i - WIN):i]
                if len(past) < MIN_PREV or not (np.mean(past <= disp[i]) >= par["q"]):
                    sel = cand.iloc[0:0]
            elif rule == "name_conf":
                lo = max(0, i - WIN)
                if i - lo < MIN_PREV:
                    sel = cand.iloc[0:0]
                else:
                    thr = np.quantile(np.concatenate(zs[lo:i]), par["q"])
                    sel = cand[cand["z"] > thr]
            elif rule == "conformal":
                hi = i - (h + 1)
                lo = max(0, hi - WIN)
                if hi - lo < MIN_PREV:
                    sel = cand.iloc[0:0]
                else:
                    zz, yy = np.concatenate(zs[lo:hi]), np.concatenate(ys[lo:hi])
                    ok = np.isfinite(yy)
                    zz, yy = zz[ok], yy[ok]
                    edges = np.quantile(zz, np.linspace(0, 1, 11)[1:-1])
                    b = np.digitize(zz, edges)
                    q_all = np.quantile(yy, par["a"])
                    lowb = np.array([np.quantile(yy[b == k], par["a"]) if (b == k).sum() >= 30 else -9 for k in range(10)])
                    keep = lowb - q_all > par["delta"]
                    sel = cand[keep[np.digitize(cand["z"].to_numpy(), edges)]]
            elif rule == "topk":
                k = int(min(par["k"], max(1, len(w) // 3)))
                bf = par["b"]
                order = w.sort_values("score", ascending=False)["ticker"].tolist()
                rank = {t_: j for j, t_ in enumerate(order)}
                new = set(order[:k])
                if prev and bf > 0:
                    keep = sorted([t_ for t_ in prev if t_ in rank and rank[t_] < k + bf], key=lambda t_: rank[t_])[:k]
                    fill = [t_ for t_ in order if t_ not in keep]
                    new = set(keep) | set(fill[: k - len(keep)])
                sel = w[w["ticker"].isin(new)]
            elif rule != "base":
                raise ValueError(rule)
            if rule in ("name_conf", "conformal", "gate_disp") and len(sel) < 3:
                sel = cand.iloc[0:0]
            cur = set(sel["ticker"])
            traded = len(cur) > 0
            ex = (sel["ret"].mean() - w["ret"].mean()) if traded else 0.0
            if prev is None:
                turn = 1.0 if cur else 0.0
            else:
                turn = 1.0 - len(cur & prev) / max(len(cur), len(prev), 1) if (cur or prev) else 0.0
            prev = cur
            if np.isfinite(ex):
                rows.append((dt, fo, ex, turn, len(cur), traded))
    return pd.DataFrame(rows, columns=["date", "fold", "excess", "turnover", "n_inv", "traded"])


def stats_from(a: pd.DataFrame, oos: pd.DataFrame, h: int):
    if a.empty:
        return {}, np.array([])
    net = a["excess"] - 2 * COST * a["turnover"]
    byf = net.groupby(a["fold"]).mean()
    post = (pd.to_datetime(a["date"]) >= T.TEST_START).values
    tr = a[a["traded"]]
    out = {"n_rebal": int(len(a)), "exposicion": float(a["traded"].mean()), "n_inv_medio": float(tr["n_inv"].mean()) if len(tr) else 0.0,
           "excess_gross_pct": float(a["excess"].mean()), "alpha_net_pct": float(net.mean()), "alpha_net_t": T.tstat(net.values),
           "alpha_net_cond_pct": float(net[a["traded"].values].mean()) if len(tr) else None,
           "turnover": float(a["turnover"].mean()), "folds_pos": int((byf > 0).sum()), "n_folds": int(len(byf)),
           "alpha_by_fold": {int(k): float(v) for k, v in byf.items()},
           "alpha_net_2024p_pct": float(net[post].mean()) if post.any() else None}
    ic = T.ic_by_date(oos[oos["date"].isin(set(tr["date"]))], 4) if len(tr) else pd.Series(dtype=float)
    if len(ic) > 5:
        m, _, t = A.nw_t(ic.values, max(A.NW_LAG, h))
        out.update({"ic_traded": m, "t_ic_traded": t, "n_ic": int(len(ic))})
    return out, net.to_numpy() / 100.0


def evaluate(oos: pd.DataFrame, liq: pd.DataFrame, h: int, rule: str, par: dict):
    a = run_strategy(oos, h, rule, par, A.MIN_NAMES)
    r, sr = stats_from(a, oos, h)
    d = oos.merge(liq[["ticker", "date", "adv"]], on=["ticker", "date"], how="left")
    rk = d.groupby("date")["adv"].rank(pct=True)
    z3 = d[rk > 2 / 3].drop(columns="adv")
    a3 = run_strategy(z3, h, rule, par, B.MIN_CELL)
    if len(a3):
        r3, _ = stats_from(a3, z3, h)
        r["T3"] = {k: r3.get(k) for k in ("alpha_net_pct", "alpha_net_t", "turnover", "exposicion", "folds_pos", "ic_traded")}
    return r, sr


# ─── descriptivo ──────────────────────────────────────────────────────────────

def descriptive(oos: pd.DataFrame, h: int) -> dict:
    d = oos.sort_values(["date", "ticker"]).reset_index(drop=True)
    d["z"] = d["score"] - d.groupby("date")["score"].transform("mean")
    d["ex"] = d["ret"] - d.groupby("date")["ret"].transform("mean")
    dates = np.sort(d["date"].unique())
    grp = {dt: g for dt, g in d.groupby("date")}
    disp = np.array([grp[dt]["score"].std() for dt in dates])
    zs = [np.abs(grp[dt]["z"].to_numpy()) for dt in dates]
    pct = np.full(len(dates), np.nan)
    for i in range(MIN_PREV, len(dates)):
        pct[i] = np.mean(disp[max(0, i - WIN):i] <= disp[i])
    ic = T.ic_by_date(d, A.MIN_NAMES)
    dq = pd.Series(pct, index=pd.DatetimeIndex(dates)).reindex(ic.index)
    labs = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    q5 = pd.cut(dq, [0, .2, .4, .6, .8, 1.0001], labels=labs, include_lowest=True)
    res_ic = {}
    for q, v in ic.groupby(q5):
        if len(v) > 10:
            m, _, t = A.nw_t(v.values, max(A.NW_LAG, h))
            res_ic[str(q)] = {"ic": m, "t": t, "n": int(len(v))}
    rows = []
    for i in range(MIN_PREV, len(dates)):
        g = grp[dates[i]]
        past = np.sort(np.concatenate(zs[max(0, i - WIN):i]))
        p = np.searchsorted(past, np.abs(g["z"].to_numpy())) / len(past)
        rows.append(pd.DataFrame({"p": p, "z": g["z"].to_numpy(), "ex": g["ex"].to_numpy()}))
    P = pd.concat(rows, ignore_index=True)
    P["q"] = pd.cut(P["p"], [0, .2, .4, .6, .8, 1.0001], labels=labs, include_lowest=True)
    res_z = {}
    for q, g in P.groupby("q"):
        res_z[str(q)] = {"ex_largo_pct": float(g.loc[g.z > 0, "ex"].mean()), "ex_corto_pct": float(g.loc[g.z < 0, "ex"].mean()),
                         "corr_z_ex": float(np.corrcoef(g["z"], g["ex"])[0, 1]), "n": int(len(g))}
    return {"ic_por_quintil_dispersion": res_ic, "exceso_por_quintil_abs_z": res_z}


# ─── main ─────────────────────────────────────────────────────────────────────

def f(x, nd=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


CONFIGS = ([("base", {}, "base_tercil")] +
           [("gate_disp", {"q": q}, f"gate_disp_q{int(q * 100)}") for q in (0.5, 0.8)] +
           [("name_conf", {"q": q}, f"name_conf_q{int(q * 100)}") for q in (0.8, 0.9)] +
           [("conformal", {"a": a, "delta": dl}, f"conf_a{int(a * 100)}_d{int(dl * 1000)}") for a, dl in ((0.25, 0.02), (0.40, 0.0))] +
           [("topk", {"k": k, "b": b}, f"top{k}_b{b}") for k in (3, 5, 10) for b in (0, k)])


def main() -> None:
    df = A.load_panel(SCRATCH / "panel.parquet")
    cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df[np.isfinite(df["ret_fwd"])].reset_index(drop=True)
    df = df.merge(T.build_extra(df), on=["date", "ticker"], how="left")
    gc.collect()
    df = T.add_targets(df)
    df = T.calendar_cols(df)
    df["rev1d"] = -df["ret1d"]
    X = T.rank_features(df, cols)
    rr = (df.groupby("date")["rev1d"].rank(pct=True) - 0.5).fillna(0.0).to_numpy(dtype="float32")
    X_aug = np.column_stack([X, rr, rr * df["last5"].to_numpy(dtype="float32")]).astype("float32")
    liq, _ = B.liquidity_panel(sorted(df["ticker"].unique()))
    n_glob0 = count_trials()
    R = {"meta": {"win": WIN, "min_prev": MIN_PREV, "costo_lado_pct": COST, "n_trials_global_inicio": n_glob0,
                  "nota_meta_model": "src/ml/meta_model.py = meta-labeling supervisado (triple barrier); no cubre abstencion por dispersion/conformal"},
         "cfg": {}, "desc": {}}
    series = {}
    for (bn, h) in BASES:
        Xm, tcol = (X_aug, f"t_rank{h}") if bn == "rankaug" else (X, f"t_resid{h}")
        oos = T.ridge_oos(df, Xm, tcol, h)
        tg = df[["date", "ticker", f"t_rank{h}"]].rename(columns={f"t_rank{h}": "tgt"})
        oos = oos.merge(tg, on=["date", "ticker"], how="left")
        oos.to_parquet(SCRATCH / f"n1c_oos_{bn}_h{h}.parquet")
        print(f"\n=== {bn} h={h}: oos {len(oos):,} filas ===", flush=True)
        desc = descriptive(oos, h)
        R["desc"][f"{bn}_h{h}"] = B.clean(desc)
        log_trial(f"n1c_desc_{bn}_h{h}", {"tanda": "n1c", "kind": "descriptivo_quintiles", "base": bn, "h": h, "win": WIN},
                  B.clean({"ic_q5": desc["ic_por_quintil_dispersion"].get("Q5", {}).get("ic"),
                           "ic_q1": desc["ic_por_quintil_dispersion"].get("Q1", {}).get("ic")}), len(oos), path=str(ROOT / TRIALS_PATH))
        for rule, par, nm in CONFIGS:
            r, sr = evaluate(oos, liq, h, rule, par)
            full = f"{bn}_h{h}_{nm}"
            series[full] = sr
            R["cfg"][full] = {"base": bn, "h": h, "regla": nm, **B.clean(r)}
            if rule != "base":
                log_trial(f"n1c_{full}", {"tanda": "n1c", "base": bn, "h": h, "regla": nm, "par": par, "win": WIN, "cost_side_pct": COST,
                                          "n_splits": A.N_SPLITS, "embargo": A.EMBARGO, "entry": "next_open"},
                          B.clean({k: r.get(k) for k in ("alpha_net_pct", "turnover", "exposicion", "folds_pos", "ic_traded")}),
                          int(r["n_rebal"]), path=str(ROOT / TRIALS_PATH))
            t3 = r.get("T3", {})
            print(f"  {nm:18s} aN={f(r['alpha_net_pct'])} t={f(r['alpha_net_t'], 2)} turn={f(r['turnover'], 2)} exp={f(r['exposicion'], 2)} "
                  f"n={f(r['n_inv_medio'], 1)} folds+={r['folds_pos']}/{r['n_folds']} 24+={f(r.get('alpha_net_2024p_pct'))} "
                  f"ICtr={f(r.get('ic_traded'), 4)} T3 aN={f(t3.get('alpha_net_pct'))}", flush=True)
            json.dump(R, open(OUT / "n1c_abstencion.json", "w"), indent=2, ensure_ascii=False, default=str)
            gc.collect()
        del oos
        gc.collect()
    n_glob = count_trials()
    R["meta"]["n_trials_global_final"] = n_glob
    for nm, v in series.items():
        per = max(1, int(round(252 / R["cfg"][nm]["h"])))
        R["cfg"][nm]["dsr_global"] = B.clean(deflated_sharpe(v, n_glob, periods=per))
    json.dump(R, open(OUT / "n1c_abstencion.json", "w"), indent=2, ensure_ascii=False, default=str)
    write_md(R)
    print("guardado n1c_abstencion.json/.md; N global =", n_glob, flush=True)


def write_md(R: dict) -> None:
    L = ["# N1-C1 abstencion / seleccion por confianza sobre Ridge de 1B", "",
         f"- Ventana previa {R['meta']['win']} fechas (min {R['meta']['min_prev']}); costo {R['meta']['costo_lado_pct']}%/lado; neto = 2*costo*turnover.",
         f"- N global de ensayos al terminar = {R['meta']['n_trials_global_final']} (inicio {R['meta']['n_trials_global_inicio']}). DSR sobre el neto por rebalanceo (incluye 0 en abstencion).",
         "- Umbrales solo con scores/etiquetas ya cerradas (ver docstring). meta_model.py (meta-labeling supervisado) no cubre esto; no se duplica.",
         "- `base_tercil` replica el alfa de 1B (sin abstencion; no se registra como ensayo). Exposicion = fraccion de rebalanceos con posicion.", ""]
    for key in sorted({(v["base"], v["h"]) for v in R["cfg"].values()}, key=lambda x: (x[1], x[0])):
        bn, h = key
        L += [f"## {bn} h={h}", "",
              "| regla | alfa neto % | t | turnover | exposicion | n medio | folds+ | 2024+ % | IC fechas operadas | T3 alfa neto % | DSR |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for nm, r in R["cfg"].items():
            if (r["base"], r["h"]) != key:
                continue
            t3 = r.get("T3") or {}
            L.append(f"| {r['regla']} | {f(r.get('alpha_net_pct'))} | {f(r.get('alpha_net_t'), 2)} | {f(r.get('turnover'), 2)} | {f(r.get('exposicion'), 2)} | "
                     f"{f(r.get('n_inv_medio'), 1)} | {r.get('folds_pos')}/{r.get('n_folds')} | {f(r.get('alpha_net_2024p_pct'))} | "
                     f"{f(r.get('ic_traded'), 4)} | {f(t3.get('alpha_net_pct'))} | {f((r.get('dsr_global') or {}).get('dsr'))} |")
        d = R["desc"][f"{bn}_h{h}"]
        L += ["", "IC por quintil de dispersion (percentil previo): " + "; ".join(f"{q}: {f(v['ic'], 4)} (t {f(v['t'], 2)})" for q, v in d["ic_por_quintil_dispersion"].items()),
              "", "Exceso bruto % por quintil de |z| (percentil previo; largo z>0 / corto z<0): " +
              "; ".join(f"{q}: {f(v['ex_largo_pct'])}/{f(v['ex_corto_pct'])}" for q, v in d["exceso_por_quintil_abs_z"].items()), ""]
    (OUT / "n1c_abstencion.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
