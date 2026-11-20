"""
Nivel 4: UNA combinacion pre-registrada de lo que sumo un poco (sin exploracion, sin ajustar pesos).

DISEÑO FIJO (escrito ANTES de correr; no se cambia despues)
-----------------------------------------------------------
Score (h=21, cache OOS de N1-B/N1-C `n1c_oos_rankaug_h21.parquet`, 6 folds por fechas, F1 = mas reciente, NO se reentrena):
  S1 = Ridge-rank (rank_aug, target rank) h=21.
  S2 = promedio simple (pesos 0.5/0.5, sin optimizar) del percentil transversal por fecha del score Ridge y del percentil
       transversal de la reversion 1d (rev1d = -ln(C_t/C_{t-1})). Sin pesos ajustados.
Portafolio: universo del panel de 35, rebalanceo cada 21 fechas por fold (~mensual), top-10 equiponderado con buffer
  (histeresis: entra top-10, se conserva mientras rank < 20), entrada Open t+1 (r21 = C_{t+22}/O_{t+1}-1), solo datos pasados.
  Misma regla que n2b_costos (`top10_b10`): se replica y se VERIFICA contra scripts.n2b_costos.run.
Costos: costo por lado por emisora de n2b (c_1x principal; c_05x; c_2x); 0.30 % fijo solo como referencia (no se registra).
  costo = suma(compras*|w|*c_i + ventas*|w|*c_i); rebalanceo inicial solo compra.
Capa de riesgo: sin ella / con vol targeting EWMA (src/ml/vol_target.ewma_vol, span 21, objetivo 10 % anual, cap 1, exposicion
  actualizada cada 5 dias con rezago 1 dia, igual que n2a; antes de tener sigma, exposicion = 1). El cambio de exposicion paga
  el costo medio por lado de la cartera vigente. La serie diaria usa buy&hold por periodo (entrada Open t+1) y solo cuenta
  los dias en posicion. Alfa de la config con capa (aproximado, aritmetico por periodo) = suma de e_d*(port_neto_d - univ_d)
  menos el costo de cambiar exposicion; univ_d = 1/N del universo elegible, sin costo.
Configuraciones (TOPE 8, todas con log_trial family 'n4_*'):
  {S1, S2} x {sin capa, con capa} x costo x1 (4)  +  sensibilidad de costo x0.5 / x2 sobre S1 y S2 sin capa (4).
  Referencia aparte (no es config): costo fijo 0.30 %, y la replica de entropia (abajo, 1 celda, tambien registrada).
Benchmarks: (i) 1/N del universo elegible, mismo rebalanceo y costo (regla `todos`); (ii) mismo portafolio con score ALEATORIO
  (permutacion por fecha, 200 sorteos, semilla 20261119, costo x1, sin capa); se ubica el alfa neto (y el bruto) de S1/S2
  en esa distribucion nula.
CRITERIO ENDURECIDO (exito solo si TODO, costo x1): (a) IC por fecha (todas las fechas OOS del score) t(NW, lag 21) > 2;
  (b) alfa neto > 0 en >= 4 de 6 folds; (c) alfa neto 2024+ > 0; (d) IC 95 % bootstrap por bloques (bloque 3 rebalanceos,
  B=2000, semilla 7) de la media del alfa neto excluye 0; (e) se reporta DSR con el N real del log (trial_stats), no se exige.
  Tambien: tramos de estres 2020 (2020-02-15..2020-06-30) y 2022 para la capa, y los 6 folds por separado.
REPLICA DE ENTROPIA (separada, una sola celda, NO independiente: mismos datos): h=21, entropia de permutacion pe100 (n3),
  tercil transversal T3 (percentil por fecha > 2/3 entre las filas OOS), score S1. Reporta IC por fold, IC 2024+ y alfa neto
  (costo x1, regla top10_b10 con K=min(10, n//3) dentro de T3, min 4 nombres por fecha).

Ejecucion: .venv/bin/python scripts/n4_combinacion.py  (un solo proceso, float32 donde se pueda, sin paralelismo)
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
import scripts.n1a_leadlag as N
import scripts.n1b_target as T
import scripts.n2b_costos as C
import scripts.n3_prueba as P3
from src.ml.trial_stats import deflated_sharpe
from src.ml.trials import DEFAULT_PATH as TRIALS_PATH, count_trials, log_trial
from src.ml.vol_target import ewma_vol

OUT = ROOT / "data" / "eval"
SCRATCH = T.SCRATCH
H = 21
K, BUF = 10, 10
TARGET_VOL, EXPO_EVERY = 0.10, 5
BOOT_B, BOOT_L, BOOT_SEED = 2000, 3, 7
N_PERM, PERM_SEED = 200, 20261119
STRESS = {"2020": ("2020-02-15", "2020-06-30"), "2022": ("2022-01-01", "2022-12-31")}
CCOL = {"x05": "c_05x", "x1": "c_1x", "x2": "c_2x"}
ENTRY_HOLD = H + 1                      # r21 cubre 22 sesiones (Open t+1 -> Close t+22)


# ─── periodos y portafolio (replica de n2b top10_b10) ─────────────────────────

def thin(d: pd.DataFrame) -> pd.DataFrame:
    """Solo las fechas de rebalanceo (cada H fechas por fold), identicas a C.run con h=H."""
    keep = []
    for fo, zf in d.groupby("fold"):
        keep += list(np.sort(zf["date"].unique())[::H])
    return d[d["date"].isin(set(keep))].sort_values(["date", "ticker"]).reset_index(drop=True)


def make_periods(dt_: pd.DataFrame, ccols: list[str]) -> list[dict]:
    per = []
    for dt, w in dt_.groupby("date"):
        per.append({"date": dt, "fold": int(w["fold"].iloc[0]), "tk": w["ticker"].to_numpy(), "ret": w["ret"].to_numpy(),
                    "score": w["score"].to_numpy(), "c": {c: w[c].fillna(C.FIXED).to_numpy() for c in ccols}})
    return sorted(per, key=lambda p: (p["fold"], p["date"]))


def run_port(per: list[dict], mode: str, ccol: str | None, fixed: float | None = None, score_override=None,
             min_names: int = A.MIN_NAMES) -> pd.DataFrame:
    """mode: 'top' (top-K buffer) | 'all' (1/N). Devuelve una fila por rebalanceo con `cur` (tickers)."""
    rows = []
    prev: set = set()
    prev_fold = None
    first = True
    for i, p in enumerate(per):
        if p["fold"] != prev_fold:
            prev, first, prev_fold = set(), True, p["fold"]
        n = len(p["tk"])
        if n < min_names:
            continue
        cvec = np.full(n, fixed) if fixed is not None else p["c"][ccol]
        cmap = dict(zip(p["tk"], cvec))
        sc = p["score"] if score_override is None else score_override[i]
        if mode == "all":
            cur = set(p["tk"])
        else:
            k = int(min(K, max(1, n // 3)))
            order = list(p["tk"][np.argsort(-sc, kind="stable")])
            rank = {t_: j for j, t_ in enumerate(order)}
            new = set(order[:k])
            if prev:
                keep = sorted([t_ for t_ in prev if t_ in rank and rank[t_] < k + BUF], key=lambda t_: rank[t_])[:k]
                fill = [t_ for t_ in order if t_ not in keep]
                new = set(keep) | set(fill[: k - len(keep)])
            cur = new
        if not cur:
            continue
        buys, sells = cur - prev, prev - cur
        wb, ws = 1.0 / len(cur), 1.0 / max(len(prev), 1)
        cost = wb * sum(cmap.get(t_, C.FIXED) for t_ in buys) + ws * sum(cmap.get(t_, C.FIXED) for t_ in sells)
        turn = (len(buys) / len(cur)) if not first else 1.0
        m = np.isin(p["tk"], list(cur))
        ret_p, ret_u = float(p["ret"][m].mean()), float(p["ret"].mean())
        if np.isfinite(ret_p) and np.isfinite(ret_u):
            rows.append({"date": p["date"], "fold": p["fold"], "ret_p": ret_p, "ret_u": ret_u, "excess": ret_p - ret_u,
                         "turnover": turn, "cost": cost, "c_hold": float(cvec[m].mean()), "n_inv": len(cur), "cur": cur,
                         "idx": i})
        prev, first = cur, False
    return pd.DataFrame(rows)


# ─── estadistica ──────────────────────────────────────────────────────────────

def boot_ci(x: np.ndarray, rng) -> tuple[float, float]:
    n = len(x)
    nb = int(np.ceil(n / BOOT_L))
    st = rng.integers(0, n, size=(BOOT_B, nb))
    idx = ((st[:, :, None] + np.arange(BOOT_L)[None, None, :]) % n).reshape(BOOT_B, -1)[:, :n]
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def stream_metrics(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return {}
    nav = np.cumprod(1 + x)
    sd = x.std(ddof=1)
    return {"n": int(len(x)), "ret_ann": float(nav[-1] ** (252 / len(x)) - 1), "vol": float(sd * np.sqrt(252)),
            "mdd": float((nav / np.maximum.accumulate(nav) - 1).min()),
            "sharpe": float(x.mean() / sd * np.sqrt(252)) if sd > 0 else None}


def window_metrics(s: pd.Series, a: str, b: str) -> dict:
    return stream_metrics(s[(s.index >= a) & (s.index <= b)].to_numpy())


# ─── serie diaria (buy&hold por periodo) y capa de riesgo ─────────────────────

class Prices:
    def __init__(self, tickers):
        px = {tk: N.stock_series(tk) for tk in tickers}
        self.cal = pd.DatetimeIndex(sorted(set().union(*[s.index for s in px.values() if s is not None])))
        self.col = {tk: j for j, tk in enumerate(tickers)}
        self.O = np.full((len(self.cal), len(tickers)), np.nan, dtype="float64")
        self.C = self.O.copy()
        for tk, s in px.items():
            if s is None:
                continue
            ix = self.cal.get_indexer(s.index)
            self.O[ix, self.col[tk]] = s["Open"].to_numpy()
            self.C[ix, self.col[tk]] = s["Close"].to_numpy()


def daily_stream(px: Prices, res: pd.DataFrame, per: list[dict]) -> pd.DataFrame:
    """Retorno bruto diario, costo (fraccion), costo medio por lado (fraccion), id de periodo; solo dias en posicion."""
    pos, g, cst, cu, pid = [], [], [], [], []
    for fo, rf in res.groupby("fold", sort=False):
        rf = rf.sort_values("date").reset_index(drop=True)
        cum_prev, p_prev = None, None
        for j, r in rf.iterrows():
            p = int(px.cal.searchsorted(r["date"]))
            cols = [px.col[t_] for t_ in r["cur"]]
            nxt = int(px.cal.searchsorted(rf.loc[j + 1, "date"])) if j + 1 < len(rf) else None
            nd = (nxt - p) if nxt is not None else ENTRY_HOLD
            nd = max(1, min(nd, len(px.cal) - 1 - p))
            Cm, Cp = px.C[p + 1:p + 1 + nd][:, cols], px.C[p:p + nd][:, cols]
            G = Cm / Cp
            G[0] = Cm[0] / px.O[p + 1, cols]
            G = np.where(np.isfinite(G), G, 1.0)
            cum = np.cumprod(G, axis=0)
            bask = cum.mean(axis=1)
            dr = bask / np.concatenate([[1.0], bask[:-1]]) - 1.0
            if cum_prev is not None:           # el basket anterior se vende en Open p+1 (gap nocturno)
                gap = px.O[p_prev[1] + 1, p_prev[0]] / px.C[p_prev[1], p_prev[0]]
                gap = np.where(np.isfinite(gap), gap, 1.0)
                old = (cum_prev * gap).mean() / cum_prev.mean() - 1.0
                dr[0] = (1 + old) * (1 + dr[0]) - 1.0
            cum_prev, p_prev = cum[-1], (cols, p + nd)
            for d_ in range(nd):
                pos.append(p + 1 + d_)
                g.append(dr[d_])
                cst.append(r["cost"] / 100.0 if d_ == 0 else 0.0)
                cu.append(r["c_hold"] / 100.0)
                pid.append(int(r["idx"]))
    s = pd.DataFrame({"g": g, "cost": cst, "cu": cu, "pid": pid}, index=px.cal[np.array(pos)])
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def exposure(g: pd.Series) -> pd.Series:
    sig = ewma_vol(g, 21).shift(1)
    raw = (TARGET_VOL / sig.where(sig > 0)).clip(upper=1.0)
    e = raw.where(np.arange(len(raw)) % EXPO_EVERY == 0).ffill()
    return e.fillna(1.0)


def layered(s: pd.DataFrame, u: pd.Series) -> pd.DataFrame:
    e = exposure(s["g"])
    ep = e.shift(1).fillna(e)
    net0 = s["g"] - s["cost"]
    net_l = e * net0 - s["cu"] * (e - ep).abs()
    ua = u.reindex(s.index).fillna(0.0)
    alpha_l = e * (net0 - ua) - s["cu"] * (e - ep).abs()
    return pd.DataFrame({"net0": net0, "net_l": net_l, "e": e, "alpha_l": alpha_l, "pid": s["pid"]})


# ─── evaluacion de una configuracion ──────────────────────────────────────────

def alpha_stats(a: pd.Series, dates: pd.Series, folds: pd.Series, rng) -> dict:
    x = a.to_numpy()
    byf = a.groupby(folds.values).mean()
    post = (pd.to_datetime(dates) >= T.TEST_START).values
    lo, hi = boot_ci(x, rng)
    return {"n_rebal": int(len(x)), "alpha_net_pct": float(x.mean()), "t": T.tstat(x),
            "folds_pos": int((byf > 0).sum()), "alpha_by_fold": {int(k): float(v) for k, v in byf.items()},
            "alpha_2024p_pct": float(a[post].mean()) if post.any() else None,
            "boot95_pct": [lo, hi], "sharpe_ann": float(x.mean() / x.std(ddof=1) * np.sqrt(252 / H))}


def verdict(ic_t: float, st: dict) -> dict:
    v = {"a_IC_t_gt_2": bool(ic_t > 2), "b_folds_ge4": bool(st["folds_pos"] >= 4),
         "c_2024p_gt_0": bool((st["alpha_2024p_pct"] or -1) > 0), "d_boot_excl0": bool(st["boot95_pct"][0] > 0)}
    v["todo_ok"] = all(v.values())
    return v


def f(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main() -> None:
    oos = pd.read_parquet(SCRATCH / "n1c_oos_rankaug_h21.parquet")[["date", "ticker", "ret", "score", "fold"]]
    tick = sorted(oos["ticker"].unique())
    print("OOS", len(oos), "tickers", len(tick), "folds", sorted(oos["fold"].unique()), flush=True)
    # rev1d (info hasta el cierre t)
    rv = []
    for tk in tick:
        s = N.stock_series(tk)
        if s is not None:
            rv.append(pd.DataFrame({"date": s.index, "ticker": tk, "rev1d": -np.log(s["Close"]).diff().to_numpy() * 100.0}))
    rv = pd.concat(rv, ignore_index=True)
    d = oos.merge(rv, on=["date", "ticker"], how="left")
    del rv
    g_ = d.groupby("date")
    d["s1"] = d["score"].astype("float64")
    d["s2"] = 0.5 * g_["score"].rank(pct=True) + 0.5 * g_["rev1d"].rank(pct=True).fillna(0.5)
    cost = C.cost_panel(tick)
    cost["c_1x"] = C.side_cost(cost)
    cost["c_05x"] = C.side_cost(cost, 0.5)
    cost["c_2x"] = C.side_cost(cost, 2.0)
    d = d.merge(cost[["date", "ticker", "c_1x", "c_05x", "c_2x"]], on=["date", "ticker"], how="left")
    del cost
    gc.collect()
    n_trials0 = count_trials()

    # IC por fecha de cada score (todas las fechas OOS)
    ic_res = {}
    for nm, col in (("S1", "s1"), ("S2", "s2")):
        z = d[["date", "ticker", "ret", "fold"]].assign(score=d[col])
        st = P3.ic_stats(z, H, A.MIN_NAMES)
        ic_res[nm] = {k: v for k, v in st.items() if k != "_series"}
        print(nm, "IC", f(st["ic"], 4), "t", f(st["t"], 2), "folds+", st.get("folds_pos"), "IC24", f(st["ic24"], 4),
              "por fold", {k: round(v, 4) for k, v in st["by_fold"].items()}, flush=True)

    ccols = list(CCOL.values())
    dthin = thin(d)
    px = Prices(tick)
    R = {"meta": {"diseno": "ver docstring scripts/n4_combinacion.py", "n_trials_inicio": n_trials0, "folds": "F1 = mas reciente (cache)",
                  "n_oos": int(len(oos)), "fold_rangos": {int(k): [str(g["date"].min().date()), str(g["date"].max().date())]
                                                          for k, g in oos.groupby("fold")},
                  "ic_score": ic_res, "bootstrap": {"B": BOOT_B, "bloque": BOOT_L, "semilla": BOOT_SEED}}, "cfg": {}}
    rng = np.random.default_rng(BOOT_SEED)
    series = {}

    per = {}
    for nm, col in (("S1", "s1"), ("S2", "s2")):
        per[nm] = make_periods(dthin.assign(score=dthin[col]), ccols)
    per_all = per["S1"]

    # verificacion contra n2b (S1, x1, sin filtro)
    chk = C.run(C.prep(oos.assign(score=oos["score"]), pd.concat([d[["date", "ticker", "c_1x"]].assign(
        adv=np.nan)]).drop_duplicates(["date", "ticker"]), H), H, "top10_b10", "c_1x", None)
    mine = run_port(per["S1"], "top", "c_1x")
    R["meta"]["verificacion_vs_n2b"] = {"n2b_alpha_net": float((chk["excess"] - chk["cost"]).mean()), "n4_alpha_net": float((mine["excess"] - mine["cost"]).mean()),
                                         "n2b_n": int(len(chk)), "n4_n": int(len(mine))}
    print("verificacion vs n2b:", R["meta"]["verificacion_vs_n2b"], flush=True)
    del chk
    gc.collect()

    # benchmark 1/N y sus series
    res_all = {}
    for sc in ("x05", "x1", "x2"):
        res_all[sc] = run_port(per_all, "all", CCOL[sc])
    uni = daily_stream(px, res_all["x1"], per_all)
    u_gross = uni["g"]
    R["bench_1N"] = {"x1": {"ret_mensual_neto_pct": float((res_all["x1"]["ret_p"] - res_all["x1"]["cost"]).mean()),
                            "turnover": float(res_all["x1"]["turnover"].mean()), "costo_pct": float(res_all["x1"]["cost"].mean()),
                            "diario": stream_metrics((uni["g"] - uni["cost"]).to_numpy())},
                     "x1_con_capa": stream_metrics(layered(uni, u_gross)["net_l"].to_numpy()),
                     "estres": {k: window_metrics(uni["g"] - uni["cost"], *v) for k, v in STRESS.items()}}
    ul = layered(uni, u_gross)["net_l"]
    R["bench_1N"]["estres_con_capa"] = {k: window_metrics(ul, *v) for k, v in STRESS.items()}
    print("1/N", json.dumps(B.clean(R["bench_1N"]["x1"]["diario"])), flush=True)

    # configuraciones
    jobs = [("S1", "sinCapa", "x1"), ("S2", "sinCapa", "x1"), ("S1", "conCapa", "x1"), ("S2", "conCapa", "x1"),
            ("S1", "sinCapa", "x05"), ("S1", "sinCapa", "x2"), ("S2", "sinCapa", "x05"), ("S2", "sinCapa", "x2")]
    cache_res = {}
    for sc_, capa, cm in jobs:
        key = (sc_, cm)
        if key not in cache_res:
            cache_res[key] = run_port(per[sc_], "top", CCOL[cm])
        res = cache_res[key].reset_index(drop=True)
        name = f"{sc_}_{capa}_{cm}"
        net = res["excess"] - res["cost"]
        icsc = ic_res[sc_]
        st_nocapa = alpha_stats(net, res["date"], res["fold"], rng)
        entry = {"score": sc_, "capa": capa, "costo": cm, "ic": icsc["ic"], "ic_t_nw": icsc["t"], "ic_by_fold": icsc["by_fold"],
                 "ic_2024p": icsc["ic24"], "bruto_pct": float(res["excess"].mean()), "costo_pct": float(res["cost"].mean()),
                 "turnover": float(res["turnover"].mean()), "n_inv": float(res["n_inv"].mean()),
                 "abs_ret_mensual_neto_pct": float((res["ret_p"] - res["cost"]).mean())}
        s = daily_stream(px, res, per[sc_])
        sl = layered(s, u_gross)
        # sanity: retorno compuesto diario por periodo vs ret_p
        comp = (1 + s["g"]).groupby(s["pid"]).prod() - 1
        mp = res.set_index("idx")["ret_p"] / 100.0
        entry["sanity_diario_vs_r21_mae_pct"] = float((comp.reindex(mp.index) - mp).abs().mean() * 100)
        if capa == "sinCapa":
            st = st_nocapa
            entry["riesgo"] = stream_metrics(sl["net0"].to_numpy())
            entry["estres"] = {k: window_metrics(sl["net0"], *v) for k, v in STRESS.items()}
            by_fold_risk = {int(fo): stream_metrics(sl["net0"][(sl.index >= a) & (sl.index <= b)].to_numpy())
                            for fo, (a, b) in ((int(k), (pd.Timestamp(v[0]), pd.Timestamp(v[1]))) for k, v in R["meta"]["fold_rangos"].items())}
        else:
            al = sl["alpha_l"].groupby(sl["pid"]).sum() * 100.0
            al = al.reindex(res["idx"].values).fillna(0.0)
            al.index = res.index
            st = alpha_stats(al, res["date"], res["fold"], rng)
            entry["riesgo"] = stream_metrics(sl["net_l"].to_numpy())
            entry["expo_media"] = float(sl["e"].mean())
            entry["estres"] = {k: window_metrics(sl["net_l"], *v) for k, v in STRESS.items()}
            entry["sin_capa_mismo_score"] = stream_metrics(sl["net0"].to_numpy())
            entry["estres_sin_capa"] = {k: window_metrics(sl["net0"], *v) for k, v in STRESS.items()}
            by_fold_risk = {int(fo): stream_metrics(sl["net_l"][(sl.index >= a) & (sl.index <= b)].to_numpy())
                            for fo, (a, b) in ((int(k), (pd.Timestamp(v[0]), pd.Timestamp(v[1]))) for k, v in R["meta"]["fold_rangos"].items())}
        entry.update({"alfa": st, "riesgo_por_fold": by_fold_risk, "veredicto": verdict(icsc["t"], st)})
        if cm == "x1" and capa == "sinCapa":
            fx = run_port(per[sc_], "top", None, fixed=C.FIXED)
            nf = alpha_stats(fx["excess"] - fx["cost"], fx["date"], fx["fold"], rng)
            entry["ref_fijo030"] = {k: nf[k] for k in ("alpha_net_pct", "t", "folds_pos", "alpha_2024p_pct", "boot95_pct")}
        series[name] = (net.to_numpy() if capa == "sinCapa" else al.to_numpy()) / 100.0
        R["cfg"][name] = B.clean(entry)
        log_trial(f"n4_{name}", {"tanda": "n4", "score": sc_, "capa": capa, "costo": cm, "h": H, "k": K, "buffer": BUF,
                                 "target_vol": TARGET_VOL, "entry": "next_open", "n_splits": A.N_SPLITS, "embargo": A.EMBARGO},
                  B.clean({"alpha_net_pct": st["alpha_net_pct"], "t": st["t"], "folds_pos": st["folds_pos"], "ic": icsc["ic"],
                           "mdd": entry["riesgo"].get("mdd"), "sharpe": entry["riesgo"].get("sharpe")}),
                  int(len(res)), path=str(ROOT / TRIALS_PATH))
        print(f"{name:16s} IC={f(icsc['ic'],4)}(t{f(icsc['t'],2)}) bruto={entry['bruto_pct']:.3f} costo={entry['costo_pct']:.3f} "
              f"aN={st['alpha_net_pct']:.3f} t={f(st['t'],2)} f+={st['folds_pos']} 24+={f(st['alpha_2024p_pct'])} "
              f"boot={[round(x, 3) for x in st['boot95_pct']]} MDD={f(entry['riesgo'].get('mdd'))} SR={f(entry['riesgo'].get('sharpe'),2)} "
              f"ok={entry['veredicto']['todo_ok']}", flush=True)
        json.dump(B.clean(R), open(OUT / "n4_combinacion.json", "w"), indent=2, ensure_ascii=False, default=str)
        del s, sl
        gc.collect()

    # benchmark: score aleatorio (200 permutaciones por fecha)
    prng = np.random.default_rng(PERM_SEED)
    nulls = {"alpha_net": [], "bruto": [], "turnover": []}
    for _ in range(N_PERM):
        ov = [prng.random(len(p["tk"])) for p in per_all]
        rr = run_port(per_all, "top", "c_1x", score_override=ov)
        nulls["alpha_net"].append(float((rr["excess"] - rr["cost"]).mean()))
        nulls["bruto"].append(float(rr["excess"].mean()))
        nulls["turnover"].append(float(rr["turnover"].mean()))
    for k in nulls:
        nulls[k] = np.array(nulls[k])
    R["bench_aleatorio"] = {"n_perm": N_PERM, "alpha_net_media": float(nulls["alpha_net"].mean()), "alpha_net_sd": float(nulls["alpha_net"].std()),
                            "alpha_net_p95": float(np.percentile(nulls["alpha_net"], 95)), "bruto_media": float(nulls["bruto"].mean()),
                            "bruto_sd": float(nulls["bruto"].std()), "bruto_p95": float(np.percentile(nulls["bruto"], 95)),
                            "turnover_media": float(nulls["turnover"].mean()),
                            "percentil_S1_neto": float((nulls["alpha_net"] < R["cfg"]["S1_sinCapa_x1"]["alfa"]["alpha_net_pct"]).mean() * 100),
                            "percentil_S2_neto": float((nulls["alpha_net"] < R["cfg"]["S2_sinCapa_x1"]["alfa"]["alpha_net_pct"]).mean() * 100),
                            "percentil_S1_bruto": float((nulls["bruto"] < R["cfg"]["S1_sinCapa_x1"]["bruto_pct"]).mean() * 100),
                            "percentil_S2_bruto": float((nulls["bruto"] < R["cfg"]["S2_sinCapa_x1"]["bruto_pct"]).mean() * 100)}
    print("aleatorio", json.dumps(B.clean(R["bench_aleatorio"])), flush=True)

    # replica de entropia (una celda)
    ent = P3.build_micro(tick)[["date", "ticker", "pe100"]]
    z = d[["date", "ticker", "ret", "score", "fold", "c_1x"]].merge(ent, on=["date", "ticker"], how="left")
    del ent
    z["rk"] = z.groupby("date")["pe100"].rank(pct=True)
    z3 = z[z["rk"] > 2 / 3].copy()
    st = P3.ic_stats(z3, H, B.MIN_CELL)
    # IC por fold de la celda y del universo completo
    pe_per = make_periods(thin(z3).assign(), ["c_1x"])
    rp = run_port(pe_per, "top", "c_1x", min_names=B.MIN_CELL)
    net = rp["excess"] - rp["cost"]
    sta = alpha_stats(net, rp["date"], rp["fold"], rng)
    rep = {"celda": "h21, pe100, T3 transversal, S1", "n_filas": int(len(z3)),
           "ic": st["ic"], "ic_t_nw": st["t"], "ic_by_fold": st["by_fold"], "folds_ic_pos": st.get("folds_pos"), "ic_2024p": st["ic24"],
           "ic_universo_completo": ic_res["S1"]["ic"], "bruto_pct": float(rp["excess"].mean()), "costo_pct": float(rp["cost"].mean()),
           "turnover": float(rp["turnover"].mean()), "n_inv": float(rp["n_inv"].mean()), "alfa": sta,
           "nota": "replica NO independiente: mismos datos y misma historia que la celda hallada en N3 (48 celdas)"}
    R["replica_entropia"] = B.clean(rep)
    series["replica_entropia"] = net.to_numpy() / 100.0
    log_trial("n4_replica_entropia", {"tanda": "n4", "h": H, "var": "pe100", "tercil": "T3x", "score": "S1", "costo": "x1", "k": K,
                                      "buffer": BUF, "entry": "next_open"},
              B.clean({"ic": st["ic"], "t": st["t"], "alpha_net_pct": sta["alpha_net_pct"], "folds_pos": sta["folds_pos"]}),
              int(len(rp)), path=str(ROOT / TRIALS_PATH))
    print("replica", json.dumps(R["replica_entropia"], default=str)[:900], flush=True)

    # DSR con N real (despues de registrar todo)
    n_glob = count_trials()
    R["meta"]["n_trials_final"] = n_glob
    for k, v in series.items():
        dsr = B.clean(deflated_sharpe(v, n_glob, periods=12))
        (R["cfg"][k] if k in R["cfg"] else R["replica_entropia"])["dsr_global"] = dsr
    json.dump(B.clean(R), open(OUT / "n4_combinacion.json", "w"), indent=2, ensure_ascii=False, default=str)
    write_md(R)
    print("guardado; N global =", n_glob, flush=True)


def write_md(R: dict) -> None:
    m = R["meta"]
    L = ["# N4 combinacion pre-registrada (Ridge-rank h=21 [+ rev1d], top-10 buffer, costo por emisora, capa de vol)", "",
         "Diseno fijo en el docstring de `scripts/n4_combinacion.py` (escrito antes de correr). Alfa = exceso % por rebalanceo vs 1/N del universo "
         "elegible (relativo, benchmark sin costo, como N2-B). Folds: F1 = mas reciente. Config con capa: alfa aritmetico aproximado (ver docstring).",
         f"- Verificacion de la replica del portafolio contra `n2b_costos.run`: {m['verificacion_vs_n2b']}",
         f"- N de ensayos del log: inicio {m['n_trials_inicio']}, final {m['n_trials_final']}. Rangos de folds: {m['fold_rangos']}",
         f"- IC por fecha del score (todas las fechas OOS): " + "; ".join(f"{k}: {f(v['ic'], 4)} (t NW {f(v['t'], 2)}), 2024+ {f(v['ic24'], 4)}"
                                                                       for k, v in m["ic_score"].items()), "",
         "## Tabla clave (costo x1 salvo indicacion)", "",
         "| config | IC (t) | bruto % | costo % | alfa neto % | t | folds+ | 2024+ % | boot95 % | MDD | Sharpe neto | DSR | (a) | (b) | (c) | (d) | todo |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in R["cfg"].items():
        a, r, ve = v["alfa"], v["riesgo"], v["veredicto"]
        L.append(f"| {k} | {f(v['ic'], 4)} ({f(v['ic_t_nw'], 2)}) | {f(v['bruto_pct'])} | {f(v['costo_pct'])} | {f(a['alpha_net_pct'])} | {f(a['t'], 2)} | "
                 f"{a['folds_pos']}/6 | {f(a['alpha_2024p_pct'])} | [{f(a['boot95_pct'][0])}, {f(a['boot95_pct'][1])}] | {f(r.get('mdd'), 3)} | "
                 f"{f(r.get('sharpe'), 2)} | {f((v.get('dsr_global') or {}).get('dsr'))} | {ve['a_IC_t_gt_2']} | {ve['b_folds_ge4']} | "
                 f"{ve['c_2024p_gt_0']} | {ve['d_boot_excl0']} | {ve['todo_ok']} |")
    L += ["", "MDD/Sharpe: serie diaria neta (retorno absoluto, largo, resto en efectivo), solo dias en posicion; 'conCapa' = con vol targeting EWMA 10 %.", "",
          "## Referencias", "", f"- 1/N (mismo rebalanceo y costo x1): ret mensual neto {f(R['bench_1N']['x1']['ret_mensual_neto_pct'])} %, "
          f"turnover {f(R['bench_1N']['x1']['turnover'])}; diario {R['bench_1N']['x1']['diario']}; con capa {R['bench_1N']['x1_con_capa']}.",
          f"- Score aleatorio (200 perm.): {R['bench_aleatorio']}"]
    for k in ("S1_sinCapa_x1", "S2_sinCapa_x1"):
        if "ref_fijo030" in R["cfg"].get(k, {}):
            L.append(f"- Referencia costo fijo 0.30 % {k}: {R['cfg'][k]['ref_fijo030']}")
    L += ["", "## Estres (MDD / Sharpe, serie diaria neta)", "", "| config | 2020 MDD | 2020 SR | 2022 MDD | 2022 SR |", "|---|---|---|---|---|"]
    rows = [("1/N sin capa", R["bench_1N"]["estres"]), ("1/N con capa", R["bench_1N"]["estres_con_capa"])]
    for k, v in R["cfg"].items():
        if v["costo"] == "x1":
            rows.append((k, v["estres"]))
            if "estres_sin_capa" in v:
                rows.append((k + " (misma cartera sin capa)", v["estres_sin_capa"]))
    for nm, e in rows:
        L.append(f"| {nm} | {f(e['2020'].get('mdd'), 3)} | {f(e['2020'].get('sharpe'), 2)} | {f(e['2022'].get('mdd'), 3)} | {f(e['2022'].get('sharpe'), 2)} |")
    L += ["", "## Folds por separado (alfa neto % / IC / MDD diario)", "", "| config | " + " | ".join(f"F{i}" for i in range(1, 7)) + " |", "|---|" + "---|" * 6]
    for k, v in R["cfg"].items():
        L.append(f"| {k} | " + " | ".join(
            f"{f(v['alfa']['alpha_by_fold'].get(str(i)) if str(i) in v['alfa']['alpha_by_fold'] else v['alfa']['alpha_by_fold'].get(i), 2)} / "
            f"{f(v['ic_by_fold'].get(str(i), v['ic_by_fold'].get(i)), 3)} / {f((v['riesgo_por_fold'].get(str(i)) or v['riesgo_por_fold'].get(i) or {}).get('mdd'), 2)}"
            for i in range(1, 7)) + " |")
    e = R["replica_entropia"]
    L += ["", "## Replica de entropia (h=21, pe100, T3 transversal, S1) - NO independiente (mismos datos)", "",
          f"- IC {f(e['ic'], 4)} (t NW {f(e['ic_t_nw'], 2)}), folds IC+ {e['folds_ic_pos']}/6, IC 2024+ {f(e['ic_2024p'], 4)} (universo completo {f(e['ic_universo_completo'], 4)}).",
          f"- IC por fold: {e['ic_by_fold']}",
          f"- Alfa neto {f(e['alfa']['alpha_net_pct'])} % (t {f(e['alfa']['t'], 2)}), bruto {f(e['bruto_pct'])}, costo {f(e['costo_pct'])}, folds+ {e['alfa']['folds_pos']}/6, "
          f"2024+ {f(e['alfa']['alpha_2024p_pct'])}, boot95 {e['alfa']['boot95_pct']}, DSR {f((e.get('dsr_global') or {}).get('dsr'))}.",
          f"- {e['nota']}"]
    (OUT / "n4_combinacion.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
