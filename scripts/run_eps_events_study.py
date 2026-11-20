"""
N5 · Eventos de EPS: reversión con/sin anuncio (a), calendario esperado (b1 vol, b2 filtro), PEAD (c).

Criterios fijados en ``avances/2026-11-12_N5_preregistro.md`` (antes de ver resultados).

Uso:
    .venv/bin/python scripts/run_eps_events_study.py            # completo (~3-5 min)
    .venv/bin/python scripts/run_eps_events_study.py --skip-vol # sin b1 (el más lento)
    .venv/bin/python scripts/run_eps_events_study.py --no-log   # no escribe en trial_log

Salida: data/eval/eps_events_study.json y .md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_reversal_study as rv  # noqa: E402
from src.ml import vol_calendar as vc  # noqa: E402
from src.ml import vol_regressors as vr  # noqa: E402
from src.ml.eps_events import (  # noqa: E402
    earnings_eps_features,
    expected_accuracy,
    expected_flag,
    has_recent_event,
    sharpe_diff_ci,
)
from src.ml.reversal import build_signals, forward_returns, to_panel  # noqa: E402
from src.ml.shared.evaluators.costs import net_returns, turnover  # noqa: E402
from src.ml.shared.evaluators.cross_section import (  # noqa: E402
    equal_weight_returns,
    ic_by_date,
    ic_summary,
    random_topk_returns,
    topk_returns,
)
from src.ml.trials import log_trial  # noqa: E402

EPS = ROOT / "data" / "eps" / "eps_clean.parquet"
EVAL = ROOT / "data" / "eval"
FAMILY = "n5_eps_events"
PPY = 50.4
COST = 0.003
MIN_GROUP = 6
STEP = 5
SPLIT = "2018-01-01"
P_LATE = "2023-01-01"
PERIODS = {"2008-2017": (None, "2017-12-31"), "2018-2026": (SPLIT, None), "2023-2026": (P_LATE, None),
           "2008-2026": (None, None)}


def cut(x, period: str):
    lo, hi = PERIODS[period]
    return x.loc[lo:hi] if lo or hi else x


def summ(ic: pd.Series) -> dict:
    s = ic_summary(ic, step=1, block=4, seed=0) if len(ic) else {"mean": float("nan"), "t": float("nan"), "n": 0}
    return {k: (int(v) if k == "n" else float(v)) for k, v in s.items()}


# ── datos ────────────────────────────────────────────────────────────────────

def load_eps() -> pd.DataFrame:
    d = pd.read_parquet(EPS)
    return d[d["ticker"].str.endswith(".MX")].copy()


def build(eps: pd.DataFrame):
    """Matrices (máscara M3a) restringidas a tickers con EPS + bandera 'con evento'."""
    O, C, V, _ = rv.load_universe(True)
    cols = [c for c in C.columns if c in set(eps["ticker"])]
    O, C = O[cols], C[cols]
    sig = build_signals(O, C)
    fwd = forward_returns(O, C, 5)
    ev = pd.DataFrame({c: has_recent_event(eps[eps["ticker"] == c], C.index, 5) for c in cols}).astype(float)
    return O, C, cols, sig, fwd, ev


# ── (a) reversión con/sin evento ─────────────────────────────────────────────

def part_a(C, cols, sig, fwd, ev) -> tuple[dict, pd.DataFrame]:
    wide = {"score": -sig["ret_5d"], "ret": fwd["open_close"], "ev": ev}
    panel = to_panel(wide)
    weekly = C.index[::STEP]
    panel = panel[panel.index.isin(weekly)]
    out: dict = {"tickers": cols, "n_tickers": len(cols)}
    groups = {"simple": panel, "sin_evento": panel[panel["ev"] == 0], "con_evento": panel[panel["ev"] == 1]}
    ics = {k: ic_by_date(g, "score", "ret", min_names=(12 if k == "simple" else MIN_GROUP)) for k, g in groups.items()}
    both = pd.concat([ics["sin_evento"], ics["con_evento"]], axis=1, keys=["s", "c"], sort=True).dropna()
    diff = both["s"] - both["c"]
    res = {}
    for per in PERIODS:
        res[per] = {k: summ(cut(v, per)) for k, v in ics.items()}
        res[per]["diferencia"] = summ(cut(diff, per))
    out["ic"] = res
    out["n_fechas_ambos_grupos"] = {per: int(len(cut(diff, per))) for per in PERIODS}
    # tamaño medio de cada grupo
    n_g = panel.dropna(subset=["score", "ret"]).groupby([panel.dropna(subset=["score", "ret"]).index, "ev"]).size()
    out["tamano_medio_grupo"] = {"sin_evento": float(n_g.xs(0.0, level=1).mean()),
                                 "con_evento": float(n_g.xs(1.0, level=1).mean())}
    # baselines top-5 (2018-2026): bruto por grupo vs EW del grupo y vs aleatorio
    base = {}
    for k in ("sin_evento", "con_evento", "simple"):
        g = cut(groups[k], "2018-2026")
        mn = 12 if k == "simple" else MIN_GROUP
        tk = topk_returns(g, "score", "ret", 5, min_names=mn)
        ew = equal_weight_returns(g, "ret", min_names=mn).reindex(tk.index)
        rnd = random_topk_returns(g, "ret", 5, n_draws=500, seed=0, min_names=mn)
        base[k] = {"semanas": int(len(tk)), "top5_bruto_pct": float(tk.mean() * 100),
                   "ew_pct": float(ew.mean() * 100), "aleatorio_pct": float(rnd.mean(axis=0).mean() * 100),
                   "exceso_vs_ew_pct": float((tk - ew).mean() * 100)}
    out["baselines_top5_2018_2026"] = base
    # veredicto
    s, c, d = res["2018-2026"]["sin_evento"], res["2018-2026"]["con_evento"], res["2018-2026"]["diferencia"]
    late = res["2023-2026"]["sin_evento"]["mean"]
    cond = {"sin>+0.03_t>2": bool(s["mean"] > 0.03 and s["t"] > 2), "con<=0": bool(c["mean"] <= 0),
            "dif_ci_lo>0": bool(d["ci_lo"] > 0), "descarte_sin<0_2023_26": bool(late < 0)}
    out["veredicto"] = {"condiciones": cond, "pasa": bool(cond["sin>+0.03_t>2"] and cond["con<=0"]
                                                          and cond["dif_ci_lo>0"] and not cond["descarte_sin<0_2023_26"])}
    return out, panel


# ── (b2) filtro de riesgo ────────────────────────────────────────────────────

def part_b2(eps, C, cols, fwd) -> dict:
    flag = pd.DataFrame({c: expected_flag(C.index, eps[eps["ticker"] == c]) for c in cols})
    # oráculo: anuncio real conocido (available_from) entre t+1 y t+7 de la malla
    idx = C.index
    orc = pd.DataFrame(0.0, index=idx, columns=cols)
    for c in cols:
        pos = idx.searchsorted(pd.DatetimeIndex(eps.loc[eps["ticker"] == c, "available_from"]), side="left")
        arr = np.zeros(len(idx))
        for p in pos:
            lo, hi = max(p - 7, 0), p - 1            # t en [p-7, p-1] ⇒ anuncio en t+1..t+7
            if hi >= 0 and p < len(idx):
                arr[lo:hi + 1] = 1.0
        orc[c] = arr
    ret = fwd["open_close"]
    wk = idx[::STEP]
    r, f, o = ret.loc[wk], flag.loc[wk], orc.loc[wk]
    elig = r.notna() & f.notna()

    def ew(mask: pd.DataFrame):
        w = mask.astype(float)
        n = w.sum(axis=1)
        w = w.div(n.where(n >= 5), axis=0)           # mín. 5 nombres
        ok = w.notna().any(axis=1)
        w = w[ok].fillna(0.0)
        gross = (w * r.reindex(w.index).fillna(0.0)).sum(axis=1)
        tov = turnover(w)
        return gross, net_returns(gross, tov, COST), tov, n[ok]

    arms = {"sin_filtro": ew(elig), "filtrado": ew(elig & (f == 0)), "oraculo": ew(elig & (o == 0))}
    out: dict = {"nombre_fechas_con_calendario": int(elig.any(axis=1).sum()),
                 "frac_marcados_media": float(((f == 1) & r.notna()).sum().sum() / elig.sum().sum())}
    for per in ("2008-2017", "2018-2026"):
        net = {k: cut(v[1], per) for k, v in arms.items()}
        row = {"semanas": int(len(net["sin_filtro"]))}
        for k in net:
            row[k] = {"neto_medio_pct": float(net[k].mean() * 100), "std_pct": float(net[k].std() * 100),
                      "rotacion": float(cut(arms[k][2], per).mean()), "n_nombres_medio": float(cut(arms[k][3], per).mean())}
        for k in ("filtrado", "oraculo"):
            sd = sharpe_diff_ci(net[k], net["sin_filtro"], periods_per_year=PPY)
            row[f"dSharpe_{k}"] = sd
            row[f"dMedia_{k}_pct"] = float((net[k] - net["sin_filtro"]).mean() * 100)
        out[per] = row
    t, p = out["2018-2026"], out["2008-2017"]
    cond = {"dSharpe_ci_lo>0_2018_26": bool(t["dSharpe_filtrado"]["ci_lo"] > 0),
            "dMedia>=-0.05pct": bool(t["dMedia_filtrado_pct"] >= -0.05),
            "mismo_signo_2008_17": bool(p["dSharpe_filtrado"]["delta"] > 0)}
    out["veredicto"] = {"condiciones": cond, "pasa": bool(all(cond.values()))}
    return out


# ── (b1) volatilidad ─────────────────────────────────────────────────────────

def part_b1(eps: pd.DataFrame, n_splits: int, n_boot: int) -> dict:
    from run_vol_regressor_study import TARGETS, build_panel, load_raws
    raws = {k: v for k, v in load_raws().items() if f"{k}.MX" in set(eps["ticker"])}
    vc.prepare(raws)
    hist = {s: g for s, g in eps.groupby("ticker")}

    def fn(df):
        sym = df.attrs.get("symbol")
        h = hist.get(sym)
        return earnings_eps_features(df.index, h) if h is not None else pd.DataFrame(
            np.nan, index=df.index, columns=["earn_in5", "earn_in21", "earn_days"])

    if "earnings_eps" not in vr.REGRESSORS:
        vr.register(vr.Regressor("earnings_eps", ("earn_in5", "earn_in21", "earn_days"), "ticker", fn))
    panel = build_panel(raws, ["earnings_eps"])
    cols = list(vr.REGRESSORS["earnings_eps"].cols)
    out: dict = {"n_tickers": int(panel["ticker"].nunique()), "filas": int(len(panel))}
    for h, tgt in TARGETS.items():
        r = vr.evaluate_regressor(panel, tgt, h, cols, n_splits=n_splits, n_boot=n_boot, seed=0)
        r.pop("fold_deltas", None)
        r["pasa"] = bool(r["delta_skill_pooled"] >= vr.MIN_DELTA_SKILL and r["delta_ci_lo"] > 0)
        io = vc.skill_in_out(panel, tgt, h, cols, f"earn_in{h}", n_splits=n_splits)
        out[f"h{h}"] = {**r, "in_out": io}
    out["veredicto"] = {"pasa": bool(out["h5"]["pasa"] or out["h21"]["pasa"])}
    return out


# ── (c) PEAD diagnóstico ─────────────────────────────────────────────────────

def part_c(eps: pd.DataFrame) -> dict:
    rows = []
    for tk, g in eps.groupby("ticker"):
        f = rv.MARKET / f"{tk}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        df = df[~df.index.duplicated()].sort_index().loc[rv.START:]
        if len(df) < 1500:
            continue
        idx = df.index
        cl, op, vo = df["Close"].to_numpy(float), df["Open"].to_numpy(float), df["Volume"].to_numpy(float)
        for _, e in g.iterrows():
            p = int(idx.searchsorted(e["available_from"], side="left"))
            if p < 1 or p + 5 >= len(idx) or (idx[p] - e["available_from"]).days > 5 or pd.isna(e["surprise"]):
                continue
            if not (cl[p] > 0 and cl[p + 5] > 0):
                continue
            w = slice(p - 1, p + 6)
            limpio = bool(np.all(op[w] > 0) and np.all(vo[w] > 0) and np.all(np.isfinite(op[w]))
                          and np.all(op[w] != cl[w]))
            rows.append({"ticker": tk, "date": idx[p], "surprise": e["surprise"], "ret5": cl[p + 5] / cl[p] - 1,
                         "limpio": limpio, "season": f"{e['announced'].year}Q{(e['announced'].month - 1) // 3 + 1}"})
    ev = pd.DataFrame(rows)
    out: dict = {"eventos": int(len(ev)), "eventos_open_limpio": int(ev["limpio"].sum())}
    for label, sub in (("todos", ev), ("open_limpio", ev[ev["limpio"]])):
        for per, (lo, hi) in (("2008-2017", (None, "2017-12-31")), ("2018-2026", (SPLIT, None))):
            s = sub[(sub["date"] >= (lo or "1900")) & (sub["date"] <= (hi or "2100"))].copy()
            n_s = s.groupby("season")["surprise"].transform("size")
            s = s[n_s >= 10]
            s["q"] = s.groupby("season")["surprise"].transform(lambda x: np.ceil(x.rank(pct=True, method="first") * 5)).astype(int)
            if s.empty:
                out[f"{label}|{per}"] = None
                continue
            qm = s.groupby("q")["ret5"].mean() * 100
            ss = s.groupby(["season", "q"])["ret5"].mean().unstack()
            sp = (ss[5] - ss[1]).dropna() * 100
            t = float(sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp)))) if len(sp) > 2 and sp.std() > 0 else float("nan")
            out[f"{label}|{per}"] = {"eventos": int(len(s)), "temporadas": int(len(sp)),
                                    "ret_por_quintil_pct": {int(k): float(v) for k, v in qm.items()},
                                    "spread_q5_q1_pct": float(sp.mean()), "spread_t": t,
                                    "q5_menos_costo_0.60_pct": float(qm[5] - 0.6)}
    return out


# ── reporte ──────────────────────────────────────────────────────────────────

def render_md(res: dict) -> str:
    a, b2, c = res["a"], res["b2"], res["c"]
    L = [f"# N5 · Eventos de EPS ({res['fecha']})", "",
         "Criterios en `avances/2026-11-12_N5_preregistro.md`. Ensayos: " + str(res["ensayos"]) + ".", "",
         f"## (a) Reversión con / sin anuncio ({a['n_tickers']} tickers MX con EPS)", "",
         "| periodo | simple | sin evento | con evento | dif (sin−con) [IC95%] |", "|---|---|---|---|---|"]
    for per, r in a["ic"].items():
        d = r["diferencia"]
        L.append(f"| {per} | {r['simple']['mean']:+.4f} (t={r['simple']['t']:.1f}, n={r['simple']['n']}) | "
                 f"{r['sin_evento']['mean']:+.4f} (t={r['sin_evento']['t']:.1f}, n={r['sin_evento']['n']}) | "
                 f"{r['con_evento']['mean']:+.4f} (t={r['con_evento']['t']:.1f}, n={r['con_evento']['n']}) | "
                 f"{d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] (n={d['n']}) |")
    L += ["", f"Tamaño medio del grupo: {a['tamano_medio_grupo']}", "",
          "Top-5 (2018-2026, bruto, % por semana): " + json.dumps(a["baselines_top5_2018_2026"], ensure_ascii=False), "",
          f"**Veredicto (a): {'PASA' if a['veredicto']['pasa'] else 'NO PASA'}** — {a['veredicto']['condiciones']}", ""]
    L += ["## (b) Calendario esperado", "",
          f"Acierto de la fecha esperada (n={res['acierto']['n']}): |err|≤1 {res['acierto']['le1']:.1%}, "
          f"≤2 {res['acierto']['le2']:.1%}, ≤5 {res['acierto']['le5']:.1%} (de días hábiles).", ""]
    if "b1" in res:
        b1 = res["b1"]
        L += ["### (b1) Volatilidad vs HAR", "", "| h | n_obs | skill HAR | Δskill agrupado [IC95%] | DM p | pasa |", "|---|---|---|---|---|---|"]
        for h in (5, 21):
            r = b1[f"h{h}"]
            L.append(f"| {h} | {r['n_obs']:,} | {r['skill_base']:+.4f} | {r['delta_skill_pooled']:+.4f} "
                     f"[{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}] | {r['dm_p']:.3f} | {r['pasa']} |")
        L += ["", f"**Veredicto (b1): {'PASA' if b1['veredicto']['pasa'] else 'NO PASA'}**", ""]
    L += ["### (b2) Filtro de riesgo (EW semanal, neto 0.30%/lado)", "",
          "| periodo | sin filtro (media%, sd%) | filtrado | oráculo | ΔSharpe filtrado [IC95%] | ΔMedia% |", "|---|---|---|---|---|---|"]
    for per in ("2008-2017", "2018-2026"):
        r = b2[per]
        d = r["dSharpe_filtrado"]
        f = lambda k: f"{r[k]['neto_medio_pct']:+.3f}, {r[k]['std_pct']:.2f}"  # noqa: E731
        L.append(f"| {per} | {f('sin_filtro')} | {f('filtrado')} | {f('oraculo')} | "
                 f"{d['delta']:+.2f} [{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}] | {r['dMedia_filtrado_pct']:+.3f} |")
    L += ["", f"Fracción de nombre-semanas marcadas: {b2['frac_marcados_media']:.1%}.",
          f"**Veredicto (b2): {'PASA' if b2['veredicto']['pasa'] else 'NO PASA'}** — {b2['veredicto']['condiciones']}", "",
          "## (c) PEAD diagnóstico (entrada Close d+1, 5 días, quintil de sorpresa por trimestre)", "",
          f"Eventos: {c['eventos']} (Open limpio: {c['eventos_open_limpio']}).", "",
          "| muestra | periodo | eventos | temporadas | ret% Q1..Q5 | spread Q5−Q1 % (t) | Q5 − 0.60% |", "|---|---|---|---|---|---|---|"]
    for k, v in c.items():
        if "|" not in k or v is None:
            continue
        lab, per = k.split("|")
        q = " / ".join(f"{v['ret_por_quintil_pct'][i]:+.2f}" for i in range(1, 6))
        L.append(f"| {lab} | {per} | {v['eventos']} | {v['temporadas']} | {q} | "
                 f"{v['spread_q5_q1_pct']:+.2f} ({v['spread_t']:.1f}) | {v['q5_menos_costo_0.60_pct']:+.2f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-vol", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--splits", type=int, default=6)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    eps = load_eps()
    O, C, cols, sig, fwd, ev = build(eps)
    print(f"{len(cols)} tickers con EPS y precios | {C.index.min().date()} → {C.index.max().date()}")
    a, _ = part_a(C, cols, sig, fwd, ev)
    print("(a) listo")
    errs = pd.concat([expected_accuracy(g) for _, g in eps.groupby("ticker")], ignore_index=True)
    acierto = {"n": int(len(errs)), "le1": float((errs.abs() <= 1).mean()), "le2": float((errs.abs() <= 2).mean()),
               "le5": float((errs.abs() <= 5).mean())}
    b2 = part_b2(eps, C, cols, fwd)
    print("(b2) listo")
    res = {"fecha": datetime.now().strftime("%Y-%m-%d"), "a": a, "acierto": acierto, "b2": b2}
    if not args.skip_vol:
        res["b1"] = part_b1(eps, args.splits, args.boot)
        print("(b1) listo")
    res["c"] = part_c(eps)
    print("(c) listo")

    trials = [("a_simple", a["ic"]["2018-2026"]["simple"]), ("a_sin_evento", a["ic"]["2018-2026"]["sin_evento"]),
              ("a_con_evento", a["ic"]["2018-2026"]["con_evento"]), ("a_diferencia", a["ic"]["2018-2026"]["diferencia"]),
              ("b2_filtro", b2["2018-2026"]["dSharpe_filtrado"])]
    if "b1" in res:
        trials += [(f"b1_h{h}", {k: res["b1"][f"h{h}"][k] for k in ("delta_skill_pooled", "delta_ci_lo", "delta_ci_hi", "dm_p")})
                   for h in (5, 21)]
    for k in ("todos|2018-2026", "open_limpio|2018-2026"):
        if res["c"].get(k):
            trials.append((f"c_{k.split('|')[0]}", {m: res["c"][k][m] for m in ("spread_q5_q1_pct", "spread_t", "eventos")}))
    res["ensayos"] = len(trials)
    if not args.no_log:
        for name, m in trials:
            log_trial(f"{FAMILY}:{name}", {"estudio": name, "preregistro": "2026-11-12_N5", "periodo": "2018-2026"},
                      {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.items()},
                      int(m.get("n", m.get("eventos", 0)) or 0))
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "eps_events_study.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    md = render_md(res)
    (EVAL / "eps_events_study.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
