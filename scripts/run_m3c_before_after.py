"""
M3c · Antes/después de limpiar el panel (máscara de calidad + guardián de Open/fwd_ret).

Corre lo mismo dos veces (``use_mask=False`` y ``True``), con el mismo universo y semillas:
  (a) p_up del ensemble: IC y AUC por fecha en el test, y Sharpe bruto/neto del top-5 (como M1d).
  (b) bake-off de M9 reducido: xs_lgbm, Ridge y los 4 brazos anti-ruido (IC por fecha, 5 folds).
  (c) rejilla de M5, solo la celda 10d / K=10 / salida 2K (rank 20), con p_up.

Importa funciones de ``run_m1_report``, ``run_model_bakeoff`` y ``run_rotation_study`` (solo
lectura; no escribe sus archivos de salida). Semilla 42 en el ensemble, en el desempate de M5 y
en el bootstrap; los brazos de M9 usan sus semillas por defecto (0–4).

Uso:
    .venv/bin/python scripts/run_m3c_before_after.py            # completo (~10 min)
    .venv/bin/python scripts/run_m3c_before_after.py --fast     # 16 tickers
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from scripts.backtest_topk import (  # noqa: E402
    DROP_RULES, HORIZON, SEED, build_panel, build_test_panel, load_universe_tickers,
    run_topk_backtest, summarize_mask_drops,
)
from scripts.run_m1_report import FAST_TICKERS, TOP_K, ic_block, rebalance_panel  # noqa: E402
from scripts.run_model_bakeoff import N_SPLITS, REF, arm_table, build_arms, run_folds  # noqa: E402
from scripts.run_rotation_study import run_cell, wide  # noqa: E402
from src.ml.features import FEATURE_COLS  # noqa: E402
from src.ml.kb_enhance import select_priority_rn_cols  # noqa: E402
from src.ml.purged_cv import purged_walk_forward_date_splits  # noqa: E402
from src.ml.rank_models import EMBARGO, rank_features  # noqa: E402

N_BOOT = 2000
ROT_K, ROT_EXIT, ROT_FREQ = 10, 20, 10          # celda 10d / K10 / salida 2K
BAKEOFF_ARMS = ("xs_lgbm", "ridge_rank")        # + los 4 antinoise_*


def part_a(test: pd.DataFrame, info: dict) -> dict:
    """IC/AUC por fecha de p_up y top-5 (Sharpe bruto/neto con turnover real)."""
    blk = ic_block(test, "p_up", "y", N_BOOT)
    bt = run_topk_backtest(test, horizon=HORIZON, top_k=TOP_K, score_mode="p_up",
                           weight_mode="equal", min_p_up=0.0, cost_mode="turnover")
    s = bt["strategy"]
    return {
        "ic": blk["ic"], "auc_mean": blk["auc_mean"], "auc_ic": blk["auc_by_date"],
        "model_auc_global": info["auc"], "n_train": info["n_train"], "n_test": info["n_test"],
        "test_start": info["test_start"], "train_end": info["train_end"],
        "top5": {
            "sharpe_gross": s["sharpe_gross"], "sharpe_net": s["sharpe_net"],
            "total_return_net": s["total_return_net"], "n_periods": s["n_periods"],
            "turnover_mean": bt["turnover_mean"], "breakeven_per_side": bt["breakeven_cost_per_side"],
        },
    }


def part_c(test: pd.DataFrame) -> dict:
    """Celda de M5 (10d, K10, salida 2K) con p_up; mismo desempate aleatorio que run_rotation_study."""
    rp = rebalance_panel(test)
    ret = wide(rp, "ret")
    rng = np.random.default_rng(SEED)
    jitter = pd.DataFrame(rng.uniform(0, 1e-9, size=ret.shape), index=ret.index, columns=ret.columns)
    sc = wide(rp, "p_up").reindex(index=ret.index, columns=ret.columns) + jitter
    cell = run_cell(sc, ret, ROT_K, ROT_EXIT, ROT_FREQ)
    s, e = cell["strat"], cell["ew"]
    return {
        "n_periods": int(len(ret)), "turnover_mean": s["turnover_mean"], "gross_mean": s["gross_mean"],
        "breakeven_per_side": s["breakeven_per_side"], "sharpe_gross": s["sharpe_gross"],
        "sharpe_net_030": s["net"]["0.0030"]["sharpe"], "ew_sharpe_gross": e["sharpe_gross"],
        "excess_gross_t": cell["excess_gross_t"], "passes": cell["passes"],
    }


def part_b(tickers: dict, use_mask: bool, fast: bool) -> dict:
    """Bake-off reducido de M9: xs_lgbm, Ridge y los 4 anti-ruido, 5 folds purgados."""
    drops: list = []
    panel = build_panel(tickers, horizon=HORIZON, use_open_entry=True, use_mask=use_mask, drops_out=drops)
    panel = panel.sort_index(kind="stable")
    panel = rank_features(panel, FEATURE_COLS)
    panel["pct_fwd"] = panel.groupby(panel.index)["fwd_ret"].rank(pct=True)
    cols = select_priority_rn_cols([f"rn_{c}" for c in FEATURE_COLS])
    panel = panel.dropna(subset=cols + ["pct_fwd"])
    folds = list(purged_walk_forward_date_splits(panel.index, n_splits=N_SPLITS, horizon=HORIZON, embargo=EMBARGO))
    arms = [a for a in build_arms(cols, fast) if a.name in BAKEOFF_ARMS or a.name.startswith("antinoise")]
    oos, _ = run_folds(panel, arms, folds)
    names = [a.name for a in arms] + [REF]
    table, _ = arm_table(oos, names)
    anti = [a for a in names if a.startswith("antinoise")]
    best = max(anti, key=lambda a: table[a]["ic"]["mean"])      # selección sobre test (se declara)
    pick = lambda a: {"ic": table[a]["ic"]["mean"], "t": table[a]["ic"]["t"], "se": table[a]["ic"]["se"],
                      "auc_mean": table[a]["auc_mean"]}
    return {
        "n_rows": int(len(panel)), "n_arms": len(arms),
        "xs_lgbm": pick("xs_lgbm"), "ridge": pick("ridge_rank"), "mejor_antiruido": {"arm": best, **pick(best)},
        "todos": {a: pick(a) for a in names},
        "filas_descartadas_total": int(len(drops[0])) if drops else 0,
    }


def run_one(tickers: dict, use_mask: bool, fast: bool) -> dict:
    t0 = time.time()
    test, info = build_test_panel("bmv", "p_up", HORIZON, "open", tickers=tickers, use_mask=use_mask)
    out = {"use_mask": use_mask, "A_p_up": part_a(test, info), "C_rotacion_10d_K10_2K": part_c(test),
           "max_abs_fwd_ret_test": float(test["fwd_ret"].abs().max())}
    if "mask_drops" in info:
        out["descartes"] = info["mask_drops"]
    print(f"[use_mask={use_mask}] A y C listos ({time.time() - t0:.0f}s); bake-off…", flush=True)
    out["B_bakeoff"] = part_b(tickers, use_mask, fast)
    out["segundos"] = round(time.time() - t0, 1)
    return out


def guardian_only(tickers: dict) -> dict:
    """Filas que solo el guardián (no la máscara) elimina, sobre todo el panel."""
    out: list = []
    build_panel(tickers, use_mask=True, drops_out=out)
    d = out[0]
    solo = d[~d["mask_invalid"] & (d["open_nonpos"] | d["fwd_ret_outlier"])]
    return {"filas_solo_guardian": int(len(solo)), "tickers": sorted(solo["ticker"].unique().tolist()),
            "por_regla_total": {r: int(d[r].sum()) for r in DROP_RULES}, "filas_total": int(len(d))}


# ── markdown ─────────────────────────────────────────────────────────────────

def _f(x, n=3, sign=True):
    return "n/d" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def render_md(r: dict) -> str:
    a0, a1 = r["antes"], r["despues"]
    A0, A1 = a0["A_p_up"], a1["A_p_up"]
    B0, B1 = a0["B_bakeoff"], a1["B_bakeoff"]
    C0, C1 = a0["C_rotacion_10d_K10_2K"], a1["C_rotacion_10d_K10_2K"]
    d = a1.get("descartes", {})
    rows = [
        ("IC p_up (test)", _f(A0["ic"]["mean"]), _f(A1["ic"]["mean"]), f"EE {A0['ic']['se']:.3f}"),
        ("AUC medio por fecha p_up", _f(A0["auc_mean"], 4, False), _f(A1["auc_mean"], 4, False), ""),
        ("Top-5 Sharpe bruto", _f(A0["top5"]["sharpe_gross"], 2), _f(A1["top5"]["sharpe_gross"], 2), ""),
        ("Top-5 Sharpe neto @0.30%", _f(A0["top5"]["sharpe_net"], 2), _f(A1["top5"]["sharpe_net"], 2), ""),
        ("Top-5 rotación / equilibrio %/lado", f"{A0['top5']['turnover_mean']:.2f} / {A0['top5']['breakeven_per_side']*100:.3f}",
         f"{A1['top5']['turnover_mean']:.2f} / {A1['top5']['breakeven_per_side']*100:.3f}", "base 0.30"),
        ("IC xs_lgbm (M9)", _f(B0["xs_lgbm"]["ic"]), _f(B1["xs_lgbm"]["ic"]), f"EE {B0['xs_lgbm']['se']:.3f}"),
        ("IC Ridge (M9)", _f(B0["ridge"]["ic"]), _f(B1["ridge"]["ic"]), f"EE {B0['ridge']['se']:.3f}"),
        (f"IC mejor anti-ruido ({B0['mejor_antiruido']['arm'][10:]} / {B1['mejor_antiruido']['arm'][10:]})",
         _f(B0["mejor_antiruido"]["ic"]), _f(B1["mejor_antiruido"]["ic"]), "elegido sobre test"),
        ("M5 10d/K10/2K · rotación", f"{C0['turnover_mean']:.2f}", f"{C1['turnover_mean']:.2f}", "tope 0.30"),
        ("M5 · bruto %/sem · equilibrio %/lado", f"{C0['gross_mean']*100:.3f} / {C0['breakeven_per_side']*100:.3f}",
         f"{C1['gross_mean']*100:.3f} / {C1['breakeven_per_side']*100:.3f}", "base 0.30"),
        ("M5 · Sharpe neto@0.30 · exceso t vs EW", f"{C0['sharpe_net_030']:.2f} · {C0['excess_gross_t']:.2f}",
         f"{C1['sharpe_net_030']:.2f} · {C1['excess_gross_t']:.2f}", ""),
        ("M5 · pasa la puerta", str(C0["passes"]), str(C1["passes"]), ""),
        ("max |fwd_ret| en test (%)", f"{a0['max_abs_fwd_ret_test']:.1f}", f"{a1['max_abs_fwd_ret_test']:.1f}", ""),
    ]
    L = ["# M3c · Panel limpio: antes / después", "",
         f"Semilla {SEED} (ensemble, desempate M5, bootstrap); brazos M9 con semillas 0–4. Universo bmv, "
         f"{'16' if r['fast'] else 'todos los'} tickers. Ensayos: {r['ensayos']}.", "",
         "| Métrica | Sin máscara | Con máscara | Nota |", "|---|---|---|---|"]
    L += [f"| {m} | {x} | {y} | {n} |" for m, x, y, n in rows]
    L += ["", "**Filas descartadas** (reglas independientes; solo filas que habrían entrado al panel): "
          f"train {d.get('train', {}).get('rows_total', 0)} filas / {d.get('train', {}).get('tickers_total', 0)} tickers; "
          f"test {d.get('test', {}).get('rows_total', 0)} / {d.get('test', {}).get('tickers_total', 0)}."]
    for tag in ("train", "test"):
        if tag in d:
            L.append(f"- {tag}: " + ", ".join(f"{k} {d[tag][k]['rows']} filas/{d[tag][k]['tickers']} tick." for k in DROP_RULES))
    g = r["guardian"]
    L.append(f"- Solo el guardián (no la máscara): {g['filas_solo_guardian']} filas ({', '.join(g['tickers']) or 'ninguna'}).")
    L += ["", "**Conclusión**", r["conclusion"]]
    return "\n".join(L) + "\n"


def conclusion(r: dict) -> str:
    a0, a1 = r["antes"], r["despues"]
    dic = a1["A_p_up"]["ic"]["mean"] - a0["A_p_up"]["ic"]["mean"]
    se = a0["A_p_up"]["ic"]["se"]
    dr = a1["B_bakeoff"]["ridge"]["ic"] - a0["B_bakeoff"]["ridge"]["ic"]
    seb = a0["B_bakeoff"]["ridge"]["se"]
    pasa0, pasa1 = a0["C_rotacion_10d_K10_2K"]["passes"], a1["C_rotacion_10d_K10_2K"]["passes"]
    n0, n1 = a0["A_p_up"]["top5"]["sharpe_net"], a1["A_p_up"]["top5"]["sharpe_net"]
    chico = abs(dic) < se and abs(dr) < seb
    txt = (f"El IC de p_up cambia {dic:+.3f} (EE {se:.3f}) y el del Ridge {dr:+.3f} (EE {seb:.3f}); "
           f"el Sharpe neto del top-5 pasa de {n0:+.2f} a {n1:+.2f}; la celda de M5 "
           f"{'sigue sin pasar la puerta' if not pasa0 and not pasa1 else 'sigue pasando la puerta de rotación/equilibrio' if pasa0 and pasa1 else 'cambia su veredicto de la puerta'}. ")
    txt += ("Los cambios son menores que el error estándar: la basura no explicaba el resultado y la lectura "
            "no cambia." if chico and pasa0 == pasa1 else
            "Algún cambio supera un error estándar o altera la puerta: no es un cambio de lectura automático; "
            "ver el reporte para la interpretación.")
    return txt


def main() -> None:
    ap = argparse.ArgumentParser(description="M3c · antes/después de limpiar el panel")
    ap.add_argument("--fast", action="store_true", help=f"{FAST_TICKERS} tickers, brazos anti-ruido rápidos")
    args = ap.parse_args()
    np.random.seed(SEED)
    tickers = load_universe_tickers("bmv")
    if args.fast:
        tickers = dict(list(tickers.items())[:FAST_TICKERS])

    antes = run_one(tickers, False, args.fast)
    despues = run_one(tickers, True, args.fast)
    rep = {"generated_at": datetime.now().isoformat(timespec="seconds"), "fast": args.fast, "seed": SEED,
           "n_tickers": len(tickers), "ensayos": 2 * (1 + 6 + 1),
           "ensayos_detalle": "2 corridas × (1 ensemble p_up + 6 brazos M9 + 1 celda M5); sin búsqueda propia",
           "antes": antes, "despues": despues, "guardian": guardian_only(tickers)}
    rep["conclusion"] = conclusion(rep)
    suf = "_fast" if args.fast else ""
    out = ROOT / "data" / "eval"
    (out / f"m3c_before_after{suf}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    md = render_md(rep)
    (out / f"m3c_before_after{suf}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
