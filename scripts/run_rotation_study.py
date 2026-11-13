"""
M5 · Estudio de rotación: histéresis, salida holgada y rebalanceo lento.

Uso:
    .venv/bin/python scripts/run_rotation_study.py          # completo (entrena p_up, ~40 s)
    .venv/bin/python scripts/run_rotation_study.py --fast   # 16 tickers

Rejilla FIJADA de antemano: rebalanceo {5,10,21} × salida {K,1.5K,2K} × K {5,10} × regla
{p_up del modelo, −ret_5d} = 36 × 2 = 36 celdas totales (18 por regla = ensayos).
Convenciones: ``fwd_ret`` del panel en % → fracciones; periodo = 5 días hábiles
(``ppy = 50.4``, también con rebalanceo lento: se mide el retorno semanal del portafolio
mantenido); 21 días se aproxima con 4 periodos (20 días hábiles). Sin deriva de pesos.
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

from scripts.backtest_topk import HORIZON, build_test_panel, load_universe_tickers  # noqa: E402
from scripts.run_m1_report import rebalance_panel, sharpe  # noqa: E402
from src.ml.shared.evaluators.cross_section import ic_by_date, ic_summary  # noqa: E402
from src.ml.shared.evaluators.costs import breakeven_cost_per_side, net_returns, turnover  # noqa: E402
from src.ml.shared.evaluators.portfolio_rules import (  # noqa: E402
    equal_weight_weights,
    held_returns,
    rebalance_step,
    select_with_hysteresis,
)

PPY = 252 / HORIZON
FREQS = (5, 10, 21)
KS = (5, 10)
EXIT_MULT = (1.0, 1.5, 2.0)
COSTS = (0.001, 0.0015, 0.003, 0.005)
BASE_COST = 0.003
ROT_MAX = 0.30
SEED = 42


def wide(rp: pd.DataFrame, col: str) -> pd.DataFrame:
    """Matriz fecha × ticker de una columna del panel de rebalanceo."""
    return rp.reset_index(names="date").pivot_table(index="date", columns="ticker", values=col, aggfunc="last")


def run_cell(scores: pd.DataFrame, ret: pd.DataFrame, k: int, exit_rank: int, freq: int) -> dict:
    """Una celda de la rejilla: estrategia vs equiponderado a la misma frecuencia."""
    w = select_with_hysteresis(scores, k, exit_rank, freq)
    ew = equal_weight_weights(scores, freq)
    out = {"freq_days": freq, "k": k, "exit_rank": exit_rank, "step_periods": rebalance_step(freq)}
    for tag, wts in (("strat", w), ("ew", ew)):
        g, t = held_returns(wts, ret), turnover(wts)
        out[tag] = {
            "turnover_mean": float(t.mean()),
            "gross_mean": float(g.mean()),
            "sharpe_gross": sharpe(g),
            "breakeven_per_side": breakeven_cost_per_side(g, t),
            "net": {
                f"{c:.4f}": {"mean": float(net_returns(g, t, c).mean()),
                             "sharpe": sharpe(net_returns(g, t, c))}
                for c in COSTS
            },
        }
    s, e = out["strat"], out["ew"]
    exceso = held_returns(w, ret) - held_returns(ew, ret)
    out["excess_gross_mean"] = float(exceso.mean())
    out["excess_gross_t"] = float(exceso.mean() / (exceso.std(ddof=1) / np.sqrt(len(exceso))))
    out["excess_net_mean_030"] = s["net"]["0.0030"]["mean"] - e["net"]["0.0030"]["mean"]
    # Criterio fijado: bruto cubre el costo base (equilibrio ≥ 0.30%/lado) con rotación ≤ 30%.
    out["passes"] = bool(s["turnover_mean"] <= ROT_MAX and s["breakeven_per_side"] >= BASE_COST)
    return out


def build_study(fast: bool) -> dict:
    tickers = load_universe_tickers("bmv")
    if fast:
        tickers = dict(list(tickers.items())[:16])
    test, info = build_test_panel("bmv", "p_up", HORIZON, "open", tickers=tickers)
    rp = rebalance_panel(test)
    rp["neg_ret5"] = -rp["ret_5d"]
    ret = wide(rp, "ret")
    reglas = {"p_up": wide(rp, "p_up"), "neg_ret_5d": wide(rp, "neg_ret5")}
    grid, ic = [], {}
    rng = np.random.default_rng(SEED)
    jitter = pd.DataFrame(rng.uniform(0, 1e-9, size=ret.shape), index=ret.index, columns=ret.columns)
    for name, sc in reglas.items():
        # p_up tiene empates masivos: desempate aleatorio (semilla fija), no por orden de columnas
        sc = sc.reindex(index=ret.index, columns=ret.columns) + jitter
        full = ic_summary(ic_by_date(rp, "p_up" if name == "p_up" else "neg_ret5", "ret"),
                          step=1, seed=0)
        ic[name] = {"all_weeks": full}
        for f in FREQS:
            sub = ic_by_date(rp, "p_up" if name == "p_up" else "neg_ret5", "ret")
            ic[name][f"every_{f}d"] = ic_summary(sub, step=rebalance_step(f), seed=0)
        for k in KS:
            for m in EXIT_MULT:
                for f in FREQS:
                    cell = run_cell(sc, ret, k, int(round(m * k)), f)
                    cell["rule"], cell["exit_mult"] = name, m
                    grid.append(cell)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fast": fast, "seed": SEED, "model": info, "n_periods": int(len(ret)),
        "ppy": PPY, "costs_per_side": list(COSTS), "base_cost": BASE_COST, "rot_max": ROT_MAX,
        "criterion": "rotación media/periodo ≤ 30% y costo de equilibrio ≥ 0.30%/lado",
        "trials": len(grid), "ic": ic, "grid": grid,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="M5 · estudio de rotación")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    np.random.seed(SEED)
    rep = build_study(args.fast)
    path = ROOT / "data" / "eval" / ("rotation_study_fast.json" if args.fast else "rotation_study.json")
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"Ensayos: {rep['trials']} · periodos: {rep['n_periods']} · {path}")
    for r in rep["ic"]:
        a = rep["ic"][r]["all_weeks"]
        print(f"IC {r}: {a['mean']:+.3f} (t={a['t']:.2f}, n={a['n']})")
    print(f"{'regla':11}{'f':>3}{'K':>3}{'sal':>3} | {'rot':>5}{'bruto%':>8}{'eq%':>7}{'EWrot':>6}{'EWbr%':>7}"
          f"{'exc t':>6}{'shN30':>7}{'ok':>3}")
    for c in rep["grid"]:
        s, e = c["strat"], c["ew"]
        print(f"{c['rule']:11}{c['freq_days']:>3}{c['k']:>3}{c['exit_rank']:>3} | {s['turnover_mean']:5.2f}"
              f"{s['gross_mean']*100:8.3f}{s['breakeven_per_side']*100:7.3f}{e['turnover_mean']:6.2f}"
              f"{e['gross_mean']*100:7.3f}{c['excess_gross_t']:6.2f}{s['net']['0.0030']['sharpe']:7.2f}"
              f"{'Y' if c['passes'] else '-':>3}")


if __name__ == "__main__":
    main()
