"""
Prueba fuera de muestra v2: reglas de la v1 congeladas, ventana 2026-06-19 → 2026-10-30.

Criterio fijado ANTES de correr: ``avances/2026-11-09_oos_v2_preregistro.md``.

Uso:
    .venv/bin/python scripts/run_oos_v2.py            # completo, registra 5 ensayos
    .venv/bin/python scripts/run_oos_v2.py --no-log   # no escribe en la bitácora

Nada se reajusta: Ridge alpha=100, ensemble top_features=15, celda M5 10d/K10/salida 2K,
REV = 5 peores por ret_5d semanal. "frozen" entrena con la ventana de la v1 (train < 2023-03-09);
"refit" usa la misma receta con datos hasta ``REFIT_END`` (embargo ≈ horizonte + Open t+1).
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

from scripts.backtest_topk import (  # noqa: E402
    HORIZON, build_panel, load_universe_tickers, model_feature_cols,
)
from scripts.run_m1_report import rebalance_panel  # noqa: E402
from scripts.run_rotation_study import run_cell, wide  # noqa: E402
from src.ml.cs_targets import build_variant  # noqa: E402
from src.ml.ensemble import EnsembleModel  # noqa: E402
from src.ml.features import FEATURE_COLS  # noqa: E402
from src.ml.kb_enhance import select_priority_rn_cols  # noqa: E402
from src.ml.rank_models import RidgeArm, rank_features  # noqa: E402
from src.ml.shared.evaluators.cross_section import ic_by_date, ic_summary, percentile_in, random_topk_returns  # noqa: E402
from src.ml.trials import log_trial  # noqa: E402

SEED = 42
FROZEN_CUT = pd.Timestamp("2023-03-09")   # inicio del test de la v1
REFIT_END = pd.Timestamp("2026-06-10")    # train < esta fecha en "refit"
OOS_START = pd.Timestamp("2026-06-19")
COST = "0.0030"
V1_EXCESS_NET = 0.00145                   # exceso neto/sem de la celda M5 p_up en la v1
LEAK_IC = 0.10
FAMILY = "oos_v2"


def cell_metrics(rp: pd.DataFrame, score_col: str, k: int, exit_rank: int, freq: int, gross_only: bool) -> dict:
    """IC por fecha + celda de cartera (misma maquinaria que M5) + percentil vs top-K aleatorio."""
    rng = np.random.default_rng(SEED)
    ret = wide(rp, "ret")
    sc = wide(rp, score_col).reindex(index=ret.index, columns=ret.columns)
    sc = sc + pd.DataFrame(rng.uniform(0, 1e-9, size=ret.shape), index=ret.index, columns=ret.columns)
    c = run_cell(sc, ret, k, exit_rank, freq)
    ic = ic_summary(ic_by_date(rp, score_col, "ret"), step=1, seed=0)
    s, e = c["strat"], c["ew"]
    draws = random_topk_returns(rp, "ret", k, n_draws=500, seed=0)
    pct = percentile_in(float(s["gross_mean"]), draws.mean(axis=0)) if len(draws) else float("nan")
    out = {
        "ic": ic, "n_periods": int(len(ret)), "turnover": s["turnover_mean"],
        "gross_pct": s["gross_mean"] * 100, "ew_gross_pct": e["gross_mean"] * 100,
        "net030_pct": s["net"][COST]["mean"] * 100, "ew_net030_pct": e["net"][COST]["mean"] * 100,
        "excess_net030_pct": c["excess_net_mean_030"] * 100,
        "excess_gross_pct": (s["gross_mean"] - e["gross_mean"]) * 100,
        "excess_gross_t": c["excess_gross_t"], "breakeven_per_side_pct": s["breakeven_per_side"] * 100,
        "sharpe_net030": s["net"][COST]["sharpe"], "pctile_vs_random_topk": pct,
    }
    out["verdict"] = classify(out, gross_only)
    return out


def classify(m: dict, gross_only: bool) -> dict:
    """Criterio del pre-registro. gross_only=True para REV (S2' compara bruto)."""
    s1 = m["ic"]["mean"] > 0
    if gross_only:
        s2 = m["excess_gross_pct"] > 0
        fail = m["ic"]["mean"] <= 0
    else:
        s2 = m["excess_net030_pct"] > 0
        fail = m["ic"]["mean"] <= 0 or m["excess_net030_pct"] / 100 < -0.5 * V1_EXCESS_NET
    label = "falla grosera" if fail else ("se sostiene" if (s1 and s2) else "inconcluso")
    return {"S1": bool(s1), "S2": bool(s2), "label": label, "leak_alarm": bool(m["ic"]["mean"] > LEAK_IC)}


def main() -> None:
    ap = argparse.ArgumentParser(description="OOS v2")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    np.random.seed(SEED)

    tickers = load_universe_tickers("bmv")
    panel = build_panel(tickers, horizon=HORIZON, use_open_entry=True, use_mask=True).sort_index(kind="stable")
    last_signal = panel.index.max()
    test_mask = panel.index >= OOS_START
    print(f"Panel {panel.shape} · {panel['ticker'].nunique()} tickers · {panel.index.min().date()} → {last_signal.date()}"
          f" · test {int(test_mask.sum())} filas", flush=True)

    # --- Ridge: 24 features rn_* (mismo panel que M10/M2a), objetivo pct_fwd
    ranked = rank_features(panel, FEATURE_COLS)
    rcols = select_priority_rn_cols([f"rn_{c}" for c in FEATURE_COLS])
    ranked = ranked.dropna(subset=rcols)
    ranked, use = build_variant(ranked, "A", np.ones(len(ranked)), rcols)
    ecols = model_feature_cols(panel.columns)

    trains = {"frozen": panel.index < FROZEN_CUT, "refit": panel.index < REFIT_END}
    scores: dict[str, pd.Series] = {}
    meta = {}
    for tag, m in trains.items():
        tr = panel[m]
        assert tr.index.max() < OOS_START, "train mira la ventana OOS"
        model = EnsembleModel(top_features=15, calibrate=True)
        model.fit(tr, feature_cols=ecols, target_col="y")
        te = panel[test_mask]
        scores[f"E-{tag}"] = pd.Series(model.predict_proba(te, ecols), index=te.index)
        rtr = ranked[ranked.index < (FROZEN_CUT if tag == "frozen" else REFIT_END)]
        arm = RidgeArm(use).fit(rtr)
        rte = ranked[ranked.index >= OOS_START]
        scores[f"R-{tag}"] = pd.Series(arm.predict(rte), index=rte.index)
        meta[tag] = {"train_end": str(tr.index.max().date()), "n_train": int(len(tr)),
                     "n_train_ridge": int(len(rtr))}
        print(f"  {tag}: train hasta {meta[tag]['train_end']} ({len(tr)} filas)", flush=True)

    te = panel[test_mask].copy()
    te["_row"] = np.arange(len(te))
    results = {}
    for name, s in scores.items():
        # alinear por (fecha, ticker): el panel Ridge puede tener menos filas (dropna)
        sc = pd.DataFrame({"ticker": (te if name.startswith("E") else ranked[ranked.index >= OOS_START])["ticker"].to_numpy(),
                           "score": s.to_numpy()}, index=s.index)
        d = te.drop(columns="_row").copy()
        d["score"] = np.nan
        key = d.reset_index(names="date").set_index(["date", "ticker"]).index
        sc_key = sc.reset_index(names="date").set_index(["date", "ticker"])["score"]
        sc_key = sc_key[~sc_key.index.duplicated(keep="last")]
        d["score"] = sc_key.reindex(key).to_numpy()
        d = d.dropna(subset=["score"])
        rp = rebalance_panel(d)
        results[name] = cell_metrics(rp, "score", 10, 20, 10, gross_only=False)

    # --- REV: −ret_5d, 5 peores, semanal (cartera M4)
    d = te.drop(columns="_row").copy()
    d["neg_ret5"] = -d["ret_5d"]
    rp = rebalance_panel(d)
    results["REV"] = cell_metrics(rp.assign(neg_ret5=rp["neg_ret5"]), "neg_ret5", 5, 5, 5, gross_only=True)

    n_total = 0
    for name, r in results.items():
        if not args.no_log:
            log_trial(f"{FAMILY}:{name}", {"rule": "M5 10d/K10/2K" if name != "REV" else "5 peores ret_5d semanal",
                                          "window": f"{OOS_START.date()}→{last_signal.date()}", "mask": True,
                                          "train": meta.get(name.split("-")[-1], "n/a") if name != "REV" else "n/a"},
                      {"ic_mean": r["ic"]["mean"], "t": r["ic"]["t"], "excess_net030_pct": r["excess_net030_pct"],
                       "label": r["verdict"]["label"]}, r["ic"]["n"])

    rep = {"generated_at": datetime.now().isoformat(timespec="seconds"), "seed": SEED,
           "preregistro": "avances/2026-11-09_oos_v2_preregistro.md",
           "window": [str(OOS_START.date()), str(last_signal.date())], "meta": meta, "results": results,
           "n_trials_logged": 0 if args.no_log else 5}
    out = ROOT / "data" / "eval" / "oos_v2.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n{'id':9}{'IC':>8}{'t':>6}{'n':>4}{'rot':>6}{'netE%':>8}{'netEW%':>8}{'exc%':>8}{'pctRnd':>7}  veredicto")
    for name, r in results.items():
        v = r["verdict"]
        print(f"{name:9}{r['ic']['mean']:+8.3f}{r['ic']['t']:6.2f}{r['ic']['n']:4d}{r['turnover']:6.2f}"
              f"{r['net030_pct']:8.3f}{r['ew_net030_pct']:8.3f}{r['excess_net030_pct']:+8.3f}"
              f"{r['pctile_vs_random_topk']:7.0f}  {v['label']}{' [ALARMA FUGA]' if v['leak_alarm'] else ''}")
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
