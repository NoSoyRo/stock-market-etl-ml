"""
M9 · Bake-off de modelos con IC por fecha.

Uso:
    .venv/bin/python scripts/run_model_bakeoff.py --fast   # 16 tickers, 2 semillas (~minutos)
    .venv/bin/python scripts/run_model_bakeoff.py          # completo (lento: ~15–25 min)

Diseño FIJADO ANTES de ver resultados:
- Panel de ``build_panel`` (bmv, entrada Open t+1, horizonte 5) y features ``rn_*`` = percentil
  por fecha (mismas 24 columnas que ``xs_lgbm``). 5 folds de ``purged_walk_forward_date_splits``
  (horizonte 5, embargo 5). Todos los brazos ven los mismos folds.
- Métrica principal: IC de Spearman por fecha (``ic_summary``, step = horizonte, bootstrap por
  bloques); secundaria: AUC por fecha contra ``y_rel``.
- "Gana" un brazo si supera al Ridge en la MAYORÍA de folds (≥ 3 de 5) con diferencia pareada por
  fecha de IC ≥ 2 errores estándar de esa diferencia (EE con step = horizonte). También se reporta
  la versión literal del handoff: diferencia ≥ 2 × 0.007. Si nadie gana: empate, se queda el simple.
- Scores continuos, sin calibración isotónica. Desempate (solo para el portafolio): jitter 1e-9.
- Combinación: brazos fijados a priori (xs_lgbm, rank_xendcg, antinoise_l8_m300, ridge_rank);
  λ = 0.25, ventana 104 semanas (520 fechas) y sensibilidad a 52; rezago horizonte + embargo.
- Cada configuración cuenta como ensayo y se registra en ``data/eval/trials.jsonl`` (no con --fast).
- El "mejor brazo" para los pasos 3–4 se elige por IC agrupado OOS: es selección sobre el test y
  se declara así en el reporte.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.backtest_topk import HORIZON, build_panel, load_universe_tickers  # noqa: E402
from scripts.run_m1_report import rebalance_panel, sharpe, sharpe_cols  # noqa: E402
from scripts.run_rotation_study import run_cell, wide  # noqa: E402
from src.data.store import get as store_get  # noqa: E402
from src.ml.features import FEATURE_COLS  # noqa: E402
from src.ml.kb_enhance import select_priority_rn_cols  # noqa: E402
from src.ml.purged_cv import purged_walk_forward_date_splits  # noqa: E402
from src.ml.rank_models import (  # noqa: E402
    EMBARGO, AntiNoiseArm, MonotoneArm, RankerArm, RidgeArm, XsLgbmArm, blend_scores,
    ic_by_tercile, rank_features, weights_path,
)
from src.ml.shared.evaluators.cross_section import (  # noqa: E402
    auc_by_date, equal_weight_returns, ic_by_date, ic_summary, percentile_in, random_topk_returns,
)
from src.ml.trials import log_trial  # noqa: E402
from src.ml.vol_target import ewma_vol  # noqa: E402

N_SPLITS = 5
SE_HANDOFF = 0.007
WIN_FOLDS = 3          # mayoría de 5
BLEND_ARMS = ("xs_lgbm", "ranker_rank_xendcg", "antinoise_l8_m300", "ridge_rank")
LAM = 0.25
WINDOWS = {"104w": 520, "52w": 260}
SEED = 42
K, EXIT_RANK, FREQS = 5, 10, (5, 10)
REF = "neg_ret_5d"


def build_arms(cols: list[str], fast: bool) -> list:
    seeds = (0, 1) if fast else (0, 1, 2, 3, 4)
    n_max = 100 if fast else 300
    anti = [AntiNoiseArm(cols, nl, mc, seeds=seeds, n_max=n_max)
            for nl in (4, 8) for mc in (300, 1000)]
    return [XsLgbmArm(cols), RankerArm(cols, "lambdarank"), RankerArm(cols, "rank_xendcg"),
            *anti, RidgeArm(cols), MonotoneArm(cols)]


def paired_vs(ic_a: pd.Series, ic_b: pd.Series) -> dict:
    """Diferencia pareada por fecha de IC (a − b); EE con step = horizonte."""
    d = (ic_a - ic_b).dropna()
    s = ic_summary(d, step=HORIZON, seed=0)
    se = s["std"] / np.sqrt(s["n"]) if s["n"] > 1 else float("nan")
    return {"diff": s["mean"], "se": float(se), "t": s["t"], "n": s["n"],
            "wins_2se": bool(s["mean"] >= 2 * se), "wins_fixed": bool(s["mean"] >= 2 * SE_HANDOFF)}


def run_folds(panel: pd.DataFrame, arms: list, folds: list) -> tuple[pd.DataFrame, dict]:
    """Predicciones OOS de cada brazo (columna por brazo) + meta por fold."""
    oos = []
    meta: dict[str, list] = {}
    for fi, (tr, te) in enumerate(folds):
        train, test = panel.iloc[tr], panel.iloc[te]
        part = test[["ticker", "fwd_ret", "y_rel", "ret_5d"]].copy()
        part["fold"] = fi
        for arm in arms:
            t0 = time.time()
            arm.fit(train)
            part[arm.name] = arm.predict(test)
            if hasattr(arm, "best_n_"):
                meta.setdefault(arm.name, []).append(int(arm.best_n_))
            print(f"  fold {fi} · {arm.name:24s} {time.time() - t0:5.1f}s", flush=True)
        part[REF] = -test["ret_5d"].to_numpy()
        oos.append(part)
    return pd.concat(oos).sort_index(), meta


def arm_table(oos: pd.DataFrame, names: list[str]) -> tuple[dict, dict]:
    """IC por fecha (global y por fold), AUC y comparación pareada contra el Ridge."""
    ics = {a: ic_by_date(oos, a, "fwd_ret") for a in names}
    aucs = {a: auc_by_date(oos, "y_rel", a) for a in names}
    out = {}
    for a in names:
        row = {"ic": ic_summary(ics[a], step=HORIZON, seed=0),
               "auc_mean": float(aucs[a].mean()), "folds": {}}
        fold_of = oos.groupby(level=0)["fold"].first()
        for f in sorted(oos["fold"].unique()):
            idx = fold_of.index[fold_of == f]
            sub = ics[a].reindex(idx).dropna()
            r = ic_summary(sub, step=HORIZON, seed=0)
            row["folds"][str(f)] = {"ic_mean": r["mean"], "t": r["t"], "n": r["n"]}
            if a != "ridge_rank":
                row["folds"][str(f)]["vs_ridge"] = paired_vs(sub, ics["ridge_rank"].reindex(idx))
        if a != "ridge_rank":
            row["vs_ridge_pooled"] = paired_vs(ics[a], ics["ridge_rank"])
            nf = sum(v["vs_ridge"]["wins_2se"] for v in row["folds"].values())
            nf_fixed = sum(v["vs_ridge"]["wins_fixed"] for v in row["folds"].values())
            row["folds_won_2se"], row["folds_won_fixed"] = nf, nf_fixed
            row["gana"] = bool(nf >= WIN_FOLDS)
        fm = [v["ic_mean"] for v in row["folds"].values()]
        row["fold_ic_std"] = float(np.std(fm, ddof=1))
        out[a] = row
    return out, ics


def rank_scores(oos: pd.DataFrame, names) -> pd.DataFrame:
    """Score de cada brazo normalizado por fecha (percentil)."""
    return pd.DataFrame({a: oos.groupby(level=0)[a].rank(pct=True) for a in names}, index=oos.index)


def combos(oos: pd.DataFrame, ics: dict) -> dict:
    """Pesos 1/N, AUC, IC con shrinkage (104w/52w) y stacking (walk-forward e in-sample)."""
    sc = rank_scores(oos, BLEND_ARMS)
    ic_h = pd.DataFrame({a: ics[a] for a in BLEND_ARMS}).dropna()
    auc_h = pd.DataFrame({a: auc_by_date(oos, "y_rel", a) for a in BLEND_ARMS}).reindex(ic_h.index)
    dates = ic_h.index
    warm = dates[min(len(dates) - 1, 140)]            # evaluar todos los esquemas desde el mismo punto
    lag = HORIZON + EMBARGO
    schemes: dict[str, pd.Series] = {}
    schemes["equal_1N"] = blend_scores(sc, weights_path(ic_h, "equal"))
    schemes["auc_104w"] = blend_scores(sc, weights_path(auc_h, "auc", window=WINDOWS["104w"]))
    for tag, win in WINDOWS.items():
        schemes[f"ic_shrink_{tag}"] = blend_scores(sc, weights_path(ic_h, "ic", lam=LAM, window=win))
    # Stacking walk-forward (LR ajustada solo con filas realizadas ≤ i − lag) e in-sample (como el repo)
    y = oos["y_rel"].to_numpy()
    pos = pd.Series(np.arange(len(dates)), index=dates).reindex(oos.index).to_numpy()
    st_wf = pd.Series(np.nan, index=oos.index)
    cur = None
    for i in range(len(dates)):
        if i % HORIZON == 0:
            lo, hi = max(0, i - lag + 1 - WINDOWS["104w"]), i - lag + 1
            m = (pos >= lo) & (pos < hi)
            cur = LogisticRegression(C=1.0, max_iter=200).fit(sc.values[m], y[m]) if m.sum() > 1000 else None
        rows = pos == i
        st_wf[rows] = cur.decision_function(sc.values[rows]) if cur is not None else sc.values[rows].mean(axis=1)
    schemes["stack_walkforward"] = st_wf
    insample = LogisticRegression(C=1.0, max_iter=200).fit(sc.values, y)
    schemes["stack_insample (como el repo)"] = pd.Series(insample.decision_function(sc.values), index=oos.index)
    res, ic_s = {}, {}
    for name, s in schemes.items():
        tmp = oos[["fwd_ret"]].copy()
        tmp["s"] = s.to_numpy()
        ic_s[name] = ic_by_date(tmp, "s", "fwd_ret")
    base = ic_s["equal_1N"].loc[warm:]
    for name, ic in ic_s.items():
        ic = ic.loc[warm:]
        res[name] = {"ic": ic_summary(ic, step=HORIZON, seed=0), "vs_equal": paired_vs(ic, base)}
    return res


def sigma_panel(tickers: list[str], span: int = 20) -> pd.Series:
    """σ̂ EWMA (causal) por (fecha, ticker) con los retornos diarios de cierre."""
    parts = []
    for t in tickers:
        raw = store_get(t)
        if raw is None or "Close" not in raw.columns:
            continue
        v = ewma_vol(raw["Close"].pct_change(), span)
        v.index = pd.MultiIndex.from_arrays([v.index, [t] * len(v)], names=["d", "ticker"])
        parts.append(v.rename("sigma"))
    return pd.concat(parts)


def tercile_diag(oos: pd.DataFrame, best: str, span: int = 20) -> dict:
    sig = sigma_panel(sorted(oos["ticker"].unique()), span)
    mi = pd.MultiIndex.from_arrays([oos.index, oos["ticker"]])
    d = oos[[best, "fwd_ret"]].copy()
    d["sigma"] = sig.reindex(mi).to_numpy()
    d = d.dropna(subset=["sigma"])
    min_names = max(3, oos["ticker"].nunique() // 3 - 1)  # ~9 en 31 tickers
    res = ic_by_tercile(d, best, "fwd_ret", "sigma", min_names=min_names)
    return {f"tercil_{t}": ic_summary(res[t], step=HORIZON, seed=0) for t in res}


def portfolio_step(oos: pd.DataFrame, best: str, n_draws: int = 1000) -> dict:
    """Top-5 con histéresis (M5) + costos de costs.py, contra equiponderado y aleatorio."""
    rng = np.random.default_rng(SEED)
    d = oos.copy()
    d["score"] = d[best] + rng.uniform(0, 1e-9, size=len(d))
    rp = rebalance_panel(d)
    ret = wide(rp, "ret")
    sc = wide(rp, "score").reindex(index=ret.index, columns=ret.columns)
    out = {"n_rebalances": int(len(ret)), "cells": {}}
    for f in FREQS:
        out["cells"][f"freq_{f}d"] = run_cell(sc, ret, K, EXIT_RANK, f)
    draws = random_topk_returns(rp, "ret", K, n_draws=n_draws, seed=SEED)
    sr = sharpe_cols(draws)
    gross = sharpe(rp.assign(r=rp["ret"]).pipe(lambda x: _topk_weekly(x, "score")))
    out["random_topk"] = {"sharpe_gross_p5_p95": [float(np.nanpercentile(sr, 5)), float(np.nanpercentile(sr, 95))],
                          "mean_gross_per_period": float(draws.mean().mean()),
                          "model_top5_reselect_sharpe_gross": gross,
                          "model_percentile": percentile_in(gross, sr),
                          "nota": "aleatorio sin histéresis ni costos (reselección total); solo bruto"}
    out["equal_weight_sharpe_gross"] = sharpe(equal_weight_returns(rp, "ret"))
    return out


def _topk_weekly(rp: pd.DataFrame, col: str) -> pd.Series:
    from src.ml.shared.evaluators.cross_section import topk_returns
    return topk_returns(rp, col, "ret", K)


def main() -> None:
    ap = argparse.ArgumentParser(description="M9 · bake-off de modelos")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    np.random.seed(SEED)
    t_start = time.time()

    tickers = load_universe_tickers("bmv")
    if args.fast:
        tickers = dict(list(tickers.items())[:16])
    panel = build_panel(tickers, horizon=HORIZON, use_open_entry=True).sort_index(kind="stable")
    panel = rank_features(panel, FEATURE_COLS)
    panel["pct_fwd"] = panel.groupby(panel.index)["fwd_ret"].rank(pct=True)
    cols = select_priority_rn_cols([f"rn_{c}" for c in FEATURE_COLS])
    panel = panel.dropna(subset=cols + ["pct_fwd"])
    folds = list(purged_walk_forward_date_splits(panel.index, n_splits=N_SPLITS, horizon=HORIZON, embargo=EMBARGO))
    print(f"Panel {panel.shape} · {panel['ticker'].nunique()} tickers · {len(cols)} features · {len(folds)} folds", flush=True)

    arms = build_arms(cols, args.fast)
    oos, meta = run_folds(panel, arms, folds)
    names = [a.name for a in arms] + [REF]
    table, ics = arm_table(oos, names)

    cand = [a for a in names if a != REF]
    best = max(cand, key=lambda a: table[a]["ic"]["mean"])
    winners = [a for a in cand if a != "ridge_rank" and table[a]["gana"]]
    print("Combinando…", flush=True)
    blends = combos(oos, ics)
    print("Tercil σ̂ y portafolio…", flush=True)
    terc = tercile_diag(oos, best)
    port = portfolio_step(oos, best)

    n_trials = len(arms) + len(blends) + len(FREQS)
    if not args.fast:
        for a in arms:
            log_trial(f"m9_bakeoff:{a.name}", a.config, {"ic_mean": table[a.name]["ic"]["mean"],
                                                        "t": table[a.name]["ic"]["t"]}, table[a.name]["ic"]["n"])
        for b, r in blends.items():
            log_trial(f"m9_bakeoff:blend:{b}", {"scheme": b, "lam": LAM, "arms": list(BLEND_ARMS)},
                      {"ic_mean": r["ic"]["mean"], "t": r["ic"]["t"]}, r["ic"]["n"])
        for f in FREQS:
            log_trial(f"m9_bakeoff:portfolio:{best}", {"best": best, "k": K, "exit": EXIT_RANK, "freq": f},
                      {"turnover": port["cells"][f"freq_{f}d"]["strat"]["turnover_mean"]}, port["n_rebalances"])

    rep = {
        "generated_at": datetime.now().isoformat(timespec="seconds"), "fast": args.fast, "seed": SEED,
        "n_tickers": int(panel["ticker"].nunique()), "n_rows": int(len(panel)), "n_features": len(cols),
        "features": cols, "folds": [[int(len(tr)), int(len(te))] for tr, te in folds],
        "criterion": f"gana si supera al Ridge por ≥ 2 EE (diferencia pareada) en ≥ {WIN_FOLDS} de {N_SPLITS} folds",
        "arms": table, "early_stopping_best_n_by_fold": meta,
        "winners": winners, "verdict": ("empate: se queda el más simple (Ridge)" if not winners
                                        else f"ganan: {winners}"),
        "best_arm_by_pooled_ic (selección sobre test)": best,
        "blends": blends, "sigma_terciles": {"arm": best, **terc}, "portfolio": {"arm": best, **port},
        "trials": n_trials, "monotone_signs": dict(zip(cols, arms[-1].signs)),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    path = ROOT / "data" / "eval" / ("model_bakeoff_fast.json" if args.fast else "model_bakeoff.json")
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print(f"\nEnsayos: {n_trials} · {path} · {rep['elapsed_s']}s")
    print(f"{'brazo':26}{'IC':>8}{'t':>7}{'AUC':>7}{'σfold':>7}{'gana':>6}{'folds':>6}{'Δridge':>8}")
    for a in names:
        r = table[a]
        print(f"{a:26}{r['ic']['mean']:8.4f}{r['ic']['t']:7.2f}{r['auc_mean']:7.4f}{r['fold_ic_std']:7.4f}"
              f"{('Y' if r.get('gana') else '-'):>6}{r.get('folds_won_2se', '')!s:>6}"
              f"{r.get('vs_ridge_pooled', {}).get('diff', float('nan')):8.4f}")
    print("Veredicto:", rep["verdict"], "· mejor (sobre test):", best)
    for b, r in blends.items():
        print(f"blend {b:32} IC {r['ic']['mean']:+.4f} t={r['ic']['t']:.2f}  vs 1/N {r['vs_equal']['diff']:+.4f} (t={r['vs_equal']['t']:.2f})")
    for t, r in terc.items():
        print(f"σ̂ {t}: IC {r['mean']:+.4f} t={r['t']:.2f} n={r['n']}")
    for c, r in port["cells"].items():
        s = r["strat"]
        print(f"top5 {c}: rot {s['turnover_mean']:.2f} bruto {s['gross_mean']*100:.3f}%/sem equilibrio "
              f"{s['breakeven_per_side']*100:.3f}% Sharpe neto@0.30 {s['net']['0.0030']['sharpe']:.2f} "
              f"exceso vs EW t={r['excess_gross_t']:.2f}")


if __name__ == "__main__":
    main()
