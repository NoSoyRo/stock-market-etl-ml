"""
M10 · Cambiar la pregunta: 5 variantes de objetivo sobre el Ridge limpio.

Uso:
    .venv/bin/python scripts/run_target_redesign.py --fast   # 16 tickers (humo)
    .venv/bin/python scripts/run_target_redesign.py          # completo (Ridge: ~1–2 min)

Diseño FIJADO ANTES de ver resultados:
- Mismo modelo (``RidgeArm``, alpha=100, sin ajustar), mismas 24 features ``rn_*``, panel limpio
  (``use_mask=True``), 5 folds purgados (horizonte 5, embargo 5), entrada Open t+1, como M9.
- Todas las variantes ven exactamente las mismas filas: se descartan del panel las filas sin σ̂
  válida (primeros 20 días de cada ticker o σ̂ ≤ 0).
- Métrica PRINCIPAL: IC de Spearman por fecha del score contra ``fwd_ret`` crudo (``ic_summary``,
  step = 5, bootstrap por bloques). Secundaria (informativa): IC contra ``fwd_ret/σ̂``.
- "Gana" una variante si supera a A (diferencia pareada por fecha del IC crudo) por ≥ 2 errores
  estándar de esa diferencia en la MAYORÍA de folds (≥ 3 de 5). Si no, empate y se queda A.
- 5 ensayos (A–E), sin tocar hiperparámetros; se registran en ``data/eval/trial_log.jsonl`` (no --fast).
- D entrena solo con |fwd_ret| ≥ 0.5·σ̂·√5; el test usa TODAS las filas.
- E quita las 3 features de mercado de las 24 (``MARKET_COLS``, lista fija; el detector por datos
  solo se reporta como diagnóstico: con 31 tickers los feriados lo vuelven inestable).
- Portafolio: selección M5 (rebalanceo 10d, K=10, salida rank 2K=20), costo 0.30 % por lado.
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.backtest_topk import HORIZON, build_panel, load_universe_tickers  # noqa: E402
from scripts.run_m1_report import rebalance_panel  # noqa: E402
from scripts.run_model_bakeoff import paired_vs  # noqa: E402
from scripts.run_rotation_study import run_cell, wide  # noqa: E402
from src.data.store import get as store_get  # noqa: E402
from src.ml.cs_targets import (  # noqa: E402
    MARKET_COLS, VARIANTS, build_variant, constant_by_date_cols, daily_sigma_pct, sigma_for_panel,
)
from src.ml.features import FEATURE_COLS  # noqa: E402
from src.ml.kb_enhance import select_priority_rn_cols  # noqa: E402
from src.ml.purged_cv import purged_walk_forward_date_splits  # noqa: E402
from src.ml.rank_models import EMBARGO, RidgeArm, rank_features  # noqa: E402
from src.ml.shared.evaluators.cross_section import ic_by_date, ic_summary  # noqa: E402
from src.ml.trials import log_trial  # noqa: E402

N_SPLITS = 5
WIN_FOLDS = 3
SEED = 42
K, EXIT_RANK, FREQ = 10, 20, 10
COST = "0.0030"
DESC = {
    "A": "baseline: percentil de fwd_ret",
    "B": "percentil de fwd_ret/σ̂",
    "C": "percentil de fwd_ret − media de la fecha",
    "D": "como A, entrena solo con |fwd_ret| ≥ 0.5σ̂√5",
    "E": "como A, sin 3 features de mercado",
}


def run_variants(panel: pd.DataFrame, sigma: np.ndarray, cols: list[str], folds: list) -> tuple:
    """Predicciones OOS de cada variante (una columna por variante) + metadatos."""
    oos_parts, meta = [], {v: {"n_cols": [], "coverage": []} for v in VARIANTS}
    built = {v: build_variant(panel, v, sigma, cols) for v in VARIANTS}
    for fi, (tr, te) in enumerate(folds):
        test = panel.iloc[te]
        part = test[["ticker", "fwd_ret", "y_rel"]].copy()
        part["z"] = test["fwd_ret"].to_numpy() / sigma[te]
        part["fold"] = fi
        for v in VARIANTS:
            data, use = built[v]
            train = data.iloc[tr]
            ok = train["train_ok"].to_numpy(dtype=bool)
            meta[v]["coverage"].append(float(ok.mean()))
            meta[v]["n_cols"].append(len(use))
            arm = RidgeArm(use).fit(train[ok])
            part[v] = arm.predict(data.iloc[te])          # test completo: no se recorta
        oos_parts.append(part)
        print(f"  fold {fi}: " + " ".join(f"{v} cov={meta[v]['coverage'][-1]:.2f}" for v in VARIANTS), flush=True)
    return pd.concat(oos_parts).sort_index(), meta, built


def _wins(c: dict) -> bool:
    """≥ 2 EE con diferencia y EE estrictamente positivos (0 ≥ 0 no cuenta como victoria)."""
    return bool(c["wins_2se"] and c["se"] > 0 and c["diff"] > 0)


def evaluate(oos: pd.DataFrame) -> dict:
    """IC (crudo y en σ̂) por variante, y comparación pareada contra A por fold."""
    fold_of = oos.groupby(level=0)["fold"].first()
    ic_raw = {v: ic_by_date(oos, v, "fwd_ret") for v in VARIANTS}
    ic_z = {v: ic_by_date(oos, v, "z") for v in VARIANTS}
    out = {}
    for v in VARIANTS:
        row = {"ic_raw": ic_summary(ic_raw[v], step=HORIZON, seed=0),
               "ic_z": ic_summary(ic_z[v], step=HORIZON, seed=0), "folds": {}}
        for f in sorted(oos["fold"].unique()):
            idx = fold_of.index[fold_of == f]
            fr = ic_summary(ic_raw[v].reindex(idx).dropna(), step=HORIZON, seed=0)
            fz = {"ic_mean_z": ic_summary(ic_z[v].reindex(idx).dropna(), step=HORIZON, seed=0)["mean"]}
            cell = {"ic_mean": fr["mean"], "t": fr["t"], "n": fr["n"], **fz}
            if v != "A":
                cell["vs_A"] = paired_vs(ic_raw[v].reindex(idx).dropna(), ic_raw["A"].reindex(idx))
                cell["vs_A_z"] = paired_vs(ic_z[v].reindex(idx).dropna(), ic_z["A"].reindex(idx))
            row["folds"][str(f)] = cell
        if v != "A":
            row["vs_A_pooled"] = paired_vs(ic_raw[v], ic_raw["A"])
            row["vs_A_pooled_z"] = paired_vs(ic_z[v], ic_z["A"])
            row["folds_won"] = int(sum(_wins(c["vs_A"]) for c in row["folds"].values()))
            row["folds_won_z"] = int(sum(_wins(c["vs_A_z"]) for c in row["folds"].values()))
            row["gana"] = bool(row["folds_won"] >= WIN_FOLDS)
            row["identica_a_A"] = bool(np.allclose(oos[v], oos["A"], atol=1e-12))
        out[v] = row
    return out


def portfolio(oos: pd.DataFrame, v: str) -> dict:
    """Celda M5 (10d, K=10, salida 2K) con desempate por jitter fijo."""
    rng = np.random.default_rng(SEED)
    d = oos.copy()
    d["score"] = d[v] + rng.uniform(0, 1e-9, size=len(d))
    rp = rebalance_panel(d)
    ret = wide(rp, "ret")
    sc = wide(rp, "score").reindex(index=ret.index, columns=ret.columns)
    c = run_cell(sc, ret, K, EXIT_RANK, FREQ)
    s = c["strat"]
    return {"turnover_mean": s["turnover_mean"], "gross_mean_pct": s["gross_mean"] * 100,
            "breakeven_pct_per_side": s["breakeven_per_side"] * 100,
            "sharpe_gross": s["sharpe_gross"], "sharpe_net_030": s["net"][COST]["sharpe"],
            "excess_gross_t": c["excess_gross_t"], "n_rebalances": int(len(ret))}


def to_markdown(rep: dict) -> str:
    L = ["# M10 · Rediseño del objetivo (Ridge limpio)", "",
         f"Generado: {rep['generated_at']} · {rep['n_tickers']} tickers · {rep['n_rows']} filas · "
         f"{rep['n_features']} features · 5 folds purgados · semilla {rep['seed']}", "",
         f"Criterio (fijado antes): {rep['criterion']}", "",
         f"Features de mercado quitadas en E: {rep['market_cols_dropped']} · constantes en ≥90 % de fechas (diagnóstico): {rep['constant_cols'] or 'ninguna'}", "",
         "## IC por variante (step=5; IC95% por bootstrap de bloques)", "",
         "| var | descripción | IC crudo | t | IC95% | IC en σ̂ | Δ vs A (EE) | folds ganados | gana |",
         "|---|---|---|---|---|---|---|---|---|"]
    for v, r in rep["variants"].items():
        i, z = r["ic_raw"], r["ic_z"]
        d = r.get("vs_A_pooled")
        L.append(f"| {v} | {DESC[v]} | {i['mean']:+.4f} | {i['t']:.2f} | [{i['ci_lo']:+.3f}, {i['ci_hi']:+.3f}] | "
                 f"{z['mean']:+.4f} | " + (f"{d['diff']:+.4f} ({d['se']:.4f})" if d else "—") + " | "
                 f"{r.get('folds_won', '—')}/5 | {('idéntica a A' if r['identica_a_A'] else ('SÍ' if r['gana'] else 'no')) if v != 'A' else '—'} |")
    L += ["", "## IC crudo por fold", "", "| var | " + " | ".join(f"f{f}" for f in range(5)) + " |",
          "|---|" + "---|" * 5]
    for v, r in rep["variants"].items():
        L.append(f"| {v} | " + " | ".join(f"{r['folds'][str(f)]['ic_mean']:+.3f}" for f in range(5)) + " |")
    L += ["", f"## Portafolio (M5: {FREQ}d, K={K}, salida {EXIT_RANK}; costo 0.30 %/lado)", "",
          "| var | cobertura train | rotación | bruto %/rebal | equilibrio %/lado | Sharpe bruto | Sharpe neto | exceso t |",
          "|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        p, cov = rep["portfolio"][v], np.mean(rep["coverage_train"][v])
        L.append(f"| {v} | {cov:.0%} | {p['turnover_mean']:.2f} | {p['gross_mean_pct']:.3f} | "
                 f"{p['breakeven_pct_per_side']:.3f} | {p['sharpe_gross']:.2f} | {p['sharpe_net_030']:.2f} | "
                 f"{p['excess_gross_t']:.2f} |")
    L += ["", f"**Veredicto:** {rep['verdict']}", f"Ensayos: {rep['trials']}", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="M10 · rediseño del objetivo")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    np.random.seed(SEED)
    t0 = time.time()

    tickers = load_universe_tickers("bmv")
    if args.fast:
        tickers = dict(list(tickers.items())[:16])
    panel = build_panel(tickers, horizon=HORIZON, use_open_entry=True, use_mask=True).sort_index(kind="stable")
    panel = rank_features(panel, FEATURE_COLS)
    cols = select_priority_rn_cols([f"rn_{c}" for c in FEATURE_COLS])
    sig_by_t = {t: daily_sigma_pct(store_get(t)["Close"]) for t in panel["ticker"].unique()}
    panel["sigma"] = sigma_for_panel(panel, sig_by_t)
    n0 = len(panel)
    panel = panel.dropna(subset=cols + ["sigma"])
    panel = panel[panel["sigma"] > 0]
    sigma = panel["sigma"].to_numpy()
    const = constant_by_date_cols(panel, cols)
    folds = list(purged_walk_forward_date_splits(panel.index, n_splits=N_SPLITS, horizon=HORIZON, embargo=EMBARGO))
    print(f"Panel {panel.shape} (descartadas {n0 - len(panel)} sin σ̂/features) · "
          f"{panel['ticker'].nunique()} tickers · {len(cols)} features · constantes: {const}", flush=True)

    oos, meta, _ = run_variants(panel, sigma, cols, folds)
    table = evaluate(oos)
    port = {v: portfolio(oos, v) for v in VARIANTS}
    winners = [v for v in VARIANTS if v != "A" and table[v]["gana"]]

    if not args.fast:
        for v in VARIANTS:
            log_trial(f"m10_target:{v}", {"variant": v, "desc": DESC[v], "model": "ridge", "alpha": 100.0,
                                          "n_cols": meta[v]["n_cols"][0], "clean": True},
                      {"ic_mean": table[v]["ic_raw"]["mean"], "t": table[v]["ic_raw"]["t"]},
                      table[v]["ic_raw"]["n"])

    rep = {
        "generated_at": datetime.now().isoformat(timespec="seconds"), "fast": args.fast, "seed": SEED,
        "n_tickers": int(panel["ticker"].nunique()), "n_rows": int(len(panel)), "n_features": len(cols),
        "rows_dropped_no_sigma": int(n0 - len(panel)), "constant_cols": const, "market_cols_dropped": [c for c in MARKET_COLS if c in cols], "features": cols,
        "folds": [[int(len(tr)), int(len(te))] for tr, te in folds],
        "criterion": f"gana si supera a A (IC crudo, diferencia pareada) por ≥ 2 EE en ≥ {WIN_FOLDS} de {N_SPLITS} folds",
        "variants": table, "coverage_train": {v: meta[v]["coverage"] for v in VARIANTS},
        "n_cols_by_variant": {v: meta[v]["n_cols"][0] for v in VARIANTS}, "portfolio": port,
        "winners": winners,
        "verdict": "empate: se queda A (baseline)" if not winners else f"ganan sobre A: {winners}",
        "trials": len(VARIANTS), "elapsed_s": round(time.time() - t0, 1),
    }
    out = ROOT / "data" / "eval"
    stem = "target_redesign_fast" if args.fast else "target_redesign"
    (out / f"{stem}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (out / f"{stem}.md").write_text(to_markdown(rep), encoding="utf-8")
    print(to_markdown(rep))


if __name__ == "__main__":
    main()
