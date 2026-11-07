"""
Estudio de regresores de volatilidad: ¿algo le suma al HAR? (M7a)

Añade UN regresor solo-OHLCV a la vez al HAR (más un ensayo con todos juntos),
con los mismos folds purgados, y mide el skill incremental a 5d y 21d.
Objetivo fijo: Parkinson (targets.log_rv_fwd_*).

Criterio (fijado antes de mirar): se conserva solo el que suma ≥ 1 punto de
skill (promedio de folds); "robusto" si además el IC95% por bloques excluye 0.

Uso:
    python scripts/run_vol_regressor_study.py --fast      # 3 folds (reproduce target_study)
    python scripts/run_vol_regressor_study.py             # 6 folds
    python scripts/run_vol_regressor_study.py --regressors overnight semivar

Salida: data/eval/vol_regressor_study.json (o --out)
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.store import BMV_TICKERS  # noqa: E402
from src.data.store import get as store_get  # noqa: E402
from src.ml import vol_regressors as vr  # noqa: E402
from src.ml.targets import HAR_COLS, build_targets, extra_features  # noqa: E402

OUT_PATH = ROOT / "data" / "eval" / "vol_regressor_study.json"
TARGET_STUDY = ROOT / "data" / "eval" / "target_study.json"
TARGETS = {5: "log_rv_fwd_5", 21: "log_rv_fwd_21"}
SEED = 0
TOL_BASELINE = 0.01


def load_raws() -> dict[str, pd.DataFrame]:
    raws = {}
    for name in BMV_TICKERS:
        raw = store_get(name)
        if raw is not None and len(raw) >= 300:
            raws[name] = raw
    return raws


def build_panel(raws: dict[str, pd.DataFrame], reg_names: list[str]) -> pd.DataFrame:
    """Panel fecha × ticker: HAR, objetivos de vol y columnas de cada regresor."""
    bench = store_get("NAFTRAC")["Close"]
    computed = {n: vr.compute_regressor(n, raws) for n in reg_names}
    rows = []
    for tk, raw in raws.items():
        df = extra_features(raw)[HAR_COLS].join(
            build_targets(raw, bench, horizons=(), vol_horizons=(5, 21)))
        for n in reg_names:
            df = df.join(computed[n][tk])
        df["ticker"] = tk
        df["date"] = df.index
        rows.append(df.dropna(subset=["log_rv_5d", "log_rv_22d"]))  # igual que target_study
    return pd.concat(rows, ignore_index=True).sort_values("date", kind="stable")


def baseline_check(panel: pd.DataFrame, n_splits: int) -> dict:
    """HAR en muestra completa vs las cifras de target_study.json (±0.01)."""
    ref = {}
    if TARGET_STUDY.exists():
        d = json.loads(TARGET_STUDY.read_text())
        if d["config"]["splits"] == n_splits:
            ref = {r["target"]: r["summary"]["har"]["skill_vs_naive_mean"]
                   for r in d["results"] if r["task"] == "regression"}
    out = {}
    for h, tgt in TARGETS.items():
        r = vr.evaluate_regressor(panel, tgt, h, [], n_splits=n_splits, n_boot=10, seed=SEED)
        entry = {"skill_har": round(r["skill_base"], 4), "n_obs": r["n_obs"]}
        if tgt in ref:
            entry["skill_target_study"] = ref[tgt]
            entry["ok"] = abs(r["skill_base"] - ref[tgt]) <= TOL_BASELINE
        out[tgt] = entry
    return out


def main() -> None:
    # --extra-module (M7b): importa un módulo que registra regresores en vr.REGRESSORS;
    # se resuelve ANTES de armar `choices`. Ganchos opcionales: prepare(raws), post_study(...).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--extra-module", default=None)
    mod = importlib.import_module(pre.parse_known_args()[0].extra_module) \
        if pre.parse_known_args()[0].extra_module else None
    p = argparse.ArgumentParser(description="Skill incremental de regresores de vol sobre HAR")
    p.add_argument("--extra-module", default=None,
                   help="módulo que registra regresores extra (p. ej. src.ml.vol_calendar)")
    p.add_argument("--regressors", nargs="+", default=list(vr.REGRESSORS),
                   choices=list(vr.REGRESSORS))
    p.add_argument("--splits", type=int, default=6)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--fast", action="store_true", help="3 folds (comparable con target_study)")
    a = p.parse_args()
    if a.fast:
        a.splits = 3

    t0 = time.time()
    print(f"Panel BMV + regresores {a.regressors} ...")
    raws = load_raws()
    if mod is not None and hasattr(mod, "prepare"):
        mod.prepare(raws)
    panel = build_panel(raws, a.regressors)
    print(f"   {len(panel):,} filas · {panel['ticker'].nunique()} tickers · "
          f"{panel['date'].min().date()} → {panel['date'].max().date()}")

    base = baseline_check(panel, a.splits)
    print("\nBaseline HAR (muestra completa) vs target_study.json:")
    for tgt, v in base.items():
        print(f"   {tgt}: {v}")

    trials = [(n, list(vr.REGRESSORS[n].cols)) for n in a.regressors]
    if len(a.regressors) > 1:
        trials.append(("todos", [c for _, cols in trials for c in cols]))

    results = []
    for h, tgt in TARGETS.items():
        print(f"\n── {tgt} (h={h}d)   Δskill = skill(HAR+reg) − skill(HAR), mismos folds")
        print(f"   {'regresor':<16} {'n_obs':>8} {'skill HAR':>9} {'Δskill':>8} "
              f"{'IC95%':>17} {'DM p':>6} {'ΔQLIKE':>8}  veredicto")
        for name, cols in trials:
            r = vr.evaluate_regressor(panel, tgt, h, cols, n_splits=a.splits,
                                      n_boot=a.boot, seed=SEED)
            r["name"] = name
            r["verdict"] = vr.verdict(r["delta_skill_fold_mean"], r["delta_ci_lo"])
            results.append(r)
            print(f"   {name:<16} {r['n_obs']:>8,} {r['skill_base']:>+9.4f} "
                  f"{r['delta_skill_fold_mean']:>+8.4f} "
                  f"[{r['delta_ci_lo']:+.4f},{r['delta_ci_hi']:+.4f}] "
                  f"{r['dm_p']:>6.3f} {r['delta_qlike']:>+8.4f}  {r['verdict']}")

    extra = mod.post_study(panel, trials, a.splits) if mod is not None and hasattr(mod, "post_study") else None
    args_out = a.out
    args_out.parent.mkdir(parents=True, exist_ok=True)
    args_out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"splits": a.splits, "seed": SEED, "boot": a.boot,
                   "criterio": f"conserva si Δskill ≥ {vr.MIN_DELTA_SKILL} (promedio de folds); "
                               "robusto si IC95% > 0",
                   "ensayos": len(trials), "comparaciones": len(results),
                   "objetivo": "Parkinson (targets.log_rv_fwd_*)"},
        "baseline_check": base,
        "results": results,
        **({"extra": extra} if extra else {}),
    }, indent=2))
    print(f"\nEnsayos: {len(trials)} regresores × 2 horizontes = {len(results)} comparaciones")
    print(f"Guardado: {args_out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
