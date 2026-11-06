"""
M6 · Estudio de vol targeting.

¿Escalar la exposición por 1/σ̂ baja el drawdown sin perder Sharpe?

Parte 1 (banco de pruebas): equiponderado diario de los .MX, 2010-2026.
  σ̂ = EWMA 10d, EWMA 21d (span, sobre r²) y HAR fuera de muestra (folds purgados).
  Objetivo 8/10/12% anual, solo largos, tope 1.0 (sin apalancamiento), rezago 1 día,
  costo 0.05% por unidad de peso que cambia.
Parte 2: top-K semanal (data/eval/backtest_topk.json): solo MDD y vol.

Uso:
    python scripts/run_vol_target_study.py
Salida: data/eval/vol_target_study.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.store import BMV_TICKERS, get  # noqa: E402
from src.ml.shared.evaluators.metrics import Metrics  # noqa: E402
from src.ml.vol_target import (  # noqa: E402
    ewma_vol, har_oos_vol, target_exposure, vol_targeted_returns,
)

START = "2010-01-01"
TARGETS = (0.08, 0.10, 0.12)
COST = 0.0005
MIN_NAMES = 10
PPY_DAILY = 252.0
PPY_WEEK = 50.4
OUT = ROOT / "data" / "eval" / "vol_target_study.json"
TOPK = ROOT / "data" / "eval" / "backtest_topk.json"


def equal_weight_daily() -> pd.Series:
    """Retorno diario equiponderado (rebalanceo diario) de los .MX con datos ese día."""
    closes = {}
    for name, sym in BMV_TICKERS.items():
        if not sym.endswith(".MX"):
            continue
        df = get(name)
        if df is not None and "Close" in df.columns:
            closes[name] = df["Close"]
    px = pd.DataFrame(closes).sort_index()
    ret = px.pct_change(fill_method=None)
    ret = ret.loc[START:]
    ew = ret.mean(axis=1).where(ret.notna().sum(axis=1) >= MIN_NAMES)
    return ew.dropna()


def stats(r: pd.Series, ppy: float) -> dict:
    """Métricas de una serie de retornos periódicos."""
    r = r.dropna()
    wealth = np.cumprod(1 + r.to_numpy())
    mdd = Metrics.max_drawdown(wealth)
    return {
        "n": int(len(r)),
        "ret_anual": float(r.mean() * ppy),
        "vol_anual": float(r.std(ddof=0) * np.sqrt(ppy)),
        "sharpe": float(Metrics.sharpe_ratio(r.to_numpy(), periods_per_year=ppy)),
        "mdd": float(mdd),
        "calmar": float(Metrics.calmar_ratio(r.to_numpy(), wealth, periods_per_year=ppy)),
    }


def run_variant(r: pd.Series, sigma: pd.Series, target: float, cost: float) -> tuple[pd.Series, dict]:
    out = vol_targeted_returns(r, sigma, target, lag=1, cost_per_unit_change=cost)
    w = target_exposure(sigma.reindex(r.index).shift(1), target).reindex(out.index)
    s = stats(out, PPY_DAILY)
    s["expo_media"] = float(w.mean())
    s["rotacion_anual"] = float(w.diff().abs().mean() * PPY_DAILY)
    s["vol_vs_objetivo"] = s["vol_anual"] - target
    return out, s


def decision_dates(ew_index: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    """Último día hábil de cada semana (como el backtest top-K) dentro del test."""
    idx = ew_index[(ew_index >= start) & (ew_index <= end)]
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(idx.to_period("W")).max().to_numpy())


def topk_part(sigmas: dict[str, pd.Series], ew: pd.Series) -> dict:
    d = json.loads(TOPK.read_text())
    net = np.asarray(d["period_returns"]["net"], dtype=float)
    m = d["model"]
    dates = decision_dates(ew.index, m["test_start"], m["test_end"])
    info = {"periodos_archivo": len(net), "fechas_reconstruidas": len(dates)}
    if len(dates) < len(net):
        info["aviso"] = "no se pudieron alinear fechas; sin parte top-K"
        return info
    # Semanas descartadas por MIN_NAMES no se conocen: se alinean las últimas len(net).
    dates = dates[-len(net):]
    r = pd.Series(net, index=dates)
    info["base"] = {k: v for k, v in stats(r, PPY_WEEK).items() if k in ("n", "mdd", "vol_anual")}
    info["variantes"] = {}
    for sname, sig in sigmas.items():
        for tgt in TARGETS:
            w = target_exposure(sig.reindex(dates), tgt)  # σ̂ al cierre de la decisión
            ok = w.notna()
            if ok.sum() < 20:
                continue
            rr = (w * r)[ok]
            rr = rr - COST * w[ok].diff().abs().fillna(w[ok].iloc[0])
            s = stats(rr, PPY_WEEK)
            # Línea base en las mismas semanas
            b = stats(r[ok], PPY_WEEK)
            info["variantes"][f"{sname}_{int(tgt * 100)}"] = {
                "n": s["n"], "mdd": s["mdd"], "vol_anual": s["vol_anual"],
                "base_mdd_mismas_semanas": b["mdd"], "base_vol_mismas_semanas": b["vol_anual"],
                "expo_media": float(w[ok].mean()),
            }
    return info


def main() -> None:
    ew = equal_weight_daily()
    print(f"EW diario: {ew.index[0].date()} → {ew.index[-1].date()}  n={len(ew)}")
    sigmas = {
        "ewma10": ewma_vol(ew, 10),
        "ewma21": ewma_vol(ew, 21),
        "har": har_oos_vol(ew),
    }
    har_start = sigmas["har"].first_valid_index()
    print(f"HAR fuera de muestra desde {har_start.date()}")

    res: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "start": START, "targets": list(TARGETS), "cap": 1.0, "lag": 1,
            "cost_per_unit_change": COST, "ewma": "span, sobre r², media cero, anualizada x252",
            "har": "log_rv 1d/5d/22d (r²) → log vol 5d; 8 folds purgados (horizon 5, embargo 10, min_train 0.4)",
            "universo": "EW diario de los .MX del store; rebalanceo diario sin costo del EW",
            "ensayos": 3 * 3 + 1,
        },
        "ew_universo_n_nombres_max": int(sum(1 for s in BMV_TICKERS.values() if s.endswith(".MX"))),
    }
    windows = {"completo": ew.index[0], "ventana_har": har_start}
    res["ventanas"] = {}
    for wname, w0 in windows.items():
        sub = ew.loc[w0:]
        block = {"sin_escalar": stats(sub, PPY_DAILY), "variantes": {}}
        for sname, sig in sigmas.items():
            if wname == "completo" and sname == "har":
                continue  # HAR solo existe en su ventana
            for tgt in TARGETS:
                _, s = run_variant(sub, sig, tgt, COST)
                block["variantes"][f"{sname}_{int(tgt * 100)}"] = s
        res["ventanas"][wname] = block
    # Control del agente 05: EWMA10 objetivo 10%, con y sin costo (ventana completa).
    ctrl = {}
    for c in (0.0, COST):
        _, s = run_variant(ew, sigmas["ewma10"], 0.10, c)
        ctrl[f"costo_{c}"] = s
    res["control_agente05"] = ctrl
    res["topk"] = topk_part(sigmas, ew)

    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    # Resumen en pantalla
    for wname, block in res["ventanas"].items():
        b = block["sin_escalar"]
        print(f"\n[{wname}] sin escalar: Sharpe {b['sharpe']:.2f} vol {b['vol_anual']:.1%} "
              f"MDD {b['mdd']:.1%} Calmar {b['calmar']:.2f}")
        for k, s in block["variantes"].items():
            print(f"  {k:<10} Sharpe {s['sharpe']:.2f} vol {s['vol_anual']:.1%} MDD {s['mdd']:.1%} "
                  f"Calmar {s['calmar']:.2f} expo {s['expo_media']:.2f} rot {s['rotacion_anual']:.1f}")
    print("\nTop-K:", json.dumps(res["topk"].get("base"), ensure_ascii=False))
    for k, v in res["topk"].get("variantes", {}).items():
        print(f"  {k:<10} MDD {v['mdd']:.1%} (base {v['base_mdd_mismas_semanas']:.1%}) "
              f"vol {v['vol_anual']:.1%} (base {v['base_vol_mismas_semanas']:.1%}) expo {v['expo_media']:.2f}")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
