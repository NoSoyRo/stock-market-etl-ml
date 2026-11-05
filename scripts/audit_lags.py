"""M2b · Auditoría de rezagos y fugas en features macro y fundamentales.

Las features macro valen igual para todos los tickers en una fecha, así que el IC
"por fecha entre acciones" no está definido (varianza cero). Se mide como serie de
tiempo: Spearman entre la feature (desplazada L días) y el retorno futuro 5d de
NAFTRAC (mismo `compute_target` que el panel). `rel_str5` sí varía por ticker y se
audita con `ic_by_date`.

CRITERIO (escrito ANTES de mirar resultados):
  sospechosa si IC(lag0) − IC(lag+1) > 2·SE, con SE = 1/√n_efectivo y
  n_efectivo = n_filas / 5 (el retorno de 5d se traslapa). Sin razón económica
  declarada (p. ej. momentum de 5d que persiste) ⇒ sospechosa; si no, ok.
  lag+L = usar el valor de la feature de t−L (más viejo).

Uso:  .venv/bin/python scripts/audit_lags.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HORIZON = 5
LAGS = (0, 1, 2)
MACRO_FEATS = ["sp500_ret5", "usdmxn_ret5", "usdmxn_vs_ma20", "wti_ret5", "mkt_ret10", "mkt_trend"]
OUT = ROOT / "data" / "eval" / "lag_audit.json"


def ts_ic(feature: pd.Series, fwd: pd.Series, lag: int) -> dict:
    """Spearman serie-de-tiempo entre feature(t−lag) y fwd(t); SE con n/horizonte."""
    df = pd.DataFrame({"x": feature.shift(lag), "y": fwd}).dropna()
    n = len(df)
    if n < 30:
        return {"ic": float("nan"), "se": float("nan"), "t": float("nan"), "n": n}
    ic = float(df["x"].rank().corr(df["y"].rank()))
    se = 1.0 / np.sqrt(max(n / HORIZON, 2.0))
    return {"ic": ic, "se": float(se), "t": ic / se, "n": n}


def verdict(ic0: float, ic1: float, se: float) -> str:
    """'sospechosa' si IC(lag0) − IC(lag+1) > 2·SE; si no, 'ok'."""
    if not np.isfinite(ic0 - ic1) or not np.isfinite(se):
        return "sin_datos"
    return "sospechosa" if (ic0 - ic1) > 2 * se else "ok"


def lag_table(feats: pd.DataFrame, fwd: pd.Series, cols: list[str]) -> dict:
    """Tabla por feature: IC/t por lag y veredicto."""
    out = {}
    for c in cols:
        row = {f"lag{L}": ts_ic(feats[c], fwd, L) for L in LAGS}
        row["veredicto"] = verdict(row["lag0"]["ic"], row["lag1"]["ic"], row["lag0"]["se"])
        out[c] = row
    return out


def lead_lag_corr(market_ret: pd.Series, other_ret: pd.Series) -> dict:
    """corr(mercado_t, otro_{t+k}) para k=-1,0,+1 (k=-1: el otro de ayer)."""
    r = {}
    for k, name in ((-1, "t-1"), (0, "t"), (1, "t+1")):
        d = pd.DataFrame({"m": market_ret, "o": other_ret.shift(-k)}).dropna()
        r[name] = float(d["m"].corr(d["o"])) if len(d) > 30 else float("nan")
    return r


def fund_audit(fund: pd.DataFrame) -> dict:
    """Resumen de available_date − quarter_end y prueba del merge_asof 'backward'."""
    gap = (pd.to_datetime(fund["available_date"]) - pd.to_datetime(fund["quarter_end"])).dt.days
    return {
        "filas": int(len(fund)),
        "tickers": int(fund["ticker"].nunique()),
        "gap_dias_min": int(gap.min()),
        "gap_dias_max": int(gap.max()),
        "gap_constante": bool(gap.nunique() == 1),
    }


def merge_asof_same_day(available: pd.Timestamp) -> dict:
    """Demuestra que merge_asof(backward) ya usa el dato el MISMO día de available_date."""
    from src.ml.features import add_fundamental_features

    idx = pd.bdate_range(available - pd.Timedelta(days=3), periods=6)
    base = pd.DataFrame({"Close": 1.0}, index=idx)
    fund = pd.DataFrame({"available_date": [available], "fund_gross_margin": [0.5]})
    res = add_fundamental_features(base, fund, ticker=None, symbol=None)
    first = res["fund_gross_margin"].dropna().index.min()
    return {"available_date": str(available.date()), "primer_dia_con_dato": str(first.date()),
            "mismo_dia": bool(first == available)}


def main() -> None:
    from src.data.store import MACRO_TICKERS, UNIVERSE_GROUPS, get as store_get
    from src.ml.features import add_macro_features, compute_target
    from src.ml.shared.evaluators.cross_section import ic_by_date, ic_summary

    macro = {n: store_get(n) for n in MACRO_TICKERS}
    naf = macro["NAFTRAC"]
    feats = add_macro_features(naf[["Close"]].copy(), macro)
    fwd = compute_target(naf, horizon=HORIZON, use_open_entry=True)
    fwd.index = pd.to_datetime(fwd.index)
    n_ensayos = 0

    tabla = lag_table(feats, fwd, MACRO_FEATS)
    n_ensayos += len(MACRO_FEATS) * len(LAGS)

    # Alineación: retornos diarios sobre el calendario de NAFTRAC (como los usa la feature).
    nr = naf["Close"].pct_change()
    nr.index = pd.to_datetime(nr.index)
    alin = {}
    for nombre in ("SP500", "USDMXN", "WTI"):
        s = macro[nombre]["Close"]
        s.index = pd.to_datetime(s.index)
        r = s.reindex(nr.index, method="ffill").pct_change()
        alin[nombre] = lead_lag_corr(nr, r)
        n_ensayos += 3

    # rel_str5: específica por ticker -> IC por fecha entre acciones.
    frames = []
    for name in UNIVERSE_GROUPS["bmv"]:
        raw = store_get(name)
        if raw is None or len(raw) < 200:
            continue
        d = add_macro_features(raw[["Close"]].copy(), macro)[["rel_str5"]]
        d["fwd"] = compute_target(raw, horizon=HORIZON, use_open_entry=True).values
        d["ticker"] = name
        for L in LAGS:
            d[f"rel{L}"] = d["rel_str5"].shift(L)
        frames.append(d)
    panel = pd.concat(frames).sort_index()
    rel = {}
    for L in LAGS:
        ic = ic_by_date(panel, f"rel{L}", "fwd", min_names=10)
        s = ic_summary(ic, step=HORIZON)
        rel[f"lag{L}"] = {"ic": float(s["mean"]), "se": float(s["se"]), "t": float(s["t"]), "n": int(s["n"])}
        n_ensayos += 1
    rel["veredicto"] = verdict(rel["lag0"]["ic"], rel["lag1"]["ic"], rel["lag0"]["se"])
    tabla["rel_str5 (corte transversal)"] = rel

    fund = pd.read_parquet(ROOT / "data" / "fundamentals.parquet")
    fa = fund_audit(fund)
    fa["merge_asof"] = merge_asof_same_day(pd.Timestamp("2024-03-12"))

    res = {"criterio": "sospechosa si IC(lag0) - IC(lag+1) > 2*SE, SE=1/sqrt(n/5)",
           "rango_fechas": [str(naf.index.min().date()), str(naf.index.max().date())],
           "ic_macro": tabla, "alineacion_corr_naftrac": alin, "fundamentales": fa,
           "ensayos": n_ensayos}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for k, v in tabla.items():
        print(f"{k:32s} ic0={v['lag0']['ic']:+.3f} ic1={v['lag1']['ic']:+.3f} "
              f"ic2={v['lag2']['ic']:+.3f} t0={v['lag0']['t']:+.1f} -> {v['veredicto']}")
    print("alineacion:", json.dumps(alin))
    print("fundamentales:", json.dumps(fa))
    print("ensayos:", n_ensayos)


if __name__ == "__main__":
    main()
