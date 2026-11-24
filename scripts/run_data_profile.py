"""N1 · Perfil honesto de los datos: N efectivo, estabilidad de features, calidad de series y EPS limpio.

Uso:
    .venv/bin/python scripts/run_data_profile.py          # completo
    .venv/bin/python scripts/run_data_profile.py --fast   # 12 tickers

Salidas: data/eval/data_profile.json y data/eps/eps_clean.parquet. No hay hipótesis de retorno
ni modelos: es infraestructura de medición.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_topk import build_panel, load_universe_tickers, model_feature_cols  # noqa: E402
from src.data.eps_clean import build_eps_clean  # noqa: E402
from src.data.quality import apply_mask, load_quality_mask  # noqa: E402
from src.data.series_quality import low_quality_years, series_quality_by_year  # noqa: E402
from src.data.store import get as store_get  # noqa: E402
from src.ml.shared.evaluators.effective_n import block_stability, rolling_effective_n  # noqa: E402
from src.ml.shared.evaluators.effective_n import participation_ratio, pc1_share  # noqa: E402

OUT = ROOT / "data" / "eval" / "data_profile.json"
UMBRAL_REPETIDOS, UMBRAL_VOL0 = 20.0, 10.0
FEATURES_CLAVE = ["ret_5d", "mom_12_1", "lowvol20"]
HORIZON = 5


def _t(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def retornos_anchos(names: dict[str, str], mask: pd.DataFrame) -> pd.DataFrame:
    """Retornos diarios close-to-close (solo filas válidas según la máscara), una columna por ticker."""
    cols = {}
    for name, tk in names.items():
        raw = store_get(name)
        if raw is None or len(raw) < 200:
            continue
        raw = apply_mask(raw, mask, tk)
        cols[name] = raw["Close"].astype(float).pct_change()
    return pd.DataFrame(cols).clip(-0.5, 0.5)


def features_extra(panel: pd.DataFrame, names: dict[str, str], mask: pd.DataFrame) -> pd.DataFrame:
    """Agrega `mom_12_1` (ret 252d saltando el último mes) y `lowvol20` (−desv. 20d) al panel, solo con pasado."""
    partes = []
    for name, tk in names.items():
        if name not in set(panel["ticker"]):
            continue
        raw = store_get(name)
        c = apply_mask(raw, mask, tk)["Close"].astype(float)
        r = c.pct_change()
        f = pd.DataFrame({"mom_12_1": c.shift(21) / c.shift(252) - 1, "lowvol20": -r.rolling(20).std()})
        f["ticker"] = name
        partes.append(f)
    ext = pd.concat(partes)
    ext.index = pd.DatetimeIndex(ext.index)
    out = panel.reset_index(names="_d").merge(ext.reset_index(names="_d"), on=["_d", "ticker"], how="left")
    return out.set_index("_d").rename_axis(panel.index.name)


def estabilidad(panel: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """IC por fecha por bloque de 3 años; resumen por feature: signo estable = mismo signo en ≥ 75% de bloques."""
    feats = [f for f in FEATURES_CLAVE if f in panel.columns]
    feats += [f for f in model_feature_cols(panel.columns) if f not in feats]
    tabla = block_stability(panel, feats, "fwd_ret", block_years=3, step=HORIZON)
    resumen = []
    for f, g in tabla.groupby("feature"):
        signo = np.sign(g["ic_mean"])
        dom = signo.value_counts().idxmax()
        frac = float((signo == dom).mean())
        resumen.append({"feature": f, "n_blocks": int(len(g)), "ic_mean_avg": float(g["ic_mean"].mean()),
                        "frac_same_sign": frac, "n_blocks_|t|>2": int((g["t"].abs() > 2).sum()),
                        "stable_sign": bool(frac >= 0.75 and len(g) >= 4)})
    return resumen, tabla


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    names = load_universe_tickers("bmv")
    if args.fast:
        names = dict(list(names.items())[:12])
    mask = load_quality_mask()
    perfil: dict = {"universe_keys": len(names), "fast": args.fast}

    # 1) EPS limpio
    _, conteo = build_eps_clean()
    perfil["eps_clean"] = conteo
    _t(f"EPS limpio: {conteo}", t0)

    # 2) Panel y N efectivo
    panel = build_panel(names, horizon=HORIZON, use_open_entry=True, use_mask=True)
    tickers_panel = sorted(panel["ticker"].unique())
    perfil["panel"] = {"rows": int(len(panel)), "n_tickers": len(tickers_panel),
                       "missing_from_universe": sorted(set(names) - set(tickers_panel)),
                       "naftrac_in_panel": [t for t in tickers_panel if "NAFTRAC" in t.upper()],
                       "tickers": tickers_panel}
    _t(f"panel {panel.shape}", t0)

    rets = retornos_anchos({n: tk for n, tk in names.items() if n in tickers_panel}, mask)
    bloques = rolling_effective_n(rets, "3Y")
    perfil["effective_n"] = {
        "full_sample": {"n_eff": participation_ratio(rets), "pc1_share": pc1_share(rets), "n_assets": int(rets.shape[1])},
        "by_3y_block": [{"start": str(i.date()), **{k: (float(v) if k in ("n_eff", "pc1_share") else int(v))
                                                    for k, v in r.items()}} for i, r in bloques.iterrows()],
    }
    _t(f"N efectivo muestra completa: {perfil['effective_n']['full_sample']}", t0)

    # 3) Estabilidad por bloques
    resumen, tabla = estabilidad(features_extra(panel, names, mask))
    perfil["stability"] = {"summary": resumen, "by_block": tabla.round(5).to_dict("records")}
    _t("estabilidad lista", t0)

    # 4) Calidad por régimen y comparación con la máscara
    por_ticker, clave = {}, {}
    for name, tk in names.items():
        raw = store_get(name)
        if raw is None or raw.empty:
            continue
        por_ticker[tk] = series_quality_by_year(raw)
        clave[tk] = name
    n_filas = {(t, int(y)): int(n) for t, q in por_ticker.items() for y, n in zip(q['year'], q['n_rows'])}
    perfil['mask_missing_tickers'] = sorted(set(names.values()) - set(mask['ticker'].unique()))
    malos = low_quality_years(por_ticker, max_repeated=UMBRAL_REPETIDOS, max_zero_vol=UMBRAL_VOL0)
    m = mask.copy()
    m["date"] = pd.to_datetime(m["date"])
    for r in malos:
        sub = m[(m["ticker"] == r["ticker"]) & (m["date"].dt.year == r["year"])]
        r["pct_rows_excluded_by_mask"] = round(float((~sub["valid"]).mean() * 100), 1) if len(sub) else None
        r["pct_rows_in_panel"] = round(float(((panel["ticker"] == clave[r["ticker"]]) & (panel.index.year == r["year"])).sum()
                                             / max(n_filas[(r["ticker"], r["year"])], 1) * 100), 1)
    perfil["quality"] = {"thresholds": {"repeated_close_pct": UMBRAL_REPETIDOS, "zero_volume_pct": UMBRAL_VOL0},
                         "low_quality_ticker_years": malos,
                         "amxb": por_ticker.get("AMXB.MX", pd.DataFrame()).round(1).to_dict("records")}
    _t(f"{len(malos)} (ticker, año) de baja calidad", t0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(perfil, indent=1, default=str))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
