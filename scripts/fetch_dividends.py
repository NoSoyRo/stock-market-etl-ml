"""Descarga dividendos (BMV, FIBRA, USA) y resume cobertura y rendimiento (M3b).

Uso:  .venv/bin/python scripts/fetch_dividends.py [--force] [--groups bmv fibra usa]
Salidas: data/dividends/<símbolo>.parquet y data/eval/dividends_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.store import UNIVERSE_GROUPS, load  # noqa: E402
from src.data.total_return import fetch_dividends  # noqa: E402

OUT = ROOT / "data" / "eval" / "dividends_summary.json"


def annual_yield(prices: pd.DataFrame | None, divs: pd.Series) -> dict:
    """Rendimiento TTM (últimos 365 d / último Close) y medio por año calendario completo."""
    if prices is None or len(prices) == 0 or len(divs) == 0:
        return {"yield_ttm_pct": None, "yield_medio_anual_pct": None}
    close = prices["Close"]
    last = close.index.max()
    ttm = divs[divs.index > last - pd.Timedelta(days=365)].sum()
    por_anio = divs.groupby(divs.index.year).sum()
    px = close.groupby(close.index.year).mean()
    anios = [y for y in px.index if y < last.year and y >= close.index.min().year + 1]
    ys = [por_anio.get(y, 0.0) / px[y] * 100 for y in anios]
    return {
        "yield_ttm_pct": round(float(ttm / close.iloc[-1] * 100), 2),
        "yield_medio_anual_pct": round(float(sum(ys) / len(ys)), 2) if ys else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignora el caché")
    ap.add_argument("--groups", nargs="+", default=["bmv", "fibra", "usa"])
    args = ap.parse_args()

    resumen, errores, sin_div = {}, {}, []
    for g in args.groups:
        for name, symbol in UNIVERSE_GROUPS[g].items():
            try:
                divs = fetch_dividends(symbol, force=args.force)
            except Exception as exc:
                errores[symbol] = str(exc)[:200]
                print(f"ERROR {symbol}: {exc}")
                continue
            row = {"grupo": g, "n_dividendos": int(len(divs)),
                   "primera": str(divs.index.min().date()) if len(divs) else None,
                   "ultima": str(divs.index.max().date()) if len(divs) else None}
            row.update(annual_yield(load(symbol), divs))
            resumen[symbol] = row
            if len(divs) == 0:
                sin_div.append(symbol)
            print(f"{symbol:14s} n={row['n_dividendos']:3d} ttm={row['yield_ttm_pct']} medio={row['yield_medio_anual_pct']}")

    sin_mx = [s for s in sin_div if s.endswith(".MX")]
    res = {"descargados": len(resumen), "errores": errores, "sin_dividendos": sin_div,
           "sin_dividendos_MX": sin_mx, "por_simbolo": resumen}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nOK {len(resumen)} · errores {len(errores)} · sin dividendos {len(sin_div)} ({len(sin_mx)} .MX)")


if __name__ == "__main__":
    main()
