"""Construye data/microstructure/hechos_ticker_map.json (código `hechos` → ticker MX).

Uso: .venv/bin/python scripts/build_hechos_map.py
Solo lee `data/microstructure/hechos/` y `data/market/*.MX.parquet`. No rellena días faltantes.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hechos_map import ALIASES, build_map, missing_days  # noqa: E402

HECHOS_DIR = ROOT / "data" / "microstructure" / "hechos"
MARKET_DIR = ROOT / "data" / "market"
OUT = ROOT / "data" / "microstructure" / "hechos_ticker_map.json"


def load_wide() -> pd.DataFrame:
    """Matriz fecha × código con vol_buy diario (vol_buy == vol_sell por construcción)."""
    frames = [pd.read_parquet(p, columns=["date", "ticker", "vol_buy"])
              for p in sorted(HECHOS_DIR.glob("*.parquet"))]
    raw = pd.concat(frames, ignore_index=True)
    wide = raw.groupby(["date", "ticker"])["vol_buy"].sum().unstack()
    wide.index = pd.to_datetime(wide.index)
    return wide


def load_market_volumes() -> dict[str, pd.Series]:
    out = {}
    for p in sorted(MARKET_DIR.glob("*.MX.parquet")):
        df = pd.read_parquet(p)
        out[p.name.removesuffix(".MX.parquet")] = df["Volume"]
    return out


def main() -> None:
    warnings.filterwarnings("ignore")  # columnas constantes en corrwith
    wide = load_wide()
    vols = load_market_volumes()
    mapa = build_map(vols, wide, ALIASES)
    mapeados = {v["ticker"] for v in mapa.values()}
    sin_mapa = sorted(set(vols) - mapeados)
    alta = sorted(v["ticker"] for v in mapa.values() if v["confianza"] == "alta")

    total_mx = pd.concat(vols.values(), axis=1).sum(axis=1)
    faltan = missing_days(wide.index, total_mx)

    out = {
        "generado": date.today().isoformat(),
        "fuente_hechos": f"{wide.index.min():%Y-%m-%d} → {wide.index.max():%Y-%m-%d} "
                         f"({len(wide)} días, {wide.shape[1]} códigos)",
        "n_tickers_mx": len(vols),
        "n_mapeados": len(mapeados),
        "n_confianza_alta": len(alta),
        "sin_mapear": sin_mapa,
        "mapa": dict(sorted(mapa.items(), key=lambda kv: kv[1]["ticker"])),
        "dias_faltantes_hechos": faltan,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"tickers MX: {len(vols)} | mapeados: {len(mapeados)} | confianza alta: {len(alta)}")
    for c, v in out["mapa"].items():
        print(f"  {v['ticker']:<14}{c:<14}{v['metodo']:<12}{v['confianza']:<7}"
              f"corr={v['corr']} 2º={v['segundo_corr']}")
    print("sin mapear:", sin_mapa)
    print(f"días faltantes: captura={len(faltan['captura'])} sin_mercado={len(faltan['sin_mercado'])}")
    print(f"→ {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
