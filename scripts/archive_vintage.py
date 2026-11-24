"""Archivo diario de vintages: guarda las últimas N filas OHLCV de los tickers BMV del panel
tal como las entrega el store HOY, para medir revisiones de datos más adelante (idea 4.7).

Salida: data/vintages/YYYY-MM-DD.parquet (pequeño, < 200 KB). Idempotente: si el archivo del día
ya existe no se toca (se conserva la primera captura del día).

Uso:
    .venv/bin/python scripts/archive_vintage.py --dry-run   # solo imprime
    .venv/bin/python scripts/archive_vintage.py             # guarda

Cron (NO instalado; lo decide el usuario). Después del cierre y de la actualización del store,
p. ej. lunes-viernes 18:30 hora local:
    30 18 * * 1-5  cd /home/gomiten/Documents/stock-market-etl-ml && .venv/bin/python scripts/archive_vintage.py >> logs/vintage.log 2>&1
Debe correr DESPUÉS de que el ETL actualice `data/market/` (ver scripts/daily_cron.sh).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "vintages"
N_ROWS = 10
COLS = ["Open", "High", "Low", "Close", "Volume"]


def build_snapshot(
    tickers: list[str],
    getter: Callable[[str], pd.DataFrame | None],
    n: int = N_ROWS,
) -> pd.DataFrame:
    """Últimas `n` filas OHLCV por ticker en formato largo: ticker, date, Open..Volume."""
    frames = []
    for t in tickers:
        df = getter(t)
        if df is None or df.empty:
            continue
        tail = df.sort_index().tail(n)[COLS].copy()
        tail.insert(0, "date", pd.DatetimeIndex(tail.index).normalize())
        tail.insert(0, "ticker", t)
        frames.append(tail.reset_index(drop=True))
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", *COLS])
    return pd.concat(frames, ignore_index=True)


def write_snapshot(snap: pd.DataFrame, day: str, out_dir: Path, dry_run: bool = False) -> dict:
    """Escribe `<out_dir>/<day>.parquet` salvo que exista (idempotente) o sea dry-run."""
    path = Path(out_dir) / f"{day}.parquet"
    info = {"ruta": str(path), "filas": len(snap), "tickers": int(snap["ticker"].nunique())}
    if dry_run:
        return {**info, "estado": "dry-run"}
    if path.exists():
        return {**info, "estado": "existe"}
    path.parent.mkdir(parents=True, exist_ok=True)
    snap.to_parquet(path, index=False, compression="zstd")
    return {**info, "estado": "creado", "bytes": path.stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="imprime lo que guardaría, sin escribir")
    ap.add_argument("--day", default=date.today().isoformat(), help="fecha de captura (YYYY-MM-DD)")
    args = ap.parse_args()

    from scripts.backtest_topk import load_universe_tickers
    from src.data.store import get

    tickers = sorted(set(load_universe_tickers("bmv").values()))
    snap = build_snapshot(tickers, get, N_ROWS)
    omitidos = sorted(set(tickers) - set(snap["ticker"]))
    res = write_snapshot(snap, args.day, OUT_DIR, dry_run=args.dry_run)

    print(f"[{res['estado']}] {res['ruta']}: {res['filas']} filas, {res['tickers']} tickers"
          + (f", {res['bytes']} bytes" if "bytes" in res else ""))
    if omitidos:
        print("sin datos (omitidos):", omitidos)
    if args.dry_run and not snap.empty:
        print(snap.groupby("ticker")["date"].agg(["min", "max"]).head(5).to_string())
        print("última fila AC.MX:", snap[snap.ticker == "AC.MX"].tail(1).to_dict("records"))


if __name__ == "__main__":
    main()
