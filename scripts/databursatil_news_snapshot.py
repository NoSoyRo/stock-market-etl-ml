#!/usr/bin/env python3
"""
Snapshot diario DataBursátil V2 (noticias + cables) → histórico acumulativo.

Cron recomendado (día hábil, post-cierre):
  0 18 * * 1-5 cd /path/stock-market && venv/bin/python scripts/databursatil_news_snapshot.py

Uso:
  python scripts/databursatil_news_snapshot.py
  python scripts/databursatil_news_snapshot.py --rebuild-panel --universe bmv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from scripts.backtest_topk import load_universe_tickers
from src.ml.databursatil_news import (
    build_daily_panel_from_snapshots,
    fetch_creditos,
    merge_into_daily_features_parquet,
    save_daily_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="DataBursátil noticias/cables snapshot")
    parser.add_argument("--date", help="YYYY-MM-DD (default hoy UTC)")
    parser.add_argument("--rebuild-panel", action="store_true",
                        help="Reconstruir daily_features.parquet desde snapshots")
    parser.add_argument("--universe", default="bmv")
    args = parser.parse_args()

    cred = fetch_creditos()
    if cred.get("disponibles") is not None and cred["disponibles"] < 100:
        print(f"⚠ Créditos bajos: {cred['disponibles']}")

    print("\n══ DataBursátil News Snapshot ══")
    save_daily_snapshot(date=args.date)
    from src.data.news_medallion import build_gold
    build_gold()

    if args.rebuild_panel:
        tickers = list(load_universe_tickers(args.universe).keys())
        print(f"\n  Reconstruyendo panel ({len(tickers)} tickers)…")
        panel = build_daily_panel_from_snapshots(tickers, verbose=True)
        merged = merge_into_daily_features_parquet(panel)
        stats = {
            "rows": len(merged),
            "tickers": int(merged["ticker"].nunique()) if not merged.empty else 0,
            "date_min": str(merged["date"].min()) if not merged.empty else None,
            "date_max": str(merged["date"].max()) if not merged.empty else None,
        }
        print(f"\n✅ Panel: {json.dumps(stats, indent=2)}\n")


if __name__ == "__main__":
    main()
