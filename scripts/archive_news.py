"""Snapshot diario de titulares → data/news/archive/AAAA-MM.parquet (PIT).

Reutiliza los fetchers de src.ml.news (RSS/Yahoo/DataBursátil; GDELT apagado).
Idempotente: correrlo dos veces el mismo día no duplica. Sin cuerpo del artículo.

Uso:
  python scripts/archive_news.py                # descarga + archiva + cobertura
  python scripts/archive_news.py --report-only  # solo recalcula cobertura
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.universes import UNIVERSES
from src.ml.news_archive import ARCHIVE_DIR, append_snapshot, coverage_report, load_archive, rows_from_articles

UNIVERSE = "🇲🇽 Acciones BMV"
OUT = ROOT / "data" / "eval" / "news_coverage.json"
PAUSE_S = 1.0  # cortesía entre tickers


def fetch_all(tickers: dict[str, str]):
    import pandas as pd
    from src.ml.news import _scrape_articles

    now = datetime.now(timezone.utc)
    frames = []
    for ticker, symbol in tickers.items():
        try:
            arts = _scrape_articles(ticker, symbol, days_back=2, use_gdelt=False)
        except Exception as exc:  # una fuente caída no detiene el resto
            print(f"  {ticker}: error {type(exc).__name__}")
            continue
        raw = [{"title": a.title, "url": a.url, "published_ts": a.published_ts, "source": a.source}
               for a in arts]
        frames.append(rows_from_articles(raw, ticker, now))
        print(f"  {ticker}: {len(raw)} titulares")
        time.sleep(PAUSE_S)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_report(tickers: list[str]) -> dict:
    arch = load_archive(ARCHIVE_DIR)
    if arch.empty:
        return {}
    start = arch["fetched_at"].min().strftime("%Y-%m-%d")
    end = arch["fetched_at"].max().strftime("%Y-%m-%d")
    rep = coverage_report(arch, tickers, start=start, end=end)
    rep["n_headlines_total"] = int(len(arch))
    rep["n_snapshot_days"] = int(arch["fetched_at"].dt.date.nunique())
    pt = rep["per_ticker"]
    tot = sum(v["n_ticker_days"] for v in pt.values())
    rep["pct_ticker_days_with_headline"] = round(
        100.0 * sum(v["days_with_headline"] for v in pt.values()) / tot, 2) if tot else 0.0
    rep["n_ticker_days_cumulative"] = tot
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data["archive"] = rep
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    tickers = UNIVERSES[UNIVERSE]["tickers"]
    if not args.report_only:
        df = fetch_all(tickers)
        print(f"nuevas filas archivadas: {append_snapshot(df, ARCHIVE_DIR)}")
    rep = write_report(list(tickers))
    if rep:
        print(f"cobertura ticker-días con ≥1 titular: {rep['pct_ticker_days_with_headline']}% "
              f"(ticker-días acumulados {rep['n_ticker_days_cumulative']})")
        print(f"→ {OUT}")


if __name__ == "__main__":
    main()
