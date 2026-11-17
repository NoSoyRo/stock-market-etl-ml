"""
Descarga y procesa noticias para todos los universos.

Fuentes: Yahoo Finance, Google News RSS, El Financiero, Expansión, GDELT.
Sentiment: Gemini 2.5-flash (si GEMINI_API_KEY está en .env), fallback keywords.

Salida:
  data/news/YYYY-MM-DD.json  — features por ticker para hoy
  data/news/latest.json      — alias siempre actualizado

Uso:
  python scripts/fetch_news.py                   # todos los universos
  python scripts/fetch_news.py --universe bmv    # solo BMV
  python scripts/fetch_news.py --gdelt           # incluir GDELT (más lento, backfill)
  python scripts/fetch_news.py --days 14         # ventana de 14 días
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.data.universes import UNIVERSES
from src.ml.news import batch_news_features, NEWS_FEATURE_COLS, NEWS_DIR

logger = logging.getLogger(__name__)

UNIVERSE_SLUGS = {
    "bmv":    "🇲🇽 Acciones BMV",
    "usa":    "🇺🇸 Acciones USA (SIC)",
    "etf":    "📦 ETFs",
    "crypto": "₿ Cripto (proxies GBM)",
    "fibra":  "🏢 FIBRAs (bienes raíces MX)",
}


def run_fetch_news(
    *,
    days_back: int = 7,
    universe: str | None = None,
    gdelt: bool = False,
    save_cache: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Descarga noticias y opcionalmente persiste cache.

    universe: slug (bmv, usa, ...) o None para todos los universos.
    """
    selected = (
        {universe: UNIVERSE_SLUGS[universe]}
        if universe
        else UNIVERSE_SLUGS
    )

    all_results: dict[str, list[dict]] = {}
    ticker_counts: dict[str, int] = {}

    for slug, uname in selected.items():
        if uname not in UNIVERSES:
            if verbose:
                logger.warning("Universo no encontrado: %s", uname)
            continue

        tickers = UNIVERSES[uname]["tickers"]
        if verbose:
            logger.info("Noticias %s — %d tickers, %d días", uname, len(tickers), days_back)

        df = batch_news_features(
            tickers=tickers,
            days_back=days_back,
            use_gdelt=gdelt,
            verbose=verbose,
            save_cache=False,
        )
        records = df.reset_index().to_dict(orient="records")
        all_results[slug] = records
        ticker_counts[slug] = len(records)

        if verbose:
            sent_mean = df["news_sent_score"].mean() if len(df) else 0.0
            n_total = int(df["n_articles"].sum()) if "n_articles" in df.columns else 0
            logger.info(
                "  %s: %d tickers, %d artículos, sent=%+.3f",
                slug, len(records), n_total, sent_mean,
            )

    today = datetime.now().strftime("%Y-%m-%d")
    paths: dict[str, str] = {}

    if save_cache and all_results:
        NEWS_DIR.mkdir(parents=True, exist_ok=True)
        for slug, records in all_results.items():
            path = NEWS_DIR / f"{slug}_{today}.json"
            path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
            paths[f"universe_{slug}"] = str(path)

        all_records = [r for records in all_results.values() for r in records]
        latest_path = NEWS_DIR / "latest.json"
        dated_path = NEWS_DIR / f"{today}.json"
        latest_path.write_text(json.dumps(all_records, ensure_ascii=False, indent=2))
        dated_path.write_text(json.dumps(all_records, ensure_ascii=False, indent=2))
        paths["latest"] = str(latest_path)
        paths["dated"] = str(dated_path)

    if save_cache and all_results:
        from src.data.news_medallion import build_gold
        build_gold(days_back=days_back)

    return {
        "days_back": days_back,
        "universe": universe or "all",
        "gdelt": gdelt,
        "tickers_by_universe": ticker_counts,
        "total_tickers": sum(ticker_counts.values()),
        "paths": paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Descargar noticias financieras")
    parser.add_argument("--universe", choices=list(UNIVERSE_SLUGS.keys()), default=None,
                        help="Procesar solo un universo")
    parser.add_argument("--days", type=int, default=7,
                        help="Ventana de noticias en días (default: 7)")
    parser.add_argument("--gdelt", action="store_true",
                        help="Incluir GDELT para historial (más lento)")
    parser.add_argument("--no-cache", action="store_true",
                        help="No guardar cache en disco")
    args = parser.parse_args()

    gemini_ok = bool(os.environ.get("GEMINI_API_KEY"))
    sentiment_mode = "Gemini 2.5-flash" if gemini_ok else "keywords (fallback)"

    print(f"\n{'═'*62}")
    print(f"  NOTICIAS FINANCIERAS  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Ventana: {args.days} días  |  Sentiment: {sentiment_mode}")
    if args.gdelt:
        print(f"  GDELT: activado (backfill histórico)")
    print(f"{'═'*62}\n")

    result = run_fetch_news(
        days_back=args.days,
        universe=args.universe,
        gdelt=args.gdelt,
        save_cache=not args.no_cache,
        verbose=True,
    )

    if result.get("paths"):
        print(f"\n  Guardado en: {result['paths'].get('dated', '—')}")
        print(f"  Alias:       {result['paths'].get('latest', '—')}")

    print(f"\n  Features disponibles: {NEWS_FEATURE_COLS}")
    print(f"  Próximo paso: estas features se integran automáticamente en")
    print(f"  generate_all_signals.py al correr las señales.\n")


if __name__ == "__main__":
    main()
