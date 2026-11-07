"""Diagnóstico del manifiesto de GDELT: ¿'empty' = sin resultados o nunca consultado?

Solo lee manifest.json y la caché de chunks. Con --probe re-consulta una muestra
pequeña (respeta el throttle de 60 s de news_sources). Escribe la clave
'manifest_diagnosis' de data/eval/news_coverage.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.news_archive import classify_manifest
from src.ml.news_history import (MANIFEST_PATH, _chunk_cache_path, _chunk_ranges,
                                 _load_chunk_cache, _query_for_ticker)

OUT = ROOT / "data" / "eval" / "news_coverage.json"


def diagnose() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    rows = {}
    for key, status in manifest.items():
        ticker, s, e = key.split(":")
        c0 = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        c1 = datetime.fromisoformat(e).replace(tzinfo=timezone.utc)
        q = _query_for_ticker(ticker)
        chunks = [_load_chunk_cache(_chunk_cache_path(ticker, a, b, q)) for a, b in _chunk_ranges(c0, c1, 90)]
        n_cached = sum(c is not None for c in chunks)
        n_art = sum(c is not None and len(c) > 0 for c in chunks)
        rows[key] = {"manifest": status, "query": q, "chunks_expected": len(chunks),
                     "chunks_cached": n_cached, "chunks_with_articles": n_art,
                     "diagnosis": classify_manifest(status, len(chunks), n_cached, n_art)}
    return {"summary": dict(Counter(r["diagnosis"] for r in rows.values())), "entries": rows}


def probe(tickers: list[str]) -> list[dict]:
    """Re-consulta 1 chunk reciente (60 días) por ticker; devuelve conteos."""
    from datetime import timedelta
    from src.ml.news_sources import fetch_gdelt_range

    end = datetime.now(timezone.utc)
    out = []
    for t in tickers:
        arts, ok = fetch_gdelt_range(_query_for_ticker(t), t, end - timedelta(days=60), end, max_items=250)
        rel = sum(a.get("relevance_score", 1.0) >= 0.25 for a in arts)
        out.append({"ticker": t, "ok": ok, "n_raw": len(arts), "n_relevant": rel,
                    "result": "429/error" if not ok else ("vacío real" if not arts else "con resultados")})
        print(out[-1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", nargs="*", help="tickers a re-consultar en GDELT (lento: 60 s entre requests)")
    args = ap.parse_args()
    res = diagnose()
    if args.probe:
        res["probe_last_60d"] = probe(args.probe)
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data["manifest_diagnosis"] = res
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(res["summary"])


if __name__ == "__main__":
    main()
