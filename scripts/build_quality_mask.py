"""
Construye la máscara de calidad de datos para todo data/market/.

Salidas:
    data/quality/quality_mask.parquet  — formato largo (ticker, date, banderas, valid)
    data/eval/quality_summary.json     — filas marcadas por ticker y por regla

Uso:
    python scripts/build_quality_mask.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.quality import (
    FLAGS, LIQ_THRESHOLD, MARKET_DIR, MASK_PATH, compute_quality_mask,
    kind_for, universe_for,
)

SUMMARY_PATH = ROOT / "data" / "eval" / "quality_summary.json"


def main() -> None:
    frames, per_ticker = [], {}
    for path in sorted(MARKET_DIR.glob("*.parquet")):
        ticker = path.stem
        df = pd.read_parquet(path)
        if df.empty:
            per_ticker[ticker] = {"rows": 0, "note": "parquet vacío"}
            continue
        kind = kind_for(ticker)
        m = compute_quality_mask(df, kind=kind, liq_threshold=LIQ_THRESHOLD[universe_for(ticker)])
        per_ticker[ticker] = {
            "rows": int(len(m)),
            "kind": kind,
            "invalid": int((~m["valid"]).sum()),
            "pct_invalid": round(float((~m["valid"]).mean() * 100), 2),
            **{f: int(m[f].sum()) for f in FLAGS},
        }
        long = m.copy()
        long.insert(0, "date", pd.DatetimeIndex(m.index))
        long.insert(0, "ticker", ticker)
        frames.append(long.reset_index(drop=True))

    mask = pd.concat(frames, ignore_index=True)
    MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    mask.to_parquet(MASK_PATH, index=False)

    totals = {f: int(mask[f].sum()) for f in FLAGS}
    totals.update(rows=int(len(mask)), invalid=int((~mask["valid"]).sum()))
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps({"totals": totals, "tickers": per_ticker}, indent=2))

    print(f"{len(frames)} tickers, {totals['rows']} filas, {totals['invalid']} no válidas")
    print(f"parquet: {MASK_PATH.stat().st_size / 1e6:.1f} MB")
    for t in ("POSADASA.MX", "ASURB.MX", "NAFTRAC.MX", "MXN=X"):
        print(t, per_ticker.get(t))


if __name__ == "__main__":
    main()
