"""Archivo PIT de titulares: un snapshot diario con marca de descarga.

Guarda solo metadatos (sin cuerpo): fetched_at UTC, published_dt, ticker,
titular, fuente y URL. La clave de unicidad es (ticker, url|título); si un
titular reaparece en un snapshot posterior se conserva el PRIMER fetched_at,
que es lo que garantiza el punto en el tiempo (PIT).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT / "data" / "news" / "archive"
ARCHIVE_COLUMNS = ["fetched_at", "published_dt", "ticker", "title", "source", "url"]


def _key(df: pd.DataFrame) -> pd.Series:
    """Clave de dedupe: ticker + URL (o título si no hay URL)."""
    ident = df["url"].where(df["url"].astype(str).str.len() > 0, df["title"])
    return df["ticker"].astype(str) + "|" + ident.astype(str)


def rows_from_articles(articles: list[dict[str, Any]], ticker: str,
                       fetched_at: datetime) -> pd.DataFrame:
    """Dicts de los fetchers → filas del archivo (sin cuerpo), deduplicadas."""
    fetched = pd.Timestamp(fetched_at)
    fetched = fetched.tz_localize("UTC") if fetched.tzinfo is None else fetched.tz_convert("UTC")
    rows = []
    for a in articles:
        title = (a.get("title") or "").strip()
        if not title:
            continue
        ts = int(a.get("published_ts") or 0)
        rows.append({
            "fetched_at": fetched,
            "published_dt": pd.Timestamp(ts, unit="s", tz="UTC") if ts > 0 else pd.NaT,
            "ticker": ticker,
            "title": title,
            "source": a.get("source") or "unknown",
            "url": a.get("url") or "",
        })
    df = pd.DataFrame(rows, columns=ARCHIVE_COLUMNS)
    if df.empty:
        return df
    return df.loc[~_key(df).duplicated(keep="first")].reset_index(drop=True)


def load_archive(archive_dir: Path = ARCHIVE_DIR) -> pd.DataFrame:
    files = sorted(Path(archive_dir).glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=ARCHIVE_COLUMNS)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def append_snapshot(df: pd.DataFrame, archive_dir: Path = ARCHIVE_DIR) -> int:
    """Añade filas nuevas a AAAA-MM.parquet (mes de fetched_at). Devuelve nº nuevas.

    Idempotente: lo ya archivado (en cualquier mes) no se vuelve a escribir.
    """
    if df.empty:
        return 0
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    known = set(_key(load_archive(archive_dir)).tolist()) if any(archive_dir.glob("*.parquet")) else set()
    df = df.loc[~_key(df).duplicated(keep="first")]
    new = df.loc[~_key(df).isin(known)]
    if new.empty:
        return 0
    month = new["fetched_at"].dt.strftime("%Y-%m")
    for m, part in new.groupby(month):
        path = archive_dir / f"{m}.parquet"
        old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=ARCHIVE_COLUMNS)
        out = pd.concat([old, part[ARCHIVE_COLUMNS]], ignore_index=True)
        out.to_parquet(path, index=False)
    return len(new)


def coverage_report(df: pd.DataFrame, tickers: list[str], *, start: str, end: str) -> dict:
    """Cobertura por ticker: ticker-días (calendario) con ≥ 1 titular publicado."""
    days = pd.date_range(start, end, freq="D")
    n_days = len(days)
    out: dict[str, dict] = {}
    for t in tickers:
        sub = df[df["ticker"] == t] if not df.empty else df
        pub = pd.to_datetime(sub["published_dt"], utc=True).dt.tz_localize(None).dt.normalize() \
            if len(sub) else pd.Series([], dtype="datetime64[ns]")
        with_news = int(pub[pub.isin(days)].nunique()) if len(pub) else 0
        out[t] = {
            "n_headlines": int(len(sub)),
            "n_ticker_days": n_days,
            "days_with_headline": with_news,
            "pct_days_with_headline": round(100.0 * with_news / n_days, 2) if n_days else 0.0,
        }
    return {"start": start, "end": end, "n_days": n_days, "per_ticker": out}


def classify_manifest(status: str, n_expected: int, n_cached: int,
                      n_cached_with_articles: int) -> str:
    """Distingue 'sin resultados' de 'nunca consultado' (429 guardado como empty)."""
    if status.startswith("error"):
        return "error"
    if status == "ok":
        return "ok"
    if n_cached == 0:
        return "nunca_consultado"
    if n_cached < n_expected:
        return "parcial_pendiente"
    return "sin_resultados" if n_cached_with_articles == 0 else "ok"
