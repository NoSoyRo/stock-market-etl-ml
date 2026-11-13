"""
Panel histórico de noticias point-in-time para backtest / walk-forward.

Almacena:
  data/news/history/articles.parquet   — artículos GDELT por ticker
  data/news/history/daily_features.parquet — features diarias (ticker × date)

Uso:
    from src.ml.news_history import load_daily_features, merge_news_history
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.news import (
    NEWS_DIR,
    NEWS_FEATURE_COLS,
    Article,
    _SEARCH_QUERIES,
    _compute_features_from_articles,
    _dict_to_article,
    _score_keywords,
)
from src.ml.news_sources import ENTITY_ALIASES, fetch_gdelt_range

logger = logging.getLogger(__name__)

HISTORY_DIR = NEWS_DIR / "history"
ARTICLES_PATH = HISTORY_DIR / "articles.parquet"
DAILY_PATH = HISTORY_DIR / "daily_features.parquet"
MANIFEST_PATH = HISTORY_DIR / "manifest.json"
CHUNK_CACHE_DIR = HISTORY_DIR / "gdelt_chunks"

GDELT_SLEEP_S = 0.0  # rate limit vía gdelt_request_lock en news_sources
DEFAULT_CHUNK_DAYS = 90
CHUNK_FETCH_RETRIES = 2


def _query_for_ticker(ticker: str) -> str:
    """Frase única para GDELT (OR compuesto suele devolver vacío)."""
    info = _SEARCH_QUERIES.get(ticker, (f"{ticker} Mexico", ticker))
    company = (info[1] if len(info) > 1 else ticker).strip()
    if company:
        return company
    for alias in ENTITY_ALIASES.get(ticker, []):
        if alias.strip():
            return alias.strip()
    return f"{ticker} Mexico"


MIN_GDELT_RELEVANCE = 0.25


def _chunk_ranges(start: datetime, end: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    cur = start
    ranges: list[tuple[datetime, datetime]] = []
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1, hours=23, minutes=59), end)
        ranges.append((cur, chunk_end))
        cur = chunk_end + timedelta(seconds=1)
    return ranges


def _chunk_cache_path(ticker: str, c0: datetime, c1: datetime, query: str) -> Path:
    CHUNK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{ticker}|{c0.date()}|{c1.date()}|{query}"
    digest = hashlib.md5(key.encode()).hexdigest()[:16]
    return CHUNK_CACHE_DIR / f"{ticker}_{digest}.json"


def _load_chunk_cache(path: Path) -> list[dict] | None:
    """None = cache inválido o fallo previo → re-fetch."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        return data if data else None
    if data.get("status") == "ok":
        return data.get("articles", [])
    return None


def _save_chunk_cache(path: Path, articles: list[dict]) -> None:
    path.write_text(json.dumps({
        "status": "ok",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "articles": articles,
    }, ensure_ascii=False))


def clean_bad_chunk_caches() -> int:
    """Elimina caches vacíos legacy (429 cacheados como [])."""
    if not CHUNK_CACHE_DIR.exists():
        return 0
    removed = 0
    for path in CHUNK_CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
            removed += 1
            continue
        if isinstance(data, list) and not data:
            path.unlink(missing_ok=True)
            removed += 1
        elif isinstance(data, dict) and data.get("status") != "ok":
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def fetch_ticker_gdelt_history(
    ticker: str,
    start: datetime,
    end: datetime,
    *,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    verbose: bool = True,
    force: bool = False,
    cache_only: bool = False,
    pending_out: list | None = None,
) -> list[Article]:
    """Descarga artículos GDELT en chunks para un ticker.

    Si se pasa ``pending_out``, se le agregan los chunks que NO se pudieron
    consultar (429 tras los reintentos, o sin caché en ``cache_only``): un chunk
    sin consultar no es lo mismo que un chunk sin resultados.
    """
    query = _query_for_ticker(ticker)
    articles: list[Article] = []
    seen: set[str] = set()

    for i, (c0, c1) in enumerate(_chunk_ranges(start, end, chunk_days), start=1):
        if verbose:
            print(f"    chunk {i}: {c0.date()} → {c1.date()}", end=" ", flush=True)
        cache_path = _chunk_cache_path(ticker, c0, c1, query)
        raw: list[dict] | None = None if force else _load_chunk_cache(cache_path)
        if raw is None:
            if cache_only:
                if verbose:
                    print("skip (sin cache)", end=" ", flush=True)
                raw = []
                if pending_out is not None:
                    pending_out.append((c0, c1))
            else:
                for attempt in range(CHUNK_FETCH_RETRIES):
                    fetched, ok = fetch_gdelt_range(query, ticker, c0, c1, max_items=250)
                    if ok:
                        raw = fetched
                        _save_chunk_cache(cache_path, raw)
                        from src.data.news_medallion import ingest
                        ingest("gdelt", [dict(d, ticker=ticker) for d in raw])
                        break
                    if attempt + 1 < CHUNK_FETCH_RETRIES and verbose:
                        print("(rate limit, reintento…)", end=" ", flush=True)
                if raw is None:
                    raw = []
                    if pending_out is not None:
                        pending_out.append((c0, c1))
                    if verbose:
                        print("pendiente", end=" ", flush=True)
        n_new = 0
        for d in raw:
            if d.get("relevance_score", 1.0) < MIN_GDELT_RELEVANCE:
                continue
            art = _dict_to_article(d, ticker)
            art.sentiment = _score_keywords(art.text())
            key = art.url or art.title
            if key and key not in seen:
                seen.add(key)
                articles.append(art)
                n_new += 1
        if verbose:
            print(f"+{n_new} (total {len(articles)})")
    return articles


def articles_to_frame(articles: list[Article]) -> pd.DataFrame:
    rows = []
    for a in articles:
        if not a.published_ts:
            continue
        rows.append({
            "ticker": a.ticker,
            "published_dt": a.published_dt,
            "title": a.title,
            "url": a.url,
            "source": a.source,
            "sentiment": a.sentiment,
            "relevance_score": a.relevance_score,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "published_dt", "title", "url", "source",
            "sentiment", "relevance_score",
        ])
    df = pd.DataFrame(rows)
    df["published_dt"] = pd.to_datetime(df["published_dt"], utc=True)
    return df.sort_values(["ticker", "published_dt"])


def build_daily_features(
    articles_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    *,
    days_back: int = 7,
    verbose: bool = False,
) -> pd.DataFrame:
    """Calcula features diarias point-in-time por ticker."""
    if articles_df.empty:
        return pd.DataFrame(columns=["date", "ticker", *NEWS_FEATURE_COLS])

    records: list[dict] = []
    tickers = sorted(articles_df["ticker"].unique())
    lookback = days_back * 2

    for ticker in tickers:
        sub = articles_df[articles_df["ticker"] == ticker].sort_values("published_dt")
        if sub.empty:
            continue
        pub_ns = sub["published_dt"].values.astype("datetime64[ns]")

        for dt in trading_dates:
            as_of = pd.Timestamp(dt)
            if as_of.tzinfo is None:
                as_of = as_of.tz_localize("UTC")
            else:
                as_of = as_of.tz_convert("UTC")
            end = as_of.replace(hour=23, minute=59, second=59)
            win_start = end - timedelta(days=lookback)
            end_ns = np.datetime64(end.to_datetime64())
            win_ns = np.datetime64(win_start.to_datetime64())
            i0 = int(pub_ns.searchsorted(win_ns, side="left"))
            i1 = int(pub_ns.searchsorted(end_ns, side="right"))
            window = sub.iloc[i0:i1]

            if window.empty:
                records.append({
                    "date": as_of.normalize(),
                    "ticker": ticker,
                    **{c: 0.0 for c in NEWS_FEATURE_COLS},
                    "n_articles": 0,
                })
                continue

            arts = [
                Article(
                    ticker=ticker,
                    title=r.title,
                    summary="",
                    url=r.url or "",
                    published_ts=int(r.published_dt.timestamp()),
                    source=r.source,
                    sentiment=float(r.sentiment),
                    relevance_score=float(r.relevance_score),
                )
                for r in window.itertuples(index=False)
            ]
            feat = _compute_features_from_articles(
                arts, days_back=days_back, as_of=end.to_pydatetime(), apply_dedup=True,
            )
            records.append({
                "date": as_of.normalize(),
                "ticker": ticker,
                **{c: feat.get(c, 0.0) for c in NEWS_FEATURE_COLS},
                "n_articles": feat.get("n_articles", 0),
            })
        if verbose:
            n_with = sum(1 for r in records if r["ticker"] == ticker and r["n_articles"] > 0)
            print(f"  {ticker}: {n_with}/{len(trading_dates)} días con artículos")

    return pd.DataFrame(records)


def build_history_panel(
    tickers: dict[str, str],
    start: str,
    end: str,
    *,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    force: bool = False,
    cache_only: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pipeline completo: GDELT → articles.parquet → daily_features.parquet.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(hour=23, minute=59, tzinfo=timezone.utc)

    manifest: dict = {}
    if MANIFEST_PATH.exists() and not force:
        manifest = json.loads(MANIFEST_PATH.read_text())

    all_frames: list[pd.DataFrame] = []
    if ARTICLES_PATH.exists() and not force:
        existing = pd.read_parquet(ARTICLES_PATH)
        if not existing.empty:
            all_frames.append(existing)

    for name in tickers:
        key = f"{name}:{start}:{end}"
        query = _query_for_ticker(name)
        chunks_done = all(
            _load_chunk_cache(p) is not None
            for p in (
                _chunk_cache_path(name, c0, c1, query)
                for c0, c1 in _chunk_ranges(start_dt, end_dt, chunk_days)
            )
        )
        if not force and chunks_done and manifest.get(key) in ("ok", "empty"):
            if verbose:
                print(f"  {name}: skip (chunks en cache)")
            continue
        if verbose:
            print(f"  {name}: GDELT backfill…")
        try:
            pending: list = []
            arts = fetch_ticker_gdelt_history(
                name, start_dt, end_dt,
                chunk_days=chunk_days, verbose=verbose, force=force, cache_only=cache_only,
                pending_out=pending,
            )
            df = articles_to_frame(arts)
            if not df.empty:
                all_frames.append(df)
                manifest[key] = "ok"
            elif pending:
                # Vacío porque no se consultó (429 / sin caché), no porque GDELT no tenga nada.
                manifest[key] = "pending"
            else:
                manifest[key] = "empty"
        except Exception as exc:
            logger.error("%s: %s", name, exc)
            manifest[key] = f"error:{exc}"

    if all_frames:
        articles = pd.concat(all_frames, ignore_index=True)
        articles = articles.drop_duplicates(subset=["ticker", "url", "title"], keep="last")
        articles = articles.sort_values(["ticker", "published_dt"])
        articles.to_parquet(ARTICLES_PATH, index=False)
    else:
        articles = pd.read_parquet(ARTICLES_PATH) if ARTICLES_PATH.exists() else pd.DataFrame()

    if articles.empty:
        daily = pd.DataFrame(columns=["date", "ticker", *NEWS_FEATURE_COLS])
        daily.to_parquet(DAILY_PATH, index=False)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
        return daily

    trading_dates = pd.bdate_range(start=start, end=end)
    if verbose:
        print(f"\n  Computando features diarias ({len(trading_dates)} días × {articles['ticker'].nunique()} tickers)…")
    daily = build_daily_features(articles, trading_dates, verbose=verbose)
    daily.to_parquet(DAILY_PATH, index=False)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    if verbose:
        cov = (daily["news_sent_score"].abs() > 1e-6).mean() if len(daily) else 0
        print(f"  Guardado: {DAILY_PATH} ({len(daily):,} filas, {cov:.1%} con sent≠0)")
    return daily


def load_daily_features() -> pd.DataFrame | None:
    if not DAILY_PATH.exists():
        return None
    df = pd.read_parquet(DAILY_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def merge_news_history(panel: pd.DataFrame, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge point-in-time: panel con DatetimeIndex + columna ticker."""
    out = panel.copy()
    daily = daily if daily is not None else load_daily_features()

    if daily is None or daily.empty:
        for col in NEWS_FEATURE_COLS:
            out[col] = 0.0
        return out

    idx_name = out.index.name or "date"
    merged = out.reset_index()
    date_col = merged.columns[0]
    merged["_date"] = pd.to_datetime(merged[date_col]).dt.normalize()
    daily = daily.copy()
    daily["_date"] = pd.to_datetime(daily["date"], utc=True).dt.tz_convert(None).dt.normalize()
    merged["_date"] = pd.to_datetime(merged["_date"], utc=True).dt.tz_convert(None).dt.normalize()

    cols = ["_date", "ticker", *NEWS_FEATURE_COLS]
    extra = [c for c in ("n_articles",) if c in daily.columns]
    merged = merged.merge(daily[cols + extra], on=["_date", "ticker"], how="left")
    for col in NEWS_FEATURE_COLS:
        merged[col] = merged[col].fillna(0.0)
    merged = merged.drop(columns=["_date"])
    return merged.set_index(date_col)


def history_stats() -> dict:
    daily = load_daily_features()
    if daily is None or daily.empty:
        return {"status": "missing"}
    return {
        "rows": len(daily),
        "tickers": int(daily["ticker"].nunique()),
        "date_min": str(daily["date"].min().date()),
        "date_max": str(daily["date"].max().date()),
        "pct_with_articles": round(float((daily.get("n_articles", 0) > 0).mean()) * 100, 1),
        "pct_nonzero_sent": round(float((daily["news_sent_score"].abs() > 1e-6).mean()) * 100, 1),
    }
