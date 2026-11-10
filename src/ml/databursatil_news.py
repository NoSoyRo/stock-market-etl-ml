"""
DataBursátil V2 — noticias, cables y snapshots históricos acumulativos.

Endpoints:
  GET /v2/noticias   — titulares n, contenido c, fuente f
  GET /v2/cables     — cables por categoría (t, f)
  GET /v2/creditos   — créditos disponibles

Histórico: la API no expone fecha en noticias/cables; acumulamos un snapshot
por día hábil en data/news/history/databursatil/YYYY-MM-DD.json
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.ml.news import (
    NEWS_FEATURE_COLS,
    Article,
    _compute_features_from_articles,
    _score_keywords,
)
from src.ml.news_sources import (
    _article,
    _parse_databursatil_date,
    _with_relevance,
    get_entity_aliases,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = ROOT / "data" / "news" / "history" / "databursatil"
BASE_URL = "https://api.databursatil.com"

DEFAULT_CABLE_CATEGORIES = "boletines,top,valuaciones,flash,generales"


def _token() -> str | None:
    return os.getenv("DATABURSATIL_API_TOKEN") or None


def fetch_creditos() -> dict[str, Any]:
    """GET /v2/creditos — créditos restantes del token."""
    token = _token()
    if not token:
        return {"disponibles": None, "error": "no_token"}
    try:
        r = requests.get(
            f"{BASE_URL}/v2/creditos",
            params={"token": token},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        disp = data.get("disponibles")
        try:
            disp_i = int(str(disp).replace(",", ""))
        except (ValueError, TypeError):
            disp_i = None
        return {"disponibles": disp_i, "raw": data}
    except Exception as exc:
        logger.warning("creditos: %s", exc)
        return {"disponibles": None, "error": str(exc)}


def fetch_noticias_raw() -> list[dict[str, str]]:
    """GET /v2/noticias — lista [{n, c, f}, ...]."""
    token = _token()
    if not token:
        return []
    try:
        r = requests.get(
            f"{BASE_URL}/v2/noticias",
            params={"token": token},
            timeout=30,
        )
        if not r.ok:
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.debug("noticias: %s", exc)
        return []


def fetch_cables_raw(categorias: str = DEFAULT_CABLE_CATEGORIES) -> dict[str, list]:
    """GET /v2/cables — dict categoría → [{t, f}, ...]."""
    token = _token()
    if not token:
        return {}
    try:
        r = requests.get(
            f"{BASE_URL}/v2/cables",
            params={"token": token, "categorias": categorias},
            timeout=30,
        )
        if not r.ok:
            return {}
        data = r.json()
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("cables: %s", exc)
        return {}


def _raw_to_article(raw: dict[str, Any], ticker: str, aliases: list[str]) -> dict[str, Any]:
    title = (raw.get("n") or raw.get("t") or raw.get("titulo") or "").strip()
    title = re.sub(r"^\d{1,2}:\d{2}\s+", "", title)
    if not title:
        return {}
    summary = raw.get("c") or raw.get("resumen") or raw.get("summary") or ""
    url = raw.get("f") or raw.get("url") or ""
    if url and not url.startswith("http"):
        url = ""
    ts = _parse_databursatil_date(raw.get("fecha") or "", title)
    art = _article(
        title=title,
        summary=summary,
        url=url,
        published_ts=ts,
        source="databursatil",
    )
    return _with_relevance(art, ticker, aliases)


def fetch_databursatil_noticias_articles(
    ticker: str,
    aliases: list[str] | None = None,
    max_items: int = 30,
) -> list[dict[str, Any]]:
    """Noticias V2 filtradas por relevancia al ticker."""
    alias_list = aliases or get_entity_aliases(ticker)
    out: list[dict[str, Any]] = []
    for raw in fetch_noticias_raw():
        art = _raw_to_article(raw, ticker, alias_list)
        if art and art.get("title"):
            out.append(art)
    out.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    relevant = [a for a in out if a.get("relevance_score", 0) >= 0.35]
    return (relevant or out[:3])[:max_items]


def fetch_databursatil_cables_articles(
    ticker: str,
    aliases: list[str] | None = None,
    max_items: int = 30,
    categorias: str = DEFAULT_CABLE_CATEGORIES,
) -> list[dict[str, Any]]:
    """Cables V2 filtrados por relevancia al ticker."""
    alias_list = aliases or get_entity_aliases(ticker)
    out: list[dict[str, Any]] = []
    for _cat, items in fetch_cables_raw(categorias).items():
        for it in items if isinstance(items, list) else []:
            art = _raw_to_article(it, ticker, alias_list)
            if art and art.get("title"):
                out.append(art)
    out.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    relevant = [a for a in out if a.get("relevance_score", 0) >= 0.35]
    return (relevant or out[:3])[:max_items]


def fetch_databursatil_all_articles(
    ticker: str,
    aliases: list[str] | None = None,
    max_items: int = 30,
) -> list[dict[str, Any]]:
    """Noticias + cables, deduplicados por título."""
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for art in (
        fetch_databursatil_cables_articles(ticker, aliases, max_items=max_items)
        + fetch_databursatil_noticias_articles(ticker, aliases, max_items=max_items)
    ):
        key = art.get("url") or art.get("title", "")
        if key and key not in seen:
            seen.add(key)
            merged.append(art)
    merged.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return merged[:max_items]


def save_daily_snapshot(date: str | None = None, *, verbose: bool = True) -> Path:
    """
    Guarda snapshot del día (noticias + cables completos sin filtrar por ticker).
    Idempotente: sobreescribe el mismo date.
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    day = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cred = fetch_creditos()
    payload = {
        "date": day,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "creditos_disponibles": cred.get("disponibles"),
        "noticias": fetch_noticias_raw(),
        "cables": fetch_cables_raw(),
    }
    from src.data.news_medallion import ingest
    from src.ml.news_sources import ENTITY_ALIASES
    ingest("databursatil", payload, fetched_at=payload["fetched_at"],
           representation="api_payload", tickers=ENTITY_ALIASES)
    path = SNAPSHOT_DIR / f"{day}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    n_news = len(payload["noticias"])
    n_cables = sum(len(v) for v in payload["cables"].values() if isinstance(v, list))
    if verbose:
        print(f"  Snapshot {day}: {n_news} noticias, {n_cables} cables → {path}")
        if cred.get("disponibles") is not None:
            print(f"  Créditos restantes: {cred['disponibles']:,}")
    return path


def list_snapshot_dates() -> list[str]:
    if not SNAPSHOT_DIR.exists():
        return []
    return sorted(p.stem for p in SNAPSHOT_DIR.glob("*.json") if p.stem != "manifest")


def _articles_from_snapshot(snapshot: dict, ticker: str) -> list[Article]:
    aliases = get_entity_aliases(ticker)
    day = snapshot.get("date", "")
    end = datetime.strptime(day, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc,
    )
    ts = int(end.timestamp())
    arts: list[Article] = []
    seen: set[str] = set()

    def _add(raw: dict) -> None:
        art_d = _raw_to_article(raw, ticker, aliases)
        if not art_d or not art_d.get("title"):
            return
        key = art_d.get("url") or art_d["title"]
        if key in seen:
            return
        seen.add(key)
        arts.append(
            Article(
                ticker=ticker,
                title=art_d["title"],
                summary=art_d.get("summary") or "",
                url=art_d.get("url") or "",
                published_ts=ts,
                source="databursatil",
                sentiment=_score_keywords(f"{art_d['title']} {art_d.get('summary','')}"),
                relevance_score=float(art_d.get("relevance_score", 1.0)),
            )
        )

    for raw in snapshot.get("noticias") or []:
        _add(raw)
    for items in (snapshot.get("cables") or {}).values():
        for raw in items if isinstance(items, list) else []:
            _add(raw)
    return arts


def build_daily_panel_from_snapshots(
    tickers: list[str],
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Construye panel point-in-time desde snapshots acumulados.
    Una fila por (date, ticker) usando SOLO noticias de ese día.
    """
    dates = list_snapshot_dates()
    if not dates:
        if verbose:
            print("  Sin snapshots en data/news/history/databursatil/")
        return pd.DataFrame(columns=["date", "ticker", *NEWS_FEATURE_COLS])

    records: list[dict] = []
    for day in dates:
        try:
            snap = json.loads((SNAPSHOT_DIR / f"{day}.json").read_text())
        except Exception as exc:
            logger.warning("snapshot %s: %s", day, exc)
            continue
        as_of = pd.Timestamp(day).tz_localize("UTC")
        for ticker in tickers:
            arts = _articles_from_snapshot(snap, ticker)
            if not arts:
                records.append({
                    "date": as_of.normalize(),
                    "ticker": ticker,
                    **{c: 0.0 for c in NEWS_FEATURE_COLS},
                    "n_articles": 0,
                })
                continue
            feat = _compute_features_from_articles(arts, days_back=7, as_of=as_of)
            records.append({
                "date": as_of.normalize(),
                "ticker": ticker,
                **{c: feat.get(c, 0.0) for c in NEWS_FEATURE_COLS},
                "n_articles": feat.get("n_articles", 0),
            })
        if verbose:
            n_hit = sum(1 for t in tickers if _articles_from_snapshot(snap, t))
            print(f"  {day}: {n_hit}/{len(tickers)} tickers con match")

    df = pd.DataFrame(records)
    manifest = {
        "source": "databursatil_snapshots",
        "n_days": len(dates),
        "date_min": dates[0] if dates else None,
        "date_max": dates[-1] if dates else None,
        "tickers": len(tickers),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (SNAPSHOT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return df


def merge_into_daily_features_parquet(
    panel: pd.DataFrame,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Merge panel DB en daily_features.parquet (prioriza filas DB)."""
    from src.ml.news_history import DAILY_PATH, HISTORY_DIR

    out_path = out_path or DAILY_PATH
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    if panel.empty:
        return panel

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()
        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        key = ["date", "ticker"]
        existing = existing[
            ~existing.set_index(key).index.isin(panel.set_index(key).index)
        ]
        merged = pd.concat([existing, panel], ignore_index=True)
    else:
        merged = panel

    merged = merged.sort_values(["date", "ticker"]).drop_duplicates(
        subset=["date", "ticker"], keep="last",
    )
    merged.to_parquet(out_path, index=False)
    return merged
