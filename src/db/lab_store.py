"""Escritura al esquema lab (Postgres 16, base mercado).

Dependencia de ejecución: psycopg v3 (requirements.txt).

El import de psycopg ocurre dentro de connect(), al usarse. Importar este
módulo no carga el driver. Las sentencias son SQL plano (sin ORM) y solo
tocan el esquema lab.

Conexión, en este orden:
  1. LAB_DATABASE_URL
  2. LAB_PGHOST / LAB_PGUSER / LAB_PGPASSWORD / LAB_PGDATABASE
     (alias: LAB_HOST, LAB_USER, LAB_PASSWORD, LAB_DB; puerto LAB_PGPORT)
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

# psycopg no se importa aquí a propósito: si falta el driver, falla al llamar
# connect(), no al importar este módulo ni al leer sql/03_lab_schema.sql.

_PRICE_SQL = """
INSERT INTO lab.price_daily (
    ticker, trade_date, open, high, low, close, volume, moneda, source, loaded_at
) VALUES (
    %(ticker)s, %(trade_date)s, %(open)s, %(high)s, %(low)s, %(close)s,
    %(volume)s, %(moneda)s, %(source)s, COALESCE(%(loaded_at)s, now())
)
ON CONFLICT (ticker, trade_date) DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume,
    moneda = EXCLUDED.moneda,
    source = EXCLUDED.source,
    loaded_at = EXCLUDED.loaded_at
"""

_NEWS_SQL = """
INSERT INTO lab.news_daily (ticker, trade_date, title, source, sentiment, url)
VALUES (%(ticker)s, %(trade_date)s, %(title)s, %(source)s, %(sentiment)s, %(url)s)
"""

_NEWS_FEATURE_SQL = """
INSERT INTO lab.news_feature_daily (
    ticker, trade_date, universe, news_sent_score, news_count_7d,
    news_sent_delta, news_sent_vol, news_relevance_avg, news_esg_score,
    n_articles, sources, loaded_at
) VALUES (
    %(ticker)s, %(trade_date)s, %(universe)s, %(news_sent_score)s, %(news_count_7d)s,
    %(news_sent_delta)s, %(news_sent_vol)s, %(news_relevance_avg)s, %(news_esg_score)s,
    %(n_articles)s, %(sources)s, COALESCE(%(loaded_at)s, now())
)
ON CONFLICT (ticker, trade_date) DO UPDATE SET
    universe = EXCLUDED.universe,
    news_sent_score = EXCLUDED.news_sent_score,
    news_count_7d = EXCLUDED.news_count_7d,
    news_sent_delta = EXCLUDED.news_sent_delta,
    news_sent_vol = EXCLUDED.news_sent_vol,
    news_relevance_avg = EXCLUDED.news_relevance_avg,
    news_esg_score = EXCLUDED.news_esg_score,
    n_articles = EXCLUDED.n_articles,
    sources = EXCLUDED.sources,
    loaded_at = EXCLUDED.loaded_at
"""

_DECISION_SQL = """
INSERT INTO lab.decision (decided_at, session_date, model, cash_usd, motivo, raw)
VALUES (
    COALESCE(%(decided_at)s, now()),
    %(session_date)s,
    %(model)s,
    %(cash_usd)s,
    %(motivo)s,
    %(raw)s::jsonb
)
RETURNING id
"""

_PLAN_SQL = """
INSERT INTO lab.entry_plan (
    decision_id, ticker, usd, horizon, stop_usd, p_sube, enter_on
) VALUES (
    %(decision_id)s, %(ticker)s, %(usd)s, %(horizon)s,
    %(stop_usd)s, %(p_sube)s, %(enter_on)s
)
"""

_PRICE_FIELDS = (
    "ticker",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "moneda",
    "source",
    "loaded_at",
)

_NEWS_FIELDS = ("ticker", "trade_date", "title", "source", "sentiment", "url")

_NEWS_FEATURE_FIELDS = (
    "ticker",
    "trade_date",
    "universe",
    "news_sent_score",
    "news_count_7d",
    "news_sent_delta",
    "news_sent_vol",
    "news_relevance_avg",
    "news_esg_score",
    "n_articles",
    "sources",
    "loaded_at",
)

_PLAN_FIELDS = ("ticker", "usd", "horizon", "stop_usd", "p_sube", "enter_on")


def _load_psycopg() -> Any:
    """Importa psycopg v3. El fallo queda en el uso, no en el import del módulo."""
    try:
        import psycopg
    except ImportError as exc:
        raise ImportError(
            "Falta el driver psycopg (v3). Instala psycopg[binary]; "
            "no está en requirements.txt. Las funciones ejecutan SQL plano."
        ) from exc
    return psycopg


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def connect() -> Any:
    """Abre una conexión psycopg a la base mercado.

    Usa LAB_DATABASE_URL si existe. Si no, arma la conexión con
    LAB_PGHOST, LAB_PGUSER, LAB_PGPASSWORD y LAB_PGDATABASE
    (alias LAB_HOST, LAB_USER, LAB_PASSWORD, LAB_DB).

    El caller es dueño de la conexión: commit, rollback y close.
    Usada como context manager, psycopg confirma al salir sin error.
    """
    psycopg = _load_psycopg()
    url = _env("LAB_DATABASE_URL")
    if url:
        return psycopg.connect(url)

    host = _env("LAB_PGHOST", "LAB_HOST")
    user = _env("LAB_PGUSER", "LAB_USER")
    if not host or not user:
        raise RuntimeError(
            "Define LAB_DATABASE_URL, o LAB_PGHOST y LAB_PGUSER "
            "(password LAB_PGPASSWORD, base LAB_PGDATABASE)."
        )
    return psycopg.connect(
        host=host,
        port=_env("LAB_PGPORT", "LAB_PORT") or "5432",
        user=user,
        password=os.environ.get("LAB_PGPASSWORD", os.environ.get("LAB_PASSWORD", "")),
        dbname=_env("LAB_PGDATABASE", "LAB_DB", "LAB_PGDB") or "mercado",
    )


def _run(conn: Any | None, fn: Any) -> Any:
    if conn is not None:
        return fn(conn)
    with connect() as owned:
        return fn(owned)


def _rows(rows: Iterable[Mapping[str, Any]] | Any) -> list[Mapping[str, Any]]:
    if rows is None:
        return []
    if hasattr(rows, "to_dict"):
        return list(rows.to_dict(orient="records"))
    if isinstance(rows, Mapping):
        return [rows]
    return list(rows)


def _as_date(value: Any) -> Any:
    if value is None or isinstance(value, datetime):
        return value.date() if isinstance(value, datetime) else value
    if isinstance(value, date):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date()
    return value


def _require(row: Mapping[str, Any], *keys: str, what: str) -> None:
    missing = [key for key in keys if row.get(key) in (None, "")]
    if missing:
        raise ValueError(f"{what} exige {', '.join(missing)}")


def _jsonb(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


def upsert_prices(
    rows: Iterable[Mapping[str, Any]],
    *,
    conn: Any | None = None,
) -> int:
    """Inserta o actualiza barras en lab.price_daily.

    Cada fila usa las columnas de la tabla. ticker y trade_date son obligatorios.
    Una fila omitida en open/high/low/close/volume se guarda como NULL y pisa
    el valor previo. Devuelve cuántas filas se enviaron.
    """
    payload = []
    for row in _rows(rows):
        item = {key: row.get(key) for key in _PRICE_FIELDS}
        _require(item, "ticker", "trade_date", what="upsert_prices")
        item["ticker"] = str(item["ticker"])
        item["trade_date"] = _as_date(item["trade_date"])
        payload.append(item)
    if not payload:
        return 0

    def _write(active: Any) -> int:
        with active.cursor() as cur:
            cur.executemany(_PRICE_SQL, payload)
        return len(payload)

    return _run(conn, _write)


def insert_news(
    rows: Iterable[Mapping[str, Any]],
    *,
    conn: Any | None = None,
) -> int:
    """Inserta noticias en lab.news_daily.

    No deduplica: puede haber varias por ticker y día, y un reintento
    vuelve a insertar. ticker y trade_date son obligatorios.
    Devuelve cuántas filas se enviaron.
    """
    payload = []
    for row in _rows(rows):
        item = {key: row.get(key) for key in _NEWS_FIELDS}
        _require(item, "ticker", "trade_date", what="insert_news")
        item["ticker"] = str(item["ticker"])
        item["trade_date"] = _as_date(item["trade_date"])
        payload.append(item)
    if not payload:
        return 0

    def _write(active: Any) -> int:
        with active.cursor() as cur:
            cur.executemany(_NEWS_SQL, payload)
        return len(payload)

    return _run(conn, _write)


def upsert_news_features(
    rows: Iterable[Mapping[str, Any]],
    *,
    conn: Any | None = None,
) -> int:
    """Inserta o actualiza features de noticias en lab.news_feature_daily.

    Cada fila usa las columnas de la tabla. ticker y trade_date son obligatorios.
    La operación es idempotente sobre la clave primaria (ticker, trade_date).
    Devuelve cuántas filas se enviaron.
    """
    payload = []
    for row in _rows(rows):
        item = {key: row.get(key) for key in _NEWS_FEATURE_FIELDS}
        _require(item, "ticker", "trade_date", what="upsert_news_features")
        item["ticker"] = str(item["ticker"])
        item["trade_date"] = _as_date(item["trade_date"])
        if item.get("universe") is not None:
            item["universe"] = str(item["universe"])
        if item.get("sources") is not None:
            if isinstance(item["sources"], (list, tuple, set)):
                item["sources"] = ",".join(str(s) for s in sorted(item["sources"]))
            else:
                item["sources"] = str(item["sources"])
        if item.get("n_articles") is not None:
            item["n_articles"] = int(item["n_articles"])
        for float_col in (
            "news_sent_score",
            "news_count_7d",
            "news_sent_delta",
            "news_sent_vol",
            "news_relevance_avg",
            "news_esg_score",
        ):
            if item.get(float_col) is not None:
                item[float_col] = float(item[float_col])
        payload.append(item)
    if not payload:
        return 0

    def _write(active: Any) -> int:
        with active.cursor() as cur:
            cur.executemany(_NEWS_FEATURE_SQL, payload)
        return len(payload)

    return _run(conn, _write)


def insert_decision(
    *,
    session_date: date | datetime | str,
    model: str | None = None,
    cash_usd: Any = None,
    motivo: str | None = None,
    raw: Mapping[str, Any] | str | None = None,
    decided_at: datetime | None = None,
    plans: Iterable[Mapping[str, Any]] | None = None,
    conn: Any | None = None,
) -> int:
    """Inserta lab.decision y, si vienen, sus lab.entry_plan.

    plans es una secuencia de dicts con ticker y, opcionales, usd, horizon,
    stop_usd, p_sube, enter_on. Todo queda en la misma transacción.
    Devuelve el id de la decisión.
    """
    if session_date in (None, ""):
        raise ValueError("insert_decision exige session_date")

    decision = {
        "decided_at": decided_at,
        "session_date": _as_date(session_date),
        "model": model,
        "cash_usd": cash_usd,
        "motivo": motivo,
        "raw": _jsonb(raw),
    }
    plan_rows = []
    for row in _rows(plans):
        item = {key: row.get(key) for key in _PLAN_FIELDS}
        _require(item, "ticker", what="insert_decision plans")
        item["ticker"] = str(item["ticker"])
        item["enter_on"] = _as_date(item["enter_on"])
        plan_rows.append(item)

    def _write(active: Any) -> int:
        with active.cursor() as cur:
            cur.execute(_DECISION_SQL, decision)
            decision_id = int(cur.fetchone()[0])
            if plan_rows:
                for item in plan_rows:
                    item["decision_id"] = decision_id
                cur.executemany(_PLAN_SQL, plan_rows)
        return decision_id

    return _run(conn, _write)
