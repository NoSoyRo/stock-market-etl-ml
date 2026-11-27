"""Agente de dominio: Mercado (precios OHLCV + FX) con texto a SQL.

El modelo escribe un SELECT guiado por la capa semántica
(semantic_layer_mercado.yaml) y consultar_mercado_sql lo valida y lo ejecuta
en la base gold (data/market/gold/mercado.db), que genera el ETL de src/etl.

Si la base no existe (por ejemplo, en una imagen que no la incluye), el agente
se queda como estaba: query_mercado calcula cierre, retorno y volatilidad
directo de los parquet de data/market.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
import statistics
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
from claude_agent_sdk import tool

from src.agents_mcp.domain_agent import DomainAgent
from src.data import store

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
SEMANTIC_LAYER = ROOT / "semantic_layer_mercado.yaml"
MERCADO_DB = ROOT / "data" / "market" / "gold" / "mercado.db"

MAX_FILAS = 50
TIMEOUT_S = 10
TABLAS_PERMITIDAS = {"precios_historicos", "tipo_cambio", "intradia_diario"}


def _resp(texto: str, error: bool = False) -> dict:
    r = {"content": [{"type": "text", "text": texto}]}
    if error:
        r["is_error"] = True
    return r


# ── Texto a SQL sobre la base gold ───────────────────────────────────────────

def _capa_semantica() -> str:
    if SEMANTIC_LAYER.exists():
        return SEMANTIC_LAYER.read_text(encoding="utf-8")
    return f"(falta la capa semántica: {SEMANTIC_LAYER.name})"


SYSTEM_PROMPT_SQL = f"""Eres el agente de dominio de Mercado: precios OHLCV diarios,
resumen intradía y tipo de cambio USD/MXN. Tu única fuente de verdad es la base
gold de Mercado, que consultas con consultar_mercado_sql. Si no tienes el dato,
dilo: nunca inventes precios, fechas ni columnas.

Cómo trabajar:
1. Escribe UN SELECT de SQLite con las tablas y columnas de la capa semántica de
   abajo, siguiendo sus reglas_de_negocio.
2. Llama consultar_mercado_sql. Si regresa error, corrige el SQL y reintenta
   (máximo 3 veces).
3. Responde con los números que regresó la consulta y di de qué fecha son.

Tickers: en precios_historicos la BMV lleva '.MX' (WALMEX -> 'WALMEX.MX'); en
intradia_diario va sin sufijo ('WALMEX'). Dólar = 'MXN=X', S&P 500 = 'GSPC',
petróleo WTI = 'CL=F'. Si dudas del nombre, búscalo primero con
SELECT DISTINCT ticker FROM precios_historicos WHERE ticker LIKE 'WAL%'.

Además de las funciones de SQLite tienes stdev(x), sqrt(x) y ln(x).
Ejemplo de "precio, retorno y volatilidad de 30 días" de WALMEX:
WITH u AS (SELECT fecha, close, ret_1d FROM precios_historicos
           WHERE ticker = 'WALMEX.MX' AND es_valido = 1
           ORDER BY fecha DESC LIMIT 31)
SELECT (SELECT fecha FROM u ORDER BY fecha DESC LIMIT 1) AS fecha,
       (SELECT close FROM u ORDER BY fecha DESC LIMIT 1) AS ultimo_cierre,
       (SELECT close FROM u ORDER BY fecha DESC LIMIT 1) * 100.0
         / (SELECT close FROM u ORDER BY fecha LIMIT 1) - 100 AS retorno_30d_pct,
       (SELECT stdev(ret_1d) FROM (SELECT ret_1d FROM u ORDER BY fecha DESC LIMIT 30))
         AS volatilidad_diaria_pct

Capa semántica (no hace falta llamar schema_mercado, ya la tienes aquí):
{_capa_semantica()}"""


class _Stdev:
    """Agregado stdev(x) para SQLite (desviación estándar muestral)."""

    def __init__(self) -> None:
        self.valores: list[float] = []

    def step(self, x) -> None:
        if x is not None:
            self.valores.append(float(x))

    def finalize(self):
        return statistics.stdev(self.valores) if len(self.valores) >= 2 else None


def _validar(sql: str) -> None:
    """Una sola sentencia SELECT (o UNION) que solo lea las tablas de Mercado."""
    import sqlglot
    from sqlglot import exp

    arboles = sqlglot.parse(sql, read="sqlite")
    if len(arboles) != 1 or arboles[0] is None:
        raise ValueError("Solo se permite una sentencia SQL.")
    arbol = arboles[0]
    if not isinstance(arbol, (exp.Select, exp.Union)):
        raise ValueError("Solo se permiten consultas SELECT.")
    ctes = {c.alias.lower() for c in arbol.find_all(exp.CTE)}
    for tabla in arbol.find_all(exp.Table):
        nombre = tabla.name.lower()
        if nombre in ctes and not tabla.db:
            continue  # referencia a una CTE, no a una tabla real
        if nombre not in TABLAS_PERMITIDAS:
            raise ValueError(f"Acceso denegado a la tabla: {tabla.name}")


def _ejecutar(sql: str) -> dict:
    """Corre el SELECT en solo lectura, con tiempo y filas máximos."""
    with closing(sqlite3.connect(MERCADO_DB.as_uri() + "?mode=ro", uri=True)) as conn:
        # Funciones propias: no todas las compilaciones de SQLite traen sqrt/ln y ninguna trae stdev.
        conn.create_aggregate("stdev", 1, _Stdev)
        conn.create_function("sqrt", 1, lambda x: math.sqrt(x) if x is not None and x >= 0 else None,
                             deterministic=True)
        conn.create_function("ln", 1, lambda x: math.log(x) if x is not None and x > 0 else None,
                             deterministic=True)
        limite = time.monotonic() + TIMEOUT_S
        conn.set_progress_handler(lambda: 1 if time.monotonic() > limite else 0, 100_000)
        df = pd.read_sql_query(f"SELECT * FROM (\n{sql}\n) LIMIT {MAX_FILAS + 1}", conn)
    datos = df.astype(object).where(df.notna(), None).to_dict(orient="records")
    return {
        "filas_devueltas": min(len(datos), MAX_FILAS),
        "truncado": len(datos) > MAX_FILAS,
        "datos": datos[:MAX_FILAS],
    }


@tool(
    "schema_mercado",
    "Capa semántica del dominio Mercado: tablas, columnas, reglas y ejemplos.",
    {},
)
async def schema_mercado(args: dict) -> dict:
    return _resp(_capa_semantica())


@tool(
    "consultar_mercado_sql",
    "Ejecuta UN SELECT de SQLite (puede llevar WITH) sobre la base gold de Mercado: "
    "precios_historicos, tipo_cambio e intradia_diario. Máximo 50 filas.",
    {"sql_query": str},
)
async def consultar_mercado_sql(args: dict) -> dict:
    sql = str(args.get("sql_query", "")).strip().rstrip(";")
    if not sql:
        return _resp("Falta la consulta SQL.", error=True)
    try:
        _validar(sql)
    except ImportError:
        return _resp("Error interno: falta instalar sqlglot.", error=True)
    except Exception as e:
        return _resp(f"Error de validación SQL: {e}", error=True)
    try:
        # En un hilo aparte: así una consulta lenta no congela al resto del servidor MCP.
        resultado = await asyncio.to_thread(_ejecutar, sql)
    except Exception as e:
        return _resp(f"Error SQL: {e}", error=True)
    return _resp(json.dumps(resultado, ensure_ascii=False, separators=(",", ":"), default=str))


# ── Respaldo sin base: el agente original, sobre los parquet ─────────────────

SYSTEM_PROMPT_PARQUET = """Eres el agente de dominio de Mercado: precios OHLCV
diarios e intradía, y tipo de cambio USD/MXN. Tu única fuente de verdad son
tus tools. Si no tienes el dato, dilo — nunca inventes un precio ni una
fecha."""


@tool(
    "schema_mercado",
    "Describe las tablas y columnas disponibles del dominio Mercado.",
    {},
)
async def schema_parquet(args: dict) -> dict:
    return _resp(
        "data/market/{SYMBOL}.parquet — Date(index), Open, High, Low, "
        "Close, Volume. Grupos (src/data/store.py): bmv, usa, etf, "
        "crypto, fibra, macro (macro incluye USDMXN='MXN=X', "
        "NAFTRAC, WTI='CL=F', SP500='^GSPC'). "
        "data/intraday/{ticker}.parquet — barras de 5 minutos, solo "
        "BMV líquidas (INTRADAY_MAP), histórico desde 2024-11-01."
    )


@tool(
    "query_mercado",
    "Da el último cierre, retorno % y volatilidad realizada de un ticker del "
    "store (nombre amigable, ej. 'WALMEX', 'SPY', 'USDMXN') sobre una ventana "
    "de días hábiles.",
    {"ticker": str, "dias": int},
)
async def query_mercado(args: dict) -> dict:
    ticker = str(args.get("ticker", "")).strip()
    try:
        dias = int(args.get("dias") or 30)
    except (TypeError, ValueError):
        dias = 30
    try:
        df = store.get(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            return _resp(f"No tengo datos de {ticker!r} en el store local.")
        closes = df["Close"].dropna()
        if closes.empty:
            return _resp(f"No tengo datos de cierre utilizables para {ticker!r}.")
        last_date = closes.index[-1]
        last_date_str = last_date.date().isoformat() if hasattr(last_date, "date") else str(last_date)
        partes = [f"{ticker}: último cierre {float(closes.iloc[-1]):,.4f} el {last_date_str}."]
        window = closes.tail(dias + 1)
        if len(window) >= 2:
            ret_pct = (window.iloc[-1] / window.iloc[0] - 1.0) * 100.0
            partes.append(f"Retorno de los últimos {len(window) - 1} días hábiles: {ret_pct:+.2f}%.")
            log_rets = np.log(window / window.shift(1)).dropna()
            if len(log_rets) >= 2:
                partes.append(
                    f"Volatilidad diaria realizada (std de log-retornos, {len(window) - 1} días): "
                    f"{float(log_rets.std()) * 100:.2f}% diario."
                )
        else:
            partes.append(f"No hay suficiente historial para calcular el retorno de {dias} días.")
        return _resp(" ".join(partes))
    except Exception as e:
        return _resp(f"No pude obtener datos de {ticker!r}: {e}")


# ── El agente ─────────────────────────────────────────────────────────────────

if MERCADO_DB.exists():
    mercado_agent = DomainAgent(
        name="mercado",
        system_prompt=SYSTEM_PROMPT_SQL,
        tools=[schema_mercado, consultar_mercado_sql],
    )
else:
    log.warning("No existe %s: el agente de Mercado usa query_mercado sobre los parquet. "
                "Corre el ETL de src/etl para activar el texto a SQL.", MERCADO_DB)
    mercado_agent = DomainAgent(
        name="mercado",
        system_prompt=SYSTEM_PROMPT_PARQUET,
        tools=[schema_parquet, query_mercado],
    )
