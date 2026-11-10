"""Agente de dominio: Fundamentales (márgenes, UPA, dividendos — todo PIT).

Esquema real, 3 fuentes sin unificar hoy (ver docstrings de cada módulo):
  - data/fundamentals.parquet         src/api/fundamentals.py (márgenes, YoY)
  - data/dividends/dividends.json     src/ml/dividend_features.py
  - data/eps/reported_eps.json        scripts/scan_value_woe.py (yfinance)

Disciplina anti-look-ahead: cada fila fundamental tiene `available_date`
(fin de trimestre + 70 días). NUNCA respondas con un dato cuya
available_date sea posterior a la fecha de la pregunta (asof) — ver
src/agent/bmv_report.py:_fundamentals_asof como referencia de cómo filtrar.

Equipo Rojo: query_fundamentales responde primero con el producto Fundamentales
PIT (productos/fundamentales_pit, capa golden, regla available_date <= fecha).
Si la capa golden no existe en este entorno, cae al data/fundamentals.parquet
de siempre. Dividendos y UPA reportada siguen en sus archivos (pendiente unificar).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from claude_agent_sdk import tool

from src.agents_mcp.domain_agent import DomainAgent
from src.agent.fundamentales import AgenteFundamentales

FUNDAMENTALS_PARQUET = Path(__file__).parent.parent.parent.parent / "data" / "fundamentals.parquet"

SYSTEM_PROMPT = """Eres el agente de dominio de Fundamentales: márgenes,
UPA reportada y dividendos de emisoras BMV, todo point-in-time. Cada dato
tiene una available_date — nunca uses uno cuya available_date sea posterior
a la fecha de corte de la pregunta (asof). Si no tienes el dato PIT, dilo."""


@tool(
    "schema_fundamentales",
    "Describe las tablas, columnas y la disciplina PIT del dominio Fundamentales.",
    {},
)
async def schema_fundamentales(args: dict) -> dict:
    return {
        "content": [{
            "type": "text",
            "text": (
                "PRODUCTO (preferido): data/fundamentals/golden/fundamentales_pit.parquet "
                "(Equipo Rojo) — un trimestre por fila, fund_* + montos "
                "(fund_revenue_abs, fund_net_abs, fund_eps), available_date, "
                "origen y reexpresado; YoY calculado por fecha. "
                "data/fundamentals.parquet (respaldo) — ticker, quarter_end, "
                "fund_gross_margin, fund_oper_margin, fund_net_margin, "
                "fund_revenue_yoy, fund_earnings_yoy, available_date "
                "(= quarter_end + 70 días, src/api/fundamentals.py). "
                "data/dividends/dividends.json — {ticker: [{ex_date, "
                "pago_date, pago, tipo, divisa}]}. "
                "data/eps/reported_eps.json — {simbolo_yfinance: "
                "[{announced, reported, surprise}]}; usable desde "
                "announced + 1 día (src/ml/value_features.py)."
            ),
        }]
    }


@tool(
    "query_fundamentales",
    "Da los últimos trimestres (PIT) de un ticker BMV, ej. 'WALMEX': márgenes, "
    "YoY, montos y UPA con su available_date. `fecha_consulta` (YYYY-MM-DD, "
    "vacía = hoy) es la fecha de corte: nunca se devuelve nada publicado después.",
    {"ticker": str, "fecha_consulta": str},
)
async def query_fundamentales(args: dict) -> dict:
    ticker = str(args.get("ticker", "")).strip().upper()
    if not ticker:
        return {"content": [{"type": "text", "text": "Necesito un ticker (ej. 'WALMEX')."}]}

    # 0) Producto Fundamentales PIT (golden). `fecha_consulta` vacía = hoy; solo se
    # ven los trimestres con available_date <= fecha_consulta.
    fecha = str(args.get("fecha_consulta") or "").strip() or None
    r = AgenteFundamentales().observe(ticker=ticker, fecha_consulta=fecha, n_trimestres=4)
    if r.get("ok") or r.get("error_tipo") != "sin_datos":
        return {"content": [{"type": "text", "text": json.dumps(r, ensure_ascii=False, default=str)}]}
    # error_tipo == "sin_datos": no hay golden en este entorno -> respaldo de abajo.

    # 1) Intentar primero la caché local (data/fundamentals.parquet) — evita
    # depender de la API externa de DataBursátil y su token.
    try:
        if FUNDAMENTALS_PARQUET.exists():
            df = pd.read_parquet(FUNDAMENTALS_PARQUET)
            rows = df[df["ticker"].astype(str).str.upper() == ticker]
            if not rows.empty:
                r = rows.sort_values("available_date").iloc[-1]
                qend = pd.Timestamp(r["quarter_end"]).date().isoformat()
                avail = pd.Timestamp(r["available_date"]).date().isoformat()

                def fmt_pct(v):
                    return f"{float(v) * 100:+.2f}%" if pd.notna(v) else "N/D"

                texto = (
                    f"{ticker} — último trimestre reportado: {qend} "
                    f"(dato disponible desde {avail}, respetando el lag de "
                    f"publicación de ~70 días). "
                    f"Margen bruto: {fmt_pct(r['fund_gross_margin'])}, "
                    f"margen operativo: {fmt_pct(r['fund_oper_margin'])}, "
                    f"margen neto: {fmt_pct(r['fund_net_margin'])}, "
                    f"crecimiento de ingresos YoY: {fmt_pct(r['fund_revenue_yoy'])}, "
                    f"crecimiento de utilidad neta YoY: {fmt_pct(r['fund_earnings_yoy'])}. "
                    f"Nota: este es el dato fundamental MÁS RECIENTE disponible en "
                    f"caché local; si tu pregunta tiene una fecha de corte "
                    f"específica anterior a {avail}, este dato no aplica todavía."
                )
                return {"content": [{"type": "text", "text": texto}]}
    except Exception as e:
        # Caché corrupta o columnas inesperadas: cae a intentar la API en vivo.
        cache_error = str(e)
    else:
        cache_error = None

    # 2) Caché sin el ticker (o inexistente/corrupta): intentar la API en vivo.
    try:
        import os

        from src.api.fundamentals import EMISORA_MAP, all_periods, fetch_ticker

        if ticker not in EMISORA_MAP:
            return {
                "content": [{
                    "type": "text",
                    "text": f"No tengo datos fundamentales de {ticker!r} (no está en el catálogo BMV).",
                }]
            }

        token = os.environ.get("DATABURSATIL_API_TOKEN") or os.environ.get("DATABURSATIL_TOKEN")
        if not token:
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        f"No pude obtener fundamentales en vivo de {ticker}: falta "
                        "DATABURSATIL_API_TOKEN y no hay dato cacheado para este ticker. "
                        "No tengo ese dato disponible."
                    ),
                }]
            }

        feats = fetch_ticker(ticker, token)
        if feats is None or feats.empty:
            return {
                "content": [{
                    "type": "text",
                    "text": f"No pude obtener datos fundamentales de {ticker} desde la API en vivo.",
                }]
            }
        r = feats.sort_values("available_date").iloc[-1]
        qend = feats.index[-1].date().isoformat()
        avail = pd.Timestamp(r["available_date"]).date().isoformat()

        def fmt_pct(v):
            return f"{float(v) * 100:+.2f}%" if pd.notna(v) else "N/D"

        texto = (
            f"{ticker} — último trimestre reportado: {qend} (dato disponible "
            f"desde {avail}). Margen bruto: {fmt_pct(r['fund_gross_margin'])}, "
            f"margen operativo: {fmt_pct(r['fund_oper_margin'])}, margen neto: "
            f"{fmt_pct(r['fund_net_margin'])}, ingresos YoY: "
            f"{fmt_pct(r['fund_revenue_yoy'])}, utilidad neta YoY: "
            f"{fmt_pct(r['fund_earnings_yoy'])}. (obtenido en vivo de DataBursátil)"
        )
        return {"content": [{"type": "text", "text": texto}]}
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"No se pudo obtener el dato fundamental de {ticker} "
                    f"(caché local sin este ticker y la API en vivo falló: {e}). "
                    "No tengo ese dato disponible — no voy a inventarlo."
                ),
            }]
        }


fundamentales_agent = DomainAgent(
    name="fundamentales",
    system_prompt=SYSTEM_PROMPT,
    tools=[schema_fundamentales, query_fundamentales],
)
