"""Agente de dominio: Lab/Trading — decisiones, plan, veto y gemelo.

Esquema real, Postgres esquema `lab` (sql/03_lab_schema.sql):
  lab.decision     — una fila por fecha de rebalanceo: regla, pesos, veto,
                     gemelo (raw jsonb)
  lab.entry_plan   — una fila por posición del plan (ticker, USD, horizonte,
                     stop, precio de entrada)
  lab.outcome      — score del plan a 20 días hábiles
  lab.price_daily / lab.news_daily / lab.metric_daily — snapshots diarios

Sin Postgres, los planes caen a data/lab/entry_plans.jsonl.

Este agente NO necesita conocer otros dominios — si una pregunta mezcla
Lab con Noticias o Mercado, eso lo resuelve el orquestador llamando a los
agentes por separado y combinando respuestas (ver ejemplo end-to-end en
la presentación de arquitectura).

TODO equipo Trading Agent: implementar query_lab de verdad sobre Postgres
`lab` (o el jsonl de fallback).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from claude_agent_sdk import tool

from src.agents_mcp.domain_agent import DomainAgent

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
BOOK_PATH = ROOT / "data" / "agent" / "book.json"
PLANS_PATH = ROOT / "data" / "lab" / "entry_plans.jsonl"
OUTCOMES_PATH = ROOT / "data" / "lab" / "outcomes.jsonl"
LLM_SIM_PATHS = [
    ROOT / "data" / "eval" / "forecast_demo" / "llm_sim_3m.json",
    ROOT / "data" / "eval" / "forecast_demo" / "llm_sim_2m.json",
    ROOT / "data" / "eval" / "forecast_demo" / "llm_sim_1m.json",
]

SYSTEM_PROMPT = """Eres el agente de dominio de Lab/Trading: la decisión
real del libro (regla, pesos, veto, gemelo), su plan de entrada y su
calificación 20 días después. Explica siempre si una decisión vino de la
regla sola, o si el veto de Haiku cambió algo respecto al gemelo. No
inventes un outcome que todavía no se calificó."""


@tool(
    "schema_lab",
    "Describe las tablas de decisión/plan/outcome del dominio Lab/Trading.",
    {},
)
async def schema_lab(args: dict) -> dict:
    return {
        "content": [{
            "type": "text",
            "text": (
                "Postgres esquema lab (sql/03_lab_schema.sql): "
                "lab.decision (regla, pesos, veto, gemelo en raw jsonb), "
                "lab.entry_plan (ticker, usd, horizonte='20d', stop, "
                "precio de entrada), lab.outcome (score a 20 días hábiles). "
                "Fallback sin Postgres: data/lab/entry_plans.jsonl y "
                "data/lab/outcomes.jsonl. "
                "Job que escribe esto: scripts/gcp_daily_job.py "
                "(run_decide, run_score)."
            ),
        }]
    }


def _regla_vigente() -> str:
    """Regla núcleo-satélite vigente, leída de src/agent/core_satellite.py
    (valores reales, no copiados a mano, para que nunca queden desfasados)."""
    try:
        from src.agent.core_satellite import (
            CATASTROPHE_DD,
            CORE,
            DRIFT_BAND,
            ENTER_TOP,
            EXIT_TOP,
            SAT_MAX_NAMES,
            SAT_WEIGHT,
            TREND_DAYS,
        )
    except Exception as exc:  # import pesado (pandas/numpy/datos) puede fallar
        return f"No pude importar src/agent/core_satellite.py ({exc})."

    core_txt = ", ".join(f"{nombre} {peso*100:.0f}%" for nombre, (_, peso) in CORE.items())
    return (
        f"Núcleo {sum(p for _, p in CORE.values())*100:.0f}%: {core_txt}. "
        f"Satélite {SAT_WEIGHT*100:.0f}% en hasta {SAT_MAX_NAMES} emisoras BMV/FIBRAs "
        f"(entra en el top {ENTER_TOP*100:.0f}%, sale al caer debajo del top "
        f"{EXIT_TOP*100:.0f}%, reselección trimestral). Filtro de tendencia: SMA de "
        f"{TREND_DAYS} sesiones (~10 meses). Stop de catástrofe del satélite: "
        f"{CATASTROPHE_DD*100:.0f}% debajo del máximo desde la entrada. "
        f"Banda de deriva: {DRIFT_BAND*100:.0f} pp antes de rebalancear."
    )


def _estado_libro() -> str:
    """Estado actual del libro (data/agent/book.json), sin ejecutar nada pesado."""
    if not BOOK_PATH.exists():
        return (
            "No encontré data/agent/book.json (el libro todavía no tiene snapshot "
            "guardado, o corre solo en el job diario de GCP)."
        )
    try:
        data = json.loads(BOOK_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return f"data/agent/book.json existe pero no se pudo leer ({exc})."
    posiciones = data.get("positions", {})
    pos_txt = ", ".join(
        f"{t}: ${float(p.get('usd', 0)):.2f} USD" for t, p in posiciones.items()
    ) or "sin posiciones abiertas"
    return (
        f"Libro: efectivo ${float(data.get('cash_usd', 0)):.2f} USD de un "
        f"presupuesto de ${float(data.get('budget_usd', 0)):.2f} USD, abierto desde "
        f"{data.get('opened_at')}. Posiciones: {pos_txt}."
    )


def _plan_entrada_reciente(n: int = 5) -> str:
    """Últimas filas de data/lab/entry_plans.jsonl (fallback sin Postgres)."""
    if not PLANS_PATH.exists():
        return "No encontré data/lab/entry_plans.jsonl."
    lineas = [l for l in PLANS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lineas:
        return "data/lab/entry_plans.jsonl está vacío."
    filas = []
    for linea in lineas[-n:]:
        try:
            row = json.loads(linea)
        except json.JSONDecodeError:
            continue
        filas.append(
            f"{row.get('ticker')} ${row.get('usd')} USD, entra {row.get('enter_on')} "
            f"a {row.get('entry_px')}, stop {row.get('stop')}, horizonte "
            f"{row.get('horizon_days')}d (p_sube={row.get('p_sube')})"
        )
    return f"Últimas {len(filas)} filas del plan de entrada: " + "; ".join(filas)


_OUTCOME_SQL = """
SELECT p.ticker, p.usd, p.enter_on, o.scored_at, o.realized_return, o.hit_stop
FROM lab.outcome o
JOIN lab.entry_plan p ON p.id = o.entry_plan_id
ORDER BY o.scored_at DESC
LIMIT %s
"""


def _db_configurado() -> bool:
    return bool(os.environ.get("LAB_DATABASE_URL")
                or (os.environ.get("LAB_PGHOST") or os.environ.get("LAB_HOST")))


def _outcomes_postgres(n: int) -> list[dict] | None:
    """Últimos outcomes de lab.outcome. None si no hay Postgres configurado
    o la consulta falla (entonces se usa el jsonl local)."""
    if not _db_configurado():
        return None
    try:
        from src.db import lab_store

        with lab_store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_OUTCOME_SQL, (n,))
                cols = [c[0] for c in cur.description]
                filas = [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        logger.warning("no se pudo leer lab.outcome: %s", type(exc).__name__)
        return None
    out = []
    for f in filas:
        ret = f.get("realized_return")
        usd = f.get("usd")
        pnl = float(usd) * float(ret) if usd is not None and ret is not None else None
        out.append({
            "ticker": f.get("ticker"),
            "enter_on": str(f.get("enter_on")),
            "scored_on": str(f.get("scored_at"))[:10],
            "return_pct": float(ret) if ret is not None else None,
            "pnl_usd": pnl,
            "hit_stop": f.get("hit_stop"),
        })
    return out


def _outcomes_locales(n: int) -> list[dict]:
    if not OUTCOMES_PATH.exists():
        return []
    filas = []
    for linea in OUTCOMES_PATH.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        try:
            filas.append(json.loads(linea))
        except json.JSONDecodeError:
            continue
    return filas[-n:]


def _resultados_planes(n: int = 10) -> str:
    """Ganancia o pérdida de los planes reales ya calificados (lab.outcome,
    o data/lab/outcomes.jsonl sin Postgres)."""
    filas = _outcomes_postgres(n)
    fuente = "lab.outcome"
    if filas is None:
        filas = _outcomes_locales(n)
        fuente = "data/lab/outcomes.jsonl"
    if not filas:
        return (
            f"Todavía no hay planes calificados en {fuente}: un plan se "
            "califica 20 días hábiles después de su entrada."
        )
    partes = []
    total = 0.0
    for f in filas:
        ret = f.get("return_pct")
        pnl = f.get("pnl_usd")
        if isinstance(pnl, (int, float)):
            total += pnl
        ret_txt = f"{ret*100:+.2f}%" if isinstance(ret, (int, float)) else "s/d"
        pnl_txt = f"{pnl:+.2f} USD" if isinstance(pnl, (int, float)) else "s/d"
        stop_txt = ", tocó stop" if f.get("hit_stop") else ""
        partes.append(
            f"{f.get('ticker')} (entra {f.get('enter_on')}, calificado "
            f"{f.get('scored_on')}): {ret_txt}, {pnl_txt}{stop_txt}"
        )
    return (
        f"Resultados reales de los últimos {len(filas)} planes calificados ({fuente}): "
        + "; ".join(partes) + f". P&L total: {total:+.2f} USD."
    )


def _decision_reciente_llm() -> str | None:
    """Última decisión del agente LLM en las simulaciones
    data/eval/forecast_demo/llm_sim_*m.json (no es una decisión de producción)."""
    for path in LLM_SIM_PATHS:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        decisiones = data.get("decisions") or []
        if not decisiones:
            continue
        ultima = decisiones[-1]
        pesos = ", ".join(f"{t}: {w*100:.0f}%" for t, w in ultima.get("weights", {}).items())
        return (
            f"Decisión más reciente del agente LLM en la simulación {path.name} "
            f"(backtest simulado, no producción; asof {ultima.get('asof')}): pesos [{pesos}]. "
            f"Razonamiento: {ultima.get('motivo', '').strip()}"
        )
    return None


@tool(
    "query_lab",
    "Responde una pregunta en lenguaje natural sobre decisiones/plan/outcome: "
    "la regla núcleo-satélite vigente, el estado del libro, el plan de "
    "entrada, la ganancia o pérdida de los planes ya calificados y la última "
    "decisión simulada del agente LLM.",
    {"pregunta": str},
)
async def query_lab(args: dict) -> dict:
    partes = [_regla_vigente(), _estado_libro(), _plan_entrada_reciente(),
              _resultados_planes()]
    decision_llm = _decision_reciente_llm()
    if decision_llm:
        partes.append(decision_llm)
    return {"content": [{"type": "text", "text": "\n\n".join(partes)}]}


lab_agent = DomainAgent(
    name="lab",
    system_prompt=SYSTEM_PROMPT,
    tools=[schema_lab, query_lab],
)
