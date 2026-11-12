"""Agente de dominio: Evaluación/Riesgo — backtests honestos, Deflated
Sharpe y bitácora anti-overfitting.

Esquema real:
  - data/eval/trial_log.jsonl       bitácora de pruebas (hash de config, N)
  - src/ml/trial_stats.py           Deflated Sharpe correcto (el de
                                     src/ml/purged_cv.py tiene la escala mal)
  - scripts/backtest_core_satellite.py  el backtest honesto vigente
                                     (--policy mecanico|veto|llm)

Regla de trabajo que este agente debe respetar al responder: no se ajustan
parámetros mirando el backtest. La evidencia que cuenta es el paper trading
hacia adelante (libro contra gemelo), no el backtest histórico — dilo
explícitamente si alguien pregunta "¿cuál es la mejor configuración?".

TODO equipo Evaluación/Riesgo: implementar query_evaluacion de verdad sobre
trial_log.jsonl y los data/eval/*.json existentes.
"""
from __future__ import annotations

import json
from pathlib import Path

from claude_agent_sdk import tool

from src.agents_mcp.domain_agent import DomainAgent
from src.agents_mcp.domains.evaluacion_consultas import (
    ARCHIVOS,
    consultar_resultado,
)

ROOT = Path(__file__).resolve().parents[3]
TRIAL_LOG = ROOT / "data" / "eval" / "trial_log.jsonl"
FORECAST_DEMO_DIR = ROOT / "data" / "eval" / "forecast_demo"
LLM_SIM_FILES = {
    "1m": FORECAST_DEMO_DIR / "llm_sim_1m.json",
    "2m": FORECAST_DEMO_DIR / "llm_sim_2m.json",
    "3m": FORECAST_DEMO_DIR / "llm_sim_3m.json",
}

BUDGET_USD = 50.0  # tamaño del libro de las simulaciones llm_sim_*
TIPOS_SIMS = ("resumen", "", "ultimo_backtest", "sims", "llm_sim")
TIPOS_PNL = ("ganancias", "pnl", "p&l", "perdidas", "pérdidas")

DISCIPLINA = (
    "Nunca se ajustan parámetros mirando el backtest histórico: la evidencia "
    "que cuenta es el paper trading hacia adelante (libro contra gemelo), no "
    "el backtest."
)

SYSTEM_PROMPT = """Eres el agente de dominio de Evaluación/Riesgo:
backtests, Deflated Sharpe, bitácora de pruebas. Nunca recomiendes ajustar
un parámetro basado solo en el backtest — esa disciplina está documentada
y es intencional (ver trial_stats.py). Si preguntan por una métrica,
siempre aclara sobre cuántas pruebas (N) se calculó el Deflated Sharpe."""


@tool(
    "schema_evaluacion",
    "Describe la bitácora de pruebas y el motor de backtest del dominio Evaluación.",
    {},
)
async def schema_evaluacion(args: dict) -> dict:
    return {
        "content": [{
            "type": "text",
            "text": (
                "data/eval/trial_log.jsonl — una fila por corrida de "
                "backtest, con hash de su configuración (sube N solo con "
                "configuraciones distintas, no con cambios de costo). "
                "Deflated Sharpe correcto: src/ml/trial_stats.py (el de "
                "purged_cv.py tiene la escala mal, no usarlo). "
                "Backtest vigente: scripts/backtest_core_satellite.py "
                "(--policy mecanico=regla sola, veto=regla+Haiku, "
                "llm=Haiku repartiendo libre, histórico). "
                "Simulaciones del agente LLM (no son resultados de producción): "
                "data/eval/forecast_demo/llm_sim_{1m,2m,3m}.json, con P&L en USD "
                "(final_equity_usd, costos_usd, atribucion_usd) y retornos en MXN."
                + _ultimo_backtest_txt()
            ),
        }]
    }


def _read_trials() -> list[dict]:
    if not TRIAL_LOG.exists():
        return []
    rows = []
    for line in TRIAL_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _count_trials() -> tuple[int, dict[str, int]]:
    """Cuenta filas de trial_log.jsonl y, por familia, configuraciones
    distintas (hashes). El N del Deflated Sharpe es el número de hashes
    distintos, igual que src/ml/trial_stats.n_trials."""
    rows = _read_trials()
    hashes: dict[str, set] = {}
    for row in rows:
        hashes.setdefault(row.get("family", "desconocida"), set()).add(row.get("hash"))
    return len(rows), {fam: len(h) for fam, h in hashes.items()}


def _ultimo_backtest_txt() -> str:
    """Último resultado registrado en la bitácora, leído del archivo (no a mano)."""
    rows = [r for r in _read_trials() if (r.get("result") or {}).get("usd")]
    if not rows:
        return ""
    row = rows[-1]
    cfg = row.get("config") or {}
    libro = row["result"]["usd"].get("libro", {})
    meses = cfg.get("months")
    periodo = f", {meses}m" if meses else ""
    return (
        f" Último backtest en la bitácora ({row.get('family')}{periodo}, "
        f"policy {cfg.get('policy')}): libro {libro.get('retorno_pct')}% USD, "
        f"Sharpe {libro.get('sharpe')}. La bitácora no guarda el Deflated "
        "Sharpe de esa corrida."
    )


def _resumen_sim(etiqueta: str, path: Path, con_pnl: bool = False) -> str | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    libro = data.get("usd", {}).get("libro", {})
    benchmarks = {k: v for k, v in data.get("usd", {}).items() if k != "libro"}
    bench_txt = "; ".join(
        f"{nombre}: {info.get('retorno_pct')}%" for nombre, info in benchmarks.items()
    )
    dsr = data.get("dsr", {})
    n_trials = data.get("n_trials")
    texto = (
        f"{etiqueta} ({data.get('start')} a {data.get('end')}): "
        f"retorno del libro {libro.get('retorno_pct')}% (Sharpe {libro.get('sharpe')}), "
        f"benchmarks [{bench_txt}], "
        f"Deflated Sharpe {dsr.get('dsr')} (Sharpe anual {dsr.get('sharpe_anual')} "
        f"vs. azar esperado {dsr.get('sr0_anual')} sobre N={n_trials} pruebas, "
        f"{dsr.get('n_obs')} observaciones diarias)."
    )
    if con_pnl:
        pnl = _pnl_sim(data)
        if pnl:
            texto += " " + pnl
    return texto


def _pnl_sim(data: dict) -> str:
    """Ganancia o pérdida en dinero de una simulación (libro de BUDGET_USD)."""
    partes = []
    final = data.get("final_equity_usd")
    if isinstance(final, (int, float)):
        pnl = final - BUDGET_USD
        signo = "+" if pnl >= 0 else "-"
        partes.append(
            f"P&L: capital final ${final:.2f} USD sobre ${BUDGET_USD:.0f} USD "
            f"({signo}${abs(pnl):.2f} USD)"
        )
    if isinstance(data.get("costos_usd"), (int, float)):
        partes.append(f"costos ${data['costos_usd']:.2f} USD")
    atrib = data.get("atribucion_usd") or {}
    if atrib:
        partes.append(
            "atribución USD [" + ", ".join(f"{k}: {v:+.2f}" for k, v in atrib.items()) + "]"
        )
    mxn = (data.get("mxn") or {}).get("libro", {})
    if mxn.get("retorno_pct") is not None:
        partes.append(f"retorno del libro en MXN {mxn['retorno_pct']}%")
    return "; ".join(partes) + "." if partes else ""


@tool(
    "query_evaluacion",
    "Responde una pregunta en lenguaje natural sobre backtests/evaluación: "
    "cuántas pruebas hay en la bitácora y cómo le fue al agente LLM en sus "
    "simulaciones de 30/60/90 días. pregunta_tipo: resumen, sims, "
    "ultimo_backtest, ganancias o pnl (estos dos agregan el P&L en dinero).",
    {"pregunta_tipo": str},
)
async def query_evaluacion(args: dict) -> dict:
    pregunta_tipo = (args.get("pregunta_tipo") or "resumen").strip().lower()

    total, por_familia = _count_trials()
    if total:
        detalle_familias = ", ".join(
            f"{fam}: N={n} configuraciones distintas" for fam, n in sorted(por_familia.items())
        )
        trial_txt = (
            f"data/eval/trial_log.jsonl tiene {total} filas registradas "
            f"({detalle_familias})."
        )
    else:
        trial_txt = "data/eval/trial_log.jsonl no existe o está vacío todavía."

    partes = [trial_txt]

    con_pnl = pregunta_tipo in TIPOS_PNL
    if pregunta_tipo in TIPOS_SIMS or con_pnl:
        resumenes = []
        for etiqueta, path in LLM_SIM_FILES.items():
            resumen = _resumen_sim(f"Simulación LLM {etiqueta}", path, con_pnl=con_pnl)
            if resumen:
                resumenes.append(resumen)
        if resumenes:
            partes.append(
                "Simulaciones del agente LLM (backtest simulado, no resultados "
                "de producción; forecast_demo):\n- "
                + "\n- ".join(resumenes)
            )
        else:
            partes.append(
                "No encontré los archivos llm_sim_1m/2m/3m.json en "
                "data/eval/forecast_demo/."
            )

    partes.append(DISCIPLINA)
    return {"content": [{"type": "text", "text": "\n\n".join(partes)}]}


@tool(
    "consultar_metricas_evaluacion",
    "Consulta métricas reales de evaluación, portafolio, backtest, "
    "experimentos, comparación y validación de estrategias.",
    {"tipo": str},
)
async def consultar_metricas_evaluacion(args: dict) -> dict:
    """Consulta resultados JSON del dominio Evaluación/Riesgo."""

    tipo = (args.get("tipo") or "").strip().lower()

    resultado = consultar_resultado(tipo)

    nombre_archivo = ARCHIVOS.get(tipo)

    if nombre_archivo:
        fuente = f"data/eval/{nombre_archivo}"
    else:
        fuente = "Consulta no reconocida"

    return {
        "content": [{
            "type": "text",
            "text": (
                f"Fuente: {fuente}\n"
                f"{resultado}\n\n"
                f"{DISCIPLINA}"
            ),
        }]
    }


evaluacion_agent = DomainAgent(
    name="evaluacion",
    system_prompt=SYSTEM_PROMPT,
    tools=[
    schema_evaluacion,
    query_evaluacion,
    consultar_metricas_evaluacion,
],
)
