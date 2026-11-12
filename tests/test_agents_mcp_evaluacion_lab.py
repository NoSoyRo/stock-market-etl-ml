"""Pruebas de las herramientas de los agentes de dominio evaluacion y lab."""
from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("claude_agent_sdk")

from src.agents_mcp.domains import evaluacion, lab  # noqa: E402


def _texto(tool, args: dict) -> str:
    fn = getattr(tool, "handler", tool)
    res = asyncio.run(fn(args))
    return "\n".join(c["text"] for c in res["content"])


def _sim(final_equity: float) -> dict:
    return {
        "start": "2026-09-30",
        "end": "2026-10-30",
        "final_equity_usd": final_equity,
        "costos_usd": 0.2,
        "atribucion_usd": {"nucleo_local": 3.17, "nucleo_fx": -1.69, "costos": -0.2},
        "usd": {"libro": {"retorno_pct": 2.57, "sharpe": 2.584}, "SPY": {"retorno_pct": -0.91}},
        "mxn": {"libro": {"retorno_pct": 7.16}},
        "dsr": {"dsr": 0.161, "sharpe_anual": 2.584, "sr0_anual": 5.993, "n_obs": 22},
        "n_trials": 15,
    }


@pytest.fixture
def eval_data(tmp_path, monkeypatch):
    sim = tmp_path / "llm_sim_1m.json"
    sim.write_text(json.dumps(_sim(51.28)), encoding="utf-8")
    log = tmp_path / "trial_log.jsonl"
    filas = [
        {"family": "f24", "hash": "a", "config": {"policy": "mecanico", "months": 24},
         "result": {"usd": {"libro": {"retorno_pct": 21.52, "sharpe": 0.972}}}},
        {"family": "f24", "hash": "a", "config": {}, "result": {}},
        {"family": "f24", "hash": "b", "config": {}, "result": {}},
        {"family": "f24", "hash": "b", "config": {}, "result": {}},
    ]
    log.write_text("\n".join(json.dumps(f) for f in filas) + "\n", encoding="utf-8")
    monkeypatch.setattr(evaluacion, "LLM_SIM_FILES", {"1m": sim})
    monkeypatch.setattr(evaluacion, "TRIAL_LOG", log)


@pytest.mark.parametrize("tipo", ["ganancias", "pnl"])
def test_query_evaluacion_pnl_en_dinero(eval_data, tipo):
    texto = _texto(evaluacion.query_evaluacion, {"pregunta_tipo": tipo})
    assert "capital final $51.28 USD" in texto
    assert "+$1.28 USD" in texto
    assert "costos $0.20 USD" in texto
    assert "nucleo_fx: -1.69" in texto
    assert "MXN 7.16%" in texto


def test_query_evaluacion_resumen_sin_pnl_y_etiquetado(eval_data):
    texto = _texto(evaluacion.query_evaluacion, {"pregunta_tipo": "resumen"})
    assert "Simulaciones del agente LLM" in texto
    assert "no resultados de producción" in texto
    assert "Resultados reales" not in texto
    assert "capital final" not in texto


def test_query_evaluacion_cuenta_n_por_hashes(eval_data):
    texto = _texto(evaluacion.query_evaluacion, {"pregunta_tipo": "resumen"})
    assert "4 filas" in texto
    assert "f24: N=2 configuraciones distintas" in texto


def test_schema_evaluacion_sin_dsr_a_mano(eval_data):
    texto = _texto(evaluacion.schema_evaluacion, {})
    assert "0.32" not in texto
    assert "21.52% USD" in texto


def test_query_lab_lee_outcomes_locales(tmp_path, monkeypatch):
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text(
        json.dumps({"ticker": "GENTERA", "enter_on": "2026-10-12", "scored_on": "2026-11-09",
                    "return_pct": -0.035, "pnl_usd": -0.56, "hit_stop": True}) + "\n",
        encoding="utf-8",
    )
    for var in ("LAB_DATABASE_URL", "LAB_PGHOST", "LAB_HOST"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(lab, "OUTCOMES_PATH", outcomes)
    texto = lab._resultados_planes()
    assert "GENTERA" in texto
    assert "-3.50%" in texto
    assert "-0.56 USD" in texto
    assert "tocó stop" in texto


def test_query_lab_sin_outcomes(tmp_path, monkeypatch):
    for var in ("LAB_DATABASE_URL", "LAB_PGHOST", "LAB_HOST"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(lab, "OUTCOMES_PATH", tmp_path / "no_existe.jsonl")
    assert "Todavía no hay planes calificados" in lab._resultados_planes()


def test_query_lab_etiqueta_simulacion():
    texto = _texto(lab.query_lab, {"pregunta": "última decisión"})
    assert "Decisión real" not in texto
    assert "backtest simulado" in texto
