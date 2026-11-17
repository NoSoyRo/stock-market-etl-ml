"""Comprueba el enlace del orquestador con Noticias sin llamadas al modelo."""
import asyncio
from unittest.mock import AsyncMock

from src.agents_mcp import orchestrator
from src.agents_mcp.domains.noticias import noticias_agent


def test_orchestrator_registers_the_integrated_news_domain():
    assert orchestrator.noticias_agent is noticias_agent
    assert {t.name for t in noticias_agent.tools} == {
        'schema_noticias', 'query_noticias', 'historial_noticias',
        'evidencia_noticias', 'cobertura_noticias',
    }
    options = orchestrator.build_options()
    assert 'mcp__orquestador__consultar_noticias' in options.allowed_tools
    assert options.tools == []


def test_orchestrator_forwards_question_and_domain_answer(monkeypatch):
    ask = AsyncMock(return_value='WALMEX: datos antiguos, corte 2026-10-31.')
    monkeypatch.setattr(noticias_agent, 'ask', ask)
    result = asyncio.run(orchestrator.consultar_noticias.handler({
        'pregunta': 'Consulta sentimiento y fuentes de WALMEX.',
    }))
    ask.assert_awaited_once_with('Consulta sentimiento y fuentes de WALMEX.')
    assert result == {'content': [{
        'type': 'text', 'text': 'WALMEX: datos antiguos, corte 2026-10-31.',
    }]}
