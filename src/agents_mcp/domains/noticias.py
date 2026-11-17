"""Dominio Noticias del orquestador de main, conectado al medallion local."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from claude_agent_sdk import tool

from src.agents_mcp.domain_agent import DomainAgent
from src.semantic.domain_queries import NewsDomainQueries
from src.semantic.models import json_safe
from src.semantic.repository import DEFAULT_NEWS_ROOT, GOLD_METRICS
from src.semantic.service import NewsSemanticService

SYSTEM_PROMPT = """Eres el agente de dominio de Noticias del sistema stock-market.
Consulta siempre tus tools: son tu única fuente de métricas y titulares.
Responde en español citando fecha de cálculo, ventana, método, fuentes y
advertencias. Sin fecha, query_noticias usa el último resultado disponible
para la entidad; puede ser antiguo: informa la fecha y el estado stale.
No interpretes ausencia de cobertura como sentimiento neutral. No mezcles
ventanas o métodos distintos ni sumes conteos de ventanas superpuestas.
Consulta evidencia antes de atribuir eventos o explicar un movimiento.
Para consultas históricas, respeta fecha y corte; nunca uses información futura.
Para evaluar sesgos del LLM o del portero/veto, pide anonimizar=true en las
tools de consulta Y evidencia, con un ticker a la vez; usa solo E1 en la
respuesta y no repitas el nombre o ticker que recibiste en la pregunta.
Las consultas son de solo lectura: no descargan ni recalculan noticias.
"""


def _reply(payload):
    return {'content': [{'type': 'text', 'text': json.dumps(
        json_safe(payload), ensure_ascii=False, allow_nan=False)}]}


def create_news_tools(news_root=DEFAULT_NEWS_ROOT, *, service=None):
    """Construye tools aisladas por raíz, útiles también sin invocar al LLM."""
    service = service or NewsSemanticService(Path(news_root))
    queries = NewsDomainQueries(service)

    @tool('schema_noticias', 'Describe los datasets, métricas y cortes del dominio Noticias.', {})
    async def schema_noticias(args):
        return _reply({
            'datasets': {'gold': 'gold/historical_features.parquet',
                         'silver': 'silver/articles.parquet'},
            'metrics': list(GOLD_METRICS),
            'queries': ['query_noticias', 'historial_noticias', 'evidencia_noticias', 'cobertura_noticias'],
            'rules': ['Solo lectura; sin scraping ni llamadas al modelo desde las consultas.',
                      'Sin fecha: último resultado disponible, nunca se presenta como dato de hoy.',
                      'Corte ISO-8601 con zona horaria; disponibilidad <= corte.',
                      'Ventanas superpuestas: no sumar conteos de artículos.',
                      'Sesgos: anonimizar=true oculta ticker, alias, URLs e identificadores de evidencia.',
                      'El golden PySpark sin ponderación no sustituye estas métricas ML.'],
        })

    @tool('query_noticias', 'Métricas por ticker o topic y fecha opcional; sin entidad resume hasta 15 tickers. '
          'Para evaluar sesgos usa anonimizar=true y un ticker.', {
              'type': 'object', 'properties': {
                  'ticker': {'type': 'string'}, 'topic': {'type': 'string'},
                  'fecha': {'type': 'string', 'description': 'YYYY-MM-DD; opcional.'},
                  'corte': {'type': 'string', 'description': 'Fecha/hora ISO-8601 con zona horaria.'},
                  'anonimizar': {'type': 'boolean', 'default': False}}, 'required': [],
              'additionalProperties': False})
    async def query_noticias(args):
        return _reply(await asyncio.to_thread(queries.query, **args))

    @tool('historial_noticias', 'Serie histórica por ticker o topic; preserva huecos y no suma ventanas.',
          {'entity_type': str, 'entity_id': str, 'inicio': str, 'fin': str})
    async def historial_noticias(args):
        return _reply(await asyncio.to_thread(service.consultar_historial, **args))

    @tool('evidencia_noticias', 'Titulares y procedencia de un resultado; excluye observaciones posteriores al corte. '
          'Para sesgos: anonimizar=true y entity_type=ticker.', {
              'type': 'object', 'properties': {
                  'entity_type': {'type': 'string', 'enum': ['ticker', 'topic']},
                  'entity_id': {'type': 'string'}, 'fecha': {'type': 'string'},
                  'corte': {'type': 'string'}, 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 100},
                  'offset': {'type': 'integer', 'minimum': 0},
                  'anonimizar': {'type': 'boolean', 'default': False}},
              'required': ['entity_type', 'entity_id', 'fecha'], 'additionalProperties': False})
    async def evidencia_noticias(args):
        return _reply(await asyncio.to_thread(queries.evidence, **args))

    @tool('cobertura_noticias', 'Entidades, fechas, fuentes y métodos realmente disponibles en el medallion.', {})
    async def cobertura_noticias(args):
        return _reply(await asyncio.to_thread(service.consultar_cobertura))

    return [schema_noticias, query_noticias, historial_noticias, evidencia_noticias, cobertura_noticias]


def create_news_agent(news_root=DEFAULT_NEWS_ROOT):
    return DomainAgent(name='noticias', system_prompt=SYSTEM_PROMPT, tools=create_news_tools(news_root))


# Mismo nombre e interfaz que importa src.agents_mcp.orchestrator en main.
noticias_agent = create_news_agent()
