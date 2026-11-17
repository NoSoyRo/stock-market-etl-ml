"""Contrato real del SDK y consultas sobre Parquet, sin llamadas al LLM."""
import asyncio
from datetime import datetime, timezone
import json

import pandas as pd
import pytest
from jsonschema import validate

from src.agents_mcp.domains.noticias import create_news_agent, create_news_tools
from src.data.news_medallion import GOLD_COLUMNS, SILVER_COLUMNS
from src.semantic.service import NewsSemanticService

NOW = datetime(2026, 11, 6, 18, tzinfo=timezone.utc)


@pytest.fixture
def tools(tmp_path):
    def gold(ticker, day, batch, *, topic=''):
        return dict(date=pd.Timestamp(f'{day}T00:00Z'), ticker=ticker, topic=topic,
                    as_of=pd.Timestamp(f'{day}T17:00Z'), window_days=7, scoring_method='keywords',
                    publication_date_missing=0, article_ids=json.dumps([f'article-{batch}']),
                    batch_ids=json.dumps([batch]), news_sent_score=0.25, news_count_7d=0.69,
                    news_sent_delta=0.1, news_sent_vol=0.0, news_relevance_avg=1.0,
                    news_esg_score=0.0, n_articles=1, sources=['local'])

    rows = [gold('WALMEX', '2026-10-30', 'old'), gold('WALMEX', '2026-11-06', 'new'),
            gold('ALSEA', '2026-10-30', 'alsea'), gold('', '2026-11-06', 'topic', topic='trump')]
    silver = []
    for row in rows:
        batch = json.loads(row['batch_ids'])[0]
        silver.append(dict(
            article_id=f'article-{batch}', batch_id=batch, bronze_path=f'bronze/WALMEX/{batch}.json',
            source='local', ticker=row['ticker'], topic=row['topic'],
            title='Walmex y Walmart Mexico reportan crecimiento', text='WALMEX anuncia inversión',
            text_kind='excerpt', url='https://example.test/WALMEX',
            published_at=row['as_of'] - pd.Timedelta(hours=2),
            fetched_at=row['as_of'] - pd.Timedelta(hours=1),
            available_at=row['as_of'] - pd.Timedelta(hours=1), sentiment=None, relevance_score=1.0))
    for folder in ('gold', 'silver'):
        (tmp_path / folder).mkdir()
    pd.DataFrame(rows, columns=GOLD_COLUMNS).to_parquet(tmp_path / 'gold/historical_features.parquet')
    pd.DataFrame(silver, columns=SILVER_COLUMNS).to_parquet(tmp_path / 'silver/articles.parquet')
    svc = NewsSemanticService(tmp_path, clock=lambda: NOW)
    return {tool.name: tool for tool in create_news_tools(service=svc)}


def call(tools, name, **args):
    tool = tools[name]
    validate(args, tool.input_schema)
    reply = asyncio.run(tool.handler(args))
    assert reply['content'][0]['type'] == 'text'
    return json.loads(reply['content'][0]['text'])


def test_main_agent_contract_has_only_registered_domain_tools(tmp_path):
    agent = create_news_agent(tmp_path)
    options = agent.build_options()
    assert agent.name == 'noticias'
    assert options.tools == []
    assert options.allowed_tools == [f'mcp__dominio-noticias__{t.name}' for t in agent.tools]
    assert {'schema_noticias', 'query_noticias', 'evidencia_noticias'} <= {t.name for t in agent.tools}


def test_query_latest_is_entity_specific_and_reports_staleness(tools):
    latest = call(tools, 'query_noticias', ticker=' walmex ')
    assert latest['status'] == 'ok'
    assert latest['data']['date_utc'] == '2026-11-06'
    assert latest['data']['metrics']['sentimiento_general'] == 0.25
    old = call(tools, 'query_noticias', ticker='ALSEA')
    assert old['status'] == 'stale'
    assert old['data']['date_utc'] == '2026-10-30'


def test_query_cutoff_never_uses_later_calculation(tools):
    result = call(tools, 'query_noticias', ticker='WALMEX', corte='2026-11-06T12:00:00Z')
    assert result['data']['date_utc'] == '2026-10-30'
    missing = call(tools, 'query_noticias', ticker='WALMEX', fecha='2026-11-06',
                   corte='2026-11-06T12:00:00Z')
    assert missing['status'] == 'not_found'


def test_topic_and_summary_do_not_mix_associations(tools):
    topic = call(tools, 'query_noticias', topic='trump')
    assert topic['query']['entity_type'] == 'topic'
    summary = call(tools, 'query_noticias')
    assert summary['data']['total_tickers'] == 1
    assert summary['data']['results'][0]['query']['entity_id'] == 'WALMEX'


@pytest.mark.parametrize('args', [
    {'ticker': '../WALMEX'}, {'ticker': 'WALMEX', 'topic': 'trump'},
    {'ticker': 'WALMEX', 'fecha': '2026-11-07'},
    {'ticker': 'WALMEX', 'corte': '2026-11-06T19:00:00Z'},
    {'ticker': 'WALMEX', 'corte': '2026-11-06T12:00:00'},
    {'anonimizar': True},
])
def test_invalid_query_is_explicit(tools, args):
    assert call(tools, 'query_noticias', **args)['status'] == 'invalid_input'


def test_evidence_and_cutoff_use_semantic_integrity_checks(tools):
    result = call(tools, 'evidencia_noticias', entity_type='ticker', entity_id='WALMEX', fecha='2026-11-06')
    assert result['status'] == 'ok'
    assert result['data']['articles'][0]['batch_id'] == 'new'
    assert result['data']['articles'][0]['sentiment'] is None
    early = call(tools, 'evidencia_noticias', entity_type='ticker', entity_id='WALMEX',
                 fecha='2026-11-06', corte='2026-11-06T12:00:00Z')
    assert early['status'] == 'not_found'


def test_bias_mode_redacts_identifiers_aliases_and_links_in_tool(tools):
    for name, args in [
        ('query_noticias', {'ticker': 'WALMEX'}),
        ('evidencia_noticias', {'entity_type': 'ticker', 'entity_id': 'WALMEX', 'fecha': '2026-11-06'}),
    ]:
        result = call(tools, name, **args, anonimizar=True)
        serialized = json.dumps(result).lower()
        assert result['anonymized'] and result['entity_label'] == 'E1'
        assert 'walmex' not in serialized and 'walmart' not in serialized
        assert 'https://' not in serialized and 'bronze_path' not in serialized
        assert 'article_id' not in serialized and 'batch_id' not in serialized


def test_history_preserves_gaps_and_coverage_matches_data(tools):
    result = call(tools, 'historial_noticias', entity_type='ticker', entity_id='WALMEX',
                  inicio='2026-10-30', fin='2026-11-06')
    assert result['status'] == 'ok'
    assert [r['date_utc'] for r in result['data']['results']] == ['2026-10-30', '2026-11-06']
    assert call(tools, 'cobertura_noticias')['data']['entities']['tickers'] == ['ALSEA', 'WALMEX']


def test_missing_datasets_do_not_fall_back_to_unrelated_snapshots(tmp_path):
    registry = {t.name: t for t in create_news_tools(tmp_path)}
    (tmp_path / '2026-11-06.json').write_text('[{"ticker":"WALMEX","news_sent_score":0.9}]')
    result = call(registry, 'query_noticias', ticker='WALMEX')
    assert result['status'] == 'data_unavailable'
