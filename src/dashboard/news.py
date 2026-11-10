"""Flask news UI: semantic reads and a bounded, server-side MCP agent."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from flask import Blueprint, Flask, current_app, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from src.agent.news import NewsAgent
from src.semantic.models import json_safe
from src.semantic.repository import DEFAULT_NEWS_ROOT
from src.semantic.service import NewsSemanticService

log = logging.getLogger(__name__)
news = Blueprint('news', __name__, template_folder='templates', static_folder='static', static_url_path='/assets')


def _service():
    return current_app.extensions['news_service']


def _entity():
    kind = request.args.get('entity_type', 'ticker')
    if kind not in {'ticker', 'topic'}:
        raise ValueError('Selecciona empresa o tema.')
    entity_id = request.args.get('entity_id', 'WALMEX')
    return kind, entity_id


def _reply(result):
    codes = {'invalid_input': 400, 'data_unavailable': 503, 'data_integrity_error': 503,
             'agent_unavailable': 502, 'not_configured': 503, 'configuration_error': 503,
             'provider_timeout': 504, 'agent_timeout': 504,
             'provider_rate_limit': 429, 'provider_auth_error': 502,
             'provider_model_error': 502, 'provider_connection_error': 502, 'provider_error': 502}
    return jsonify(json_safe(result)), codes.get(result.get('status'), 200)


@news.errorhandler(Exception)
def handle_error(exc):
    if isinstance(exc, HTTPException):
        return _reply({'status': 'invalid_input', 'warnings': ['Solicitud inválida.']})[0], exc.code
    if isinstance(exc, ValueError):
        return _reply({'status': 'invalid_input', 'warnings': [str(exc)]})
    log.exception('News dashboard request failed')
    return jsonify({'status': 'data_unavailable', 'warnings': ['No se pudo completar la consulta.']}), 503


@news.get('/')
def index():
    return render_template('news.html')


@news.get('/api/coverage')
def coverage():
    return _reply(_service().consultar_cobertura())


@news.get('/api/snapshot')
def snapshot():
    kind, entity_id = _entity()
    fn = _service().consultar_empresa if kind == 'ticker' else _service().consultar_tema
    return _reply(fn(entity_id, fecha=request.args.get('fecha') or None))


@news.get('/api/history')
def history():
    kind, entity_id = _entity()
    return _reply(_service().consultar_historial(kind, entity_id,
        request.args.get('inicio', ''), request.args.get('fin', '')))


@news.get('/api/evidence')
def evidence():
    kind, entity_id = _entity()
    return _reply(_service().consultar_evidencia(kind, entity_id,
        request.args.get('fecha', ''), limit=10, offset=int(request.args.get('offset', '0'))))


@news.get('/api/agent')
def agent_status():
    return jsonify(current_app.extensions['news_agent'].observe())


@news.post('/api/chat')
def chat():
    # Reject browser cross-origin calls; this local UI and API share an origin.
    origin = request.headers.get('Origin')
    if origin and origin.rstrip('/') != request.host_url.rstrip('/'):
        return jsonify({'status': 'invalid_input', 'error': 'Origen no permitido.'}), 403
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError('Envía una pregunta válida.')
    question = payload.get('question', '')
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
        raise ValueError('La pregunta debe tener entre 1 y 2000 caracteres.')
    history = payload.get('history', [])
    if not isinstance(history, list) or len(history) > 8:
        raise ValueError('El historial es demasiado largo.')
    clean_history = []
    for message in history:
        if (not isinstance(message, dict) or message.get('role') not in {'user', 'assistant'}
                or not isinstance(message.get('content'), str) or len(message['content']) > 6000):
            raise ValueError('El historial contiene un mensaje inválido.')
        clean_history.append({'role': message['role'], 'content': message['content']})
    if sum(len(m['content']) for m in clean_history) > 16000:
        raise ValueError('Reduce el historial de la conversación.')
    context = payload.get('context') or {}
    if not isinstance(context, dict):
        raise ValueError('La selección del panel no es válida.')
    context = {k: str(context[k])[:80] for k in ('entity_type', 'entity_id', 'fecha') if k in context}
    guard = current_app.extensions['news_chat_lock']
    if not guard.acquire(blocking=False):
        return jsonify({'ok': False, 'status': 'busy', 'error': 'Hay otro análisis en curso. Intenta en unos segundos.'}), 429
    try:
        result = current_app.extensions['news_agent'].act(
            question=question.strip(), context=context, history=clean_history)
        return _reply(result)
    finally:
        guard.release()


def configure_news(app, *, news_root=DEFAULT_NEWS_ROOT, service=None, agent=None):
    app.config['MAX_CONTENT_LENGTH'] = 48 * 1024
    app.extensions['news_service'] = service or NewsSemanticService(news_root)
    app.extensions['news_agent'] = agent or NewsAgent(news_root)
    app.extensions['news_chat_lock'] = threading.Lock()


def create_app(*, news_root=DEFAULT_NEWS_ROOT, service=None, agent=None):
    app = Flask(__name__)
    configure_news(app, news_root=news_root, service=service, agent=agent)
    app.register_blueprint(news)
    return app
