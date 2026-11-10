"""Configurable model adapter for the news agent; MCP tools stay provider-neutral."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import httpx

from src.agent.chat import chat_with_tools as anthropic_chat, DEFAULT_MODEL as ANTHROPIC_DEFAULT

GEMINI_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'
GEMINI_DEFAULT = 'gemini-3.8-flash'


class NewsProviderError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def find_provider_error(error):
    """Recover the model error wrapped by MCP/AnyIO context managers."""
    if isinstance(error, NewsProviderError):
        return error
    for child in getattr(error, 'exceptions', ()):
        result = find_provider_error(child)
        if result is not None:
            return result
    return None


@dataclass(frozen=True)
class NewsChatConfig:
    provider: str
    model: str
    key_env: str

    @property
    def available(self):
        return bool(os.getenv(self.key_env, '').strip())


def get_news_chat_config() -> NewsChatConfig:
    provider = os.getenv('NEWS_CHAT_PROVIDER', 'gemini').strip().lower()
    if provider == 'gemini':
        model = os.getenv('GEMINI_CHAT_MODEL') or GEMINI_DEFAULT
        key_env = 'GEMINI_API_KEY'
    elif provider == 'anthropic':
        model = os.getenv('ANTHROPIC_MODEL') or ANTHROPIC_DEFAULT
        key_env = 'ANTHROPIC_API_KEY'
    else:
        raise ValueError('NEWS_CHAT_PROVIDER debe ser gemini o anthropic.')
    return NewsChatConfig(provider, model.strip(), key_env)


def chat_with_news_tools(messages: list[dict], tools: list[dict], *, config=None) -> dict[str, Any]:
    config = config or get_news_chat_config()
    if not config.available:
        raise RuntimeError(f'Falta configurar {config.key_env} en el servidor.')
    if config.provider == 'anthropic':
        return anthropic_chat(messages, tools, model=config.model)
    if config.provider != 'gemini':
        raise ValueError('Proveedor no soportado.')
    # Google's compatibility endpoint accepts the common tool-call message format.
    # Preserve complete tool_calls (including provider metadata) across rounds.
    effort = os.getenv('GEMINI_CHAT_REASONING_EFFORT', 'low').strip().lower()
    if effort not in {'minimal', 'low', 'medium', 'high'}:
        raise NewsProviderError('configuration_error', 'El nivel de razonamiento de Gemini no es válido.')
    try:
        read_timeout = float(os.getenv('GEMINI_CHAT_TIMEOUT_SECONDS', '90'))
        if not 10 <= read_timeout <= 120:
            raise ValueError('out of range')
    except ValueError:
        raise NewsProviderError('configuration_error', 'El timeout de Gemini debe estar entre 10 y 120 segundos.') from None
    try:
        response = httpx.post(
            GEMINI_ENDPOINT,
            headers={'Authorization': f'Bearer {os.environ[config.key_env]}',
                     'Content-Type': 'application/json'},
            json={'model': config.model, 'messages': messages, 'tools': tools,
                  'tool_choice': 'auto', 'max_tokens': 2048, 'reasoning_effort': effort},
            timeout=httpx.Timeout(connect=10, read=read_timeout, write=15, pool=5),
        )
    except httpx.TimeoutException:
        raise NewsProviderError('provider_timeout',
            'Gemini no respondió dentro del tiempo de espera. La solicitud no se reintentó. '
            'Prueba más tarde o selecciona otro modelo disponible en tu cuenta.') from None
    except httpx.RequestError:
        raise NewsProviderError('provider_connection_error',
            'No se pudo conectar con Gemini. Revisa la conexión a internet del servidor.') from None
    if response.status_code >= 400:
        # Do not log credentials or a provider error body that might echo inputs.
        status = {401: 'provider_auth_error', 403: 'provider_auth_error',
                  404: 'provider_model_error', 429: 'provider_rate_limit'}.get(response.status_code, 'provider_error')
        guidance = {401: 'Verifica la clave configurada.', 403: 'Verifica permisos y acceso al servicio.',
                    404: 'Verifica GEMINI_CHAT_MODEL y su disponibilidad en tu cuenta.',
                    429: 'Revisa la cuota y los límites de tu proyecto; intenta más tarde.'}
        raise NewsProviderError(status, f'Gemini respondió HTTP {response.status_code}. '
                                + guidance.get(response.status_code, 'El proveedor no completó la consulta.'))
    body = response.json()
    if not isinstance(body, dict) or not body.get('choices'):
        raise RuntimeError('Gemini no devolvió una respuesta de chat válida.')
    return body
