"""Adaptador de solo lectura entre el agente de dominio y el medallion."""
from __future__ import annotations

from datetime import datetime, timezone
import re

from src.semantic.models import SemanticError, normalize_identifier, parse_iso_date, response
from src.semantic.service import NewsSemanticService


class NewsDomainQueries:
    def __init__(self, service: NewsSemanticService):
        self.service = service

    def query(self, *, ticker=None, topic=None, fecha=None, corte=None, anonimizar=False):
        """Conserva ticker/fecha del agente de main; agrega temas y corte PIT."""
        try:
            if ticker and topic:
                raise SemanticError('invalid_input', 'Consulta ticker o topic, uno a la vez.')
            if anonimizar and not ticker:
                raise SemanticError('invalid_input', 'La anonimización requiere un ticker.')
            if ticker:
                ticker = normalize_identifier(ticker, 'ticker')
            if topic:
                topic = normalize_identifier(topic, 'topic')
            cutoff = self._cutoff(corte)
            if fecha:
                day = parse_iso_date(fecha)
                if day > self.service.clock().astimezone(timezone.utc).date():
                    raise SemanticError('invalid_input', 'No se admiten fechas futuras.')
            # La selección inicial también respeta el corte: nunca elegir una
            # captura posterior para luego responder con información futura.
            data = self.service.repository.read_history()
            data = data[data.as_of <= cutoff]
            if ticker:
                data = data[(data.ticker == ticker) & (data.topic == '')]
            elif topic:
                data = data[(data.topic == topic) & (data.ticker == '')]
            else:
                data = data[(data.ticker != '') & (data.topic == '')]
            if fecha:
                data = data[data.date.dt.date == day]
            elif not data.empty:
                data = data[data.date == data.date.max()]
            if data.empty:
                result = response('not_found', {'ticker': ticker, 'topic': topic, 'fecha': fecha},
                                  warnings=['No hay resultados gold disponibles para esa consulta y corte.'])
            elif ticker or topic:
                selected_date = data.date.iloc[0].date().isoformat()
                method = self.service.consultar_empresa if ticker else self.service.consultar_tema
                result = method(ticker or topic, fecha=selected_date, corte=cutoff)
            else:
                tickers = sorted(data.ticker.unique())
                selected_date = data.date.iloc[0].date().isoformat()
                result = response('ok', {'mode': 'summary', 'fecha': selected_date}, data={
                    'total_tickers': len(tickers),
                    'results': [self.service.consultar_empresa(t, fecha=selected_date, corte=cutoff)
                                for t in tickers[:15]],
                    'has_more': len(tickers) > 15,
                }, provenance={'dataset': 'gold/historical_features.parquet'})
            return self._anonymous(result, ticker) if anonimizar else result
        except SemanticError as exc:
            # No incluir el identificador original en errores de anonimización.
            return response(exc.status, {'mode': 'query'}, warnings=[exc.message])

    def evidence(self, *, entity_type, entity_id, fecha, corte=None,
                 limit=10, offset=0, anonimizar=False):
        try:
            if anonimizar and entity_type != 'ticker':
                raise SemanticError('invalid_input', 'La anonimización requiere un ticker.')
            result = self.service.consultar_evidencia(
                entity_type, entity_id, fecha, corte=self._cutoff(corte), limit=limit, offset=offset)
            return self._anonymous(result, entity_id) if anonimizar else result
        except SemanticError as exc:
            return response(exc.status, {'mode': 'evidence'}, warnings=[exc.message])

    def _cutoff(self, value):
        now = self.service.clock().astimezone(timezone.utc)
        if value is None:
            return now
        try:
            cutoff = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
            if not isinstance(cutoff, datetime) or cutoff.tzinfo is None or cutoff.utcoffset() is None:
                raise ValueError
        except (TypeError, ValueError):
            raise SemanticError('invalid_input', 'corte debe ser ISO-8601 con zona horaria.') from None
        if cutoff > now:
            raise SemanticError('invalid_input', 'No se admiten cortes futuros.')
        return cutoff.astimezone(timezone.utc)

    @staticmethod
    def _anonymous(result, ticker):
        from src.ml.news_sources import get_entity_aliases

        ticker = normalize_identifier(ticker, 'ticker')
        aliases = sorted({ticker, *get_entity_aliases(ticker)}, key=len, reverse=True)
        # Mismo criterio de veto.py: sustituir alias sin distinguir mayúsculas.
        # Los tickers cortos usan límites para no modificar palabras ajenas.
        patterns = [re.compile(re.escape(a) if len(a) >= 3 else
                               r'(?<!\w)' + re.escape(a) + r'(?!\w)', re.I)
                    for a in aliases if a]
        omit = {'url', 'bronze_path', 'article_id', 'batch_id', 'article_ids', 'batch_ids',
                'missing_article_ids', 'missing_batch_ids', 'provenance'}

        def redact(value):
            if isinstance(value, dict):
                return {key: redact(item) for key, item in value.items() if key not in omit}
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, str):
                for pattern in patterns:
                    value = pattern.sub('E1', value)
            return value

        return {**redact(result), 'anonymized': True, 'entity_label': 'E1'}
