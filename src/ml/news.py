"""
Módulo de noticias financieras — scraping + NLP sentiment + features.

Fuentes BMV (prioridad):
  1. DataBursátil   — cables BMV en español (DATABURSATIL_API_TOKEN)
  2. Yahoo Finance  — yfinance .news
  3. Google News RSS
  4. El Financiero / Expansión RSS
  5. Alpha Vantage  — si ALPHA_VANTAGE_API_KEY (sentimiento incluido)
  6. GDELT          — opcional backfill histórico

Sentiment:
  - Alpha Vantage overall_sentiment_score cuando disponible
  - Fallback: keywords ES/EN
  - Gemini 2.5-flash ON por defecto si GEMINI_API_KEY (opt-out: GEMINI_NEWS_ENABLED=false)

Features (NEWS_FEATURE_COLS):
  news_sent_score, news_count_7d, news_sent_delta, news_sent_vol,
  news_relevance_avg, news_esg_score
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.ml.gemini_config import content_config, create_client, get_model
from src.ml.news_sources import (
    fetch_alpha_vantage_articles,
    fetch_cryptopanic as _fetch_cryptopanic_source,
    fetch_databursatil_articles,
    fetch_elfinanciero,
    fetch_expansion,
    fetch_gdelt as _fetch_gdelt_source,
    fetch_google_rss,
    fetch_yfinance_articles,
    get_entity_aliases,
    is_esg_headline,
)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# KB-004 — features de noticias (features.py importará esta lista después)
NEWS_FEATURE_COLS = [
    "news_sent_score",
    "news_count_7d",
    "news_sent_delta",
    "news_sent_vol",
    "news_relevance_avg",
    "news_esg_score",
]

LOW_RELEVANCE_THRESHOLD = 0.4
LOW_RELEVANCE_WEIGHT    = 0.3

ROOT     = Path(__file__).parent.parent.parent
NEWS_DIR = ROOT / "data" / "news"
NEWS_DIR.mkdir(parents=True, exist_ok=True)

# ── Timeouts y límites ────────────────────────────────────────────────────────
REQUEST_TIMEOUT  = 10   # segundos por request HTTP
MAX_ARTICLES     = 30   # máx artículos por fuente por ticker
GEMINI_BATCH     = 20   # artículos por llamada a Gemini
RATE_LIMIT_SLEEP = 0.3  # segundos entre requests RSS

# KB-041 / KB-043 — deduplicación y decay temporal
DEDUP_COSINE_THRESHOLD = 0.5   # similitud TF-IDF para marcar duplicado
DEDUP_DECAY_BASE       = 0.5   # peso *= 0.5^n_dup por titular similar
TEMPORAL_DECAY_LAMBDA  = 0.3   # weight = exp(-lambda * days_ago)

# ── Keywords para scoring de fallback ────────────────────────────────────────
_POS_KEYWORDS = {
    # Español
    "subida","alza","sube","gana","ganancias","beneficio","crecimiento","record",
    "máximo","supera","mejora","positivo","dividend","dividendo","compra","acumula",
    "expansión","récord","fortalece","impulsa","optimista","aumento","avanza",
    # Inglés
    "surge","rises","gains","profit","record","beats","growth","buy","upgrade",
    "positive","strong","rally","outperform","bullish","increase","boost",
}
_NEG_KEYWORDS = {
    # Español
    "caída","baja","cae","pérdida","pierde","quiebra","fraude","vende","retrocede",
    "riesgo","incertidumbre","cierre","multa","demanda","recorte","negativo","alerta",
    "preocupa","debilita","contracción","crisis","presión","decepción",
    # Inglés
    "falls","drops","loses","loss","decline","risk","fraud","sell","downgrade",
    "negative","weak","bearish","cut","miss","concern","warning","lawsuit","bankruptcy",
}


# ── Dataclass de artículo ─────────────────────────────────────────────────────

@dataclass
class Article:
    ticker:           str
    title:            str
    summary:          str
    url:              str
    published_ts:     int    # Unix timestamp UTC
    source:           str
    sentiment:        float = 0.0   # −1 a +1, se rellena después
    weight:           float = 1.0   # peso compuesto (dedup × decay × relevance)
    relevance_score:  float = 1.0
    av_sentiment:     Optional[float] = None

    @property
    def published_dt(self) -> datetime:
        return datetime.fromtimestamp(self.published_ts, tz=timezone.utc)

    def is_within(self, days: int) -> bool:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
        return self.published_dt >= cutoff

    def text(self) -> str:
        return f"{self.title}. {self.summary}".strip()


# ─────────────────────────────────────────────────────────────────────────────
# SCRAPERS (delegates to news_sources.py)
# ─────────────────────────────────────────────────────────────────────────────

def _dict_to_article(raw: dict, ticker: str) -> Article:
    return Article(
        ticker=ticker,
        title=raw.get("title", ""),
        summary=raw.get("summary", "") or "",
        url=raw.get("url", ""),
        published_ts=int(raw.get("published_ts") or 0),
        source=raw.get("source", "unknown"),
        relevance_score=float(raw.get("relevance_score", 1.0)),
        av_sentiment=raw.get("av_sentiment"),
    )


def fetch_yfinance(yf_symbol: str, ticker: str) -> list[Article]:
    """Descarga noticias via yfinance .news (últimas ~30 noticias)."""
    return [
        _dict_to_article(d, ticker)
        for d in fetch_yfinance_articles(yf_symbol, ticker, max_items=MAX_ARTICLES)
    ]


def fetch_google_news_rss(query: str, ticker: str, lang: str = "es-MX") -> list[Article]:
    """Busca en Google News RSS."""
    return [
        _dict_to_article(d, ticker)
        for d in fetch_google_rss(query, ticker, max_items=MAX_ARTICLES, lang=lang)
    ]


def fetch_elfinanciero_rss(ticker: str, keyword: str) -> list[Article]:
    """Parsea El Financiero RSS filtrando por keyword."""
    return [
        _dict_to_article(d, ticker)
        for d in fetch_elfinanciero(ticker, keyword, max_items=MAX_ARTICLES)
    ]


def fetch_expansion_rss(ticker: str, keyword: str) -> list[Article]:
    """Parsea Expansión RSS filtrando por keyword."""
    return [
        _dict_to_article(d, ticker)
        for d in fetch_expansion(ticker, keyword, max_items=MAX_ARTICLES)
    ]


def fetch_gdelt(query: str, ticker: str, days_back: int = 7) -> list[Article]:
    """GDELT Project — historial para backfill."""
    return [
        _dict_to_article(d, ticker)
        for d in _fetch_gdelt_source(query, ticker, days_back=days_back, max_items=MAX_ARTICLES)
    ]


def fetch_cryptopanic(ticker: str, currency: str = "BTC") -> list[Article]:
    """CryptoPanic API — noticias cripto (requiere CRYPTOPANIC_API_KEY)."""
    del currency  # currency resolved inside news_sources
    return [
        _dict_to_article(d, ticker)
        for d in _fetch_cryptopanic_source(ticker, max_items=MAX_ARTICLES)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

def _score_keywords(text: str) -> float:
    """
    Scoring simple por keywords.
    Retorna float en [−1, +1].
    """
    tokens = set(text.lower().split())
    pos = len(tokens & _POS_KEYWORDS)
    neg = len(tokens & _NEG_KEYWORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


def _gemini_news_enabled() -> bool:
    """ON by default when GEMINI_API_KEY is set unless GEMINI_NEWS_ENABLED=false."""
    if os.environ.get("GEMINI_NEWS_ENABLED", "").lower() == "false":
        return False
    return bool(os.environ.get("GEMINI_API_KEY"))


def score_with_gemini(articles: list[Article], company_name: str) -> list[Article]:
    """
    Asigna sentimiento a cada artículo.
    Alpha Vantage sentiment tiene prioridad; fallback a keywords.
    Gemini se activa por defecto si hay GEMINI_API_KEY (opt-out con GEMINI_NEWS_ENABLED=false).
    """
    for a in articles:
        if a.av_sentiment is not None:
            a.sentiment = round(float(a.av_sentiment), 3)
        else:
            a.sentiment = _score_keywords(a.text())

    if not _gemini_news_enabled() or not articles:
        return articles

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return articles

    try:
        client, genai_types = create_client()
        if client is None:
            return articles

        for i in range(0, len(articles), GEMINI_BATCH):
            batch = articles[i : i + GEMINI_BATCH]
            headlines = "\n".join(f"{j+1}. {a.title}" for j, a in enumerate(batch))
            prompt = f"""Eres un analista financiero. Analiza el sentimiento de cada titular
respecto a "{company_name}".
Scores: +1.0=muy positivo, +0.5=positivo, 0.0=neutral, -0.5=negativo, -1.0=muy negativo
Responde SOLO con JSON: {{"1": score, "2": score, ...}}

Titulares:
{headlines}"""

            resp = client.models.generate_content(
                model=get_model("news"),
                contents=prompt,
                config=content_config(
                    genai_types,
                    max_output_tokens=256,
                    temperature=0.1,
                ),
            )
            text = (resp.text or "").strip()
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start >= 0 and end > start:
                scores_map = json.loads(text[start:end])
                for j, art in enumerate(batch):
                    key = str(j + 1)
                    if key in scores_map:
                        try:
                            art.sentiment = float(scores_map[key])
                        except (ValueError, TypeError):
                            pass  # mantener score previo (AV o keyword)
            time.sleep(1.5)

    except Exception as e:
        logger.debug(f"Gemini sentiment: {e} — usando keyword scores")

    return articles


def _articles_in_window(articles: list[Article], days_back: int) -> list[Article]:
    """Artículos dentro de la ventana days_back (requiere published_ts válido)."""
    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(days=days_back)
    out: list[Article] = []
    for a in articles:
        if not a.published_ts:
            continue
        if a.published_dt >= cutoff:
            out.append(a)
    return out


def _neutral_news_features(n_articles: int = 0, sources: Optional[list[str]] = None) -> dict[str, float]:
    """Features neutras cuando no hay evidencia de noticias en la ventana."""
    return {
        "news_sent_score":     0.0,
        "news_count_7d":       0.0,
        "news_sent_delta":     0.0,
        "news_sent_vol":       0.0,
        "news_relevance_avg":  0.0,
        "news_esg_score":      0.0,
        "n_articles":          n_articles,
        "sources":             sources or [],
    }


def batch_score_gemini(
    tickers_articles: dict[str, list[Article]],
    top_n_headlines: int = 3,
    days_back: int = 7,
) -> dict[str, float]:
    """
    Calcula sentimiento promedio para múltiples tickers en UNA sola llamada a Gemini.
    Más eficiente que llamar por ticker (ahorra cuota de API).

    Retorna dict ticker → sent_score promedio.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {}

    # Construir resumen compacto: ticker → top-N headlines (solo ventana reciente)
    ticker_headlines: dict[str, list[str]] = {}
    for ticker, arts in tickers_articles.items():
        recent = _articles_in_window(arts, days_back)
        headlines = [a.title for a in recent if a.title][:top_n_headlines]
        if headlines:
            ticker_headlines[ticker] = headlines

    if not ticker_headlines:
        return {}

    try:
        client, genai_types = create_client()
        if client is None:
            return {}

        # Un solo prompt con todos los tickers
        lines = []
        for ticker, hs in ticker_headlines.items():
            joined = " | ".join(hs)
            lines.append(f"{ticker}: {joined}")
        content = "\n".join(lines)

        prompt = f"""Eres un analista financiero senior. Para cada empresa asigna un score
de sentimiento basado en sus noticias recientes:
  +1.0 = muy positivo · 0.0 = neutral · -1.0 = muy negativo

Responde SOLO con JSON: {{"TICKER": score, ...}}

Noticias:
{content}"""

        resp = client.models.generate_content(
            model=get_model("news"),
            contents=prompt,
            config=content_config(
                genai_types,
                max_output_tokens=512,
                temperature=0.1,
            ),
        )
        text = (resp.text or "").strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            return {k: float(v) for k, v in json.loads(text[start:end]).items()}

    except Exception as e:
        logger.debug(f"batch_score_gemini: {e}")

    return {}


# ─────────────────────────────────────────────────────────────────────────────
# DEDUP (KB-041) + TEMPORAL DECAY (KB-043)
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate_tfidf(
    articles: list[Article],
    threshold: float = DEDUP_COSINE_THRESHOLD,
) -> list[Article]:
    """
    Asigna pesos de deduplicación via TF-IDF + cosine similarity.

    Artículos del mismo ticker y día con cosine > threshold reciben
    peso *= DEDUP_DECAY_BASE^n_dup (n_dup = nº de similares previos).
    """
    if not articles or not _SKLEARN_AVAILABLE:
        return articles

    groups: dict[tuple[str, str], list[Article]] = {}
    for art in articles:
        day = art.published_dt.strftime("%Y-%m-%d") if art.published_ts else "unknown"
        groups.setdefault((art.ticker, day), []).append(art)

    for group in groups.values():
        if len(group) < 2:
            continue

        texts = [a.title or " " for a in group]
        try:
            tfidf = TfidfVectorizer().fit_transform(texts)
            sim = cosine_similarity(tfidf)
        except ValueError:
            continue

        for i, art in enumerate(group):
            n_dup = sum(1 for j in range(i) if sim[i, j] > threshold)
            if n_dup > 0:
                art.weight *= DEDUP_DECAY_BASE ** n_dup

    return articles


def _temporal_weight(
    art: Article,
    now: datetime,
    lambda_decay: float = TEMPORAL_DECAY_LAMBDA,
) -> float:
    """Peso por antigüedad: exp(-lambda * days_ago)."""
    if not art.published_ts:
        return 1.0
    days_ago = max(0.0, (now - art.published_dt).total_seconds() / 86400.0)
    return float(np.exp(-lambda_decay * days_ago))


def _effective_weights(
    articles: list[Article],
    now: datetime,
    lambda_decay: float = TEMPORAL_DECAY_LAMBDA,
) -> list[float]:
    """Combina peso de dedup (art.weight) con decay temporal."""
    return [art.weight * _temporal_weight(art, now, lambda_decay) for art in articles]


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    total = w.sum()
    if total <= 0:
        return float(np.mean(v))
    return float(np.average(v, weights=w))


def _weighted_std(values: list[float], weights: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    total = w.sum()
    if total <= 0:
        return float(np.std(v))
    mean = np.average(v, weights=w)
    variance = np.average((v - mean) ** 2, weights=w)
    return float(np.sqrt(max(variance, 0.0)))


def apply_dedup_and_decay(
    articles: list[Article],
    lambda_decay: float = TEMPORAL_DECAY_LAMBDA,
    dedup_threshold: float = DEDUP_COSINE_THRESHOLD,
) -> list[Article]:
    """
    Pipeline KB-041 + KB-043: resetea pesos, deduplica TF-IDF, aplica decay.

    Llamar después de asignar .sentiment y antes de agregar features.
    """
    for art in articles:
        art.weight = 1.0
    _deduplicate_tfidf(articles, threshold=dedup_threshold)
    now = datetime.now(tz=timezone.utc)
    for art in articles:
        art.weight *= _temporal_weight(art, now, lambda_decay)
    return articles


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _compute_features_from_articles(
    articles: list[Article],
    days_back: int = 7,
    lambda_decay: float = TEMPORAL_DECAY_LAMBDA,
    apply_dedup: bool = True,
    as_of: datetime | None = None,
) -> dict[str, float]:
    """
    Calcula features de noticias a partir de artículos con .sentiment asignado.

    as_of: fecha de corte (point-in-time). Default = ahora UTC.
    """
    now = as_of or datetime.now(tz=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cutoff  = now - timedelta(days=days_back)
    cutoff2 = now - timedelta(days=days_back * 2)

    end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=0)
    pit_articles = [a for a in articles if a.published_dt <= end_of_day]

    for art in pit_articles:
        art.weight = 1.0
        if art.relevance_score < LOW_RELEVANCE_THRESHOLD:
            art.weight *= LOW_RELEVANCE_WEIGHT
    if apply_dedup:
        _deduplicate_tfidf(pit_articles)
    for art in pit_articles:
        art.weight *= _temporal_weight(art, end_of_day, lambda_decay)

    recent = [a for a in pit_articles if cutoff <= a.published_dt <= end_of_day]
    prior = [a for a in pit_articles if cutoff2 <= a.published_dt < cutoff]

    w_recent = [a.weight for a in recent]
    w_prior  = [a.weight for a in prior]

    sent_recent = [a.sentiment for a in recent]
    sent_prior  = [a.sentiment for a in prior]

    rel_recent = [a.relevance_score for a in recent]

    esg_recent = [a for a in recent if is_esg_headline(a.title)]
    w_esg = [a.weight for a in esg_recent]
    sent_esg = [a.sentiment for a in esg_recent]

    sent_score      = _weighted_mean(sent_recent, w_recent) if sent_recent else 0.0
    sent_prior_mean = _weighted_mean(sent_prior, w_prior) if sent_prior else 0.0
    sent_vol        = _weighted_std(sent_recent, w_recent) if len(sent_recent) > 1 else 0.0
    relevance_avg   = _weighted_mean(rel_recent, w_recent) if rel_recent else 0.0
    esg_score       = _weighted_mean(sent_esg, w_esg) if sent_esg else 0.0
    eff_count       = sum(w_recent) if w_recent else 0.0
    count_7d        = float(np.log1p(eff_count))
    sent_delta      = round(sent_score - sent_prior_mean, 4)

    return {
        "news_sent_score":     round(sent_score, 4),
        "news_count_7d":       round(count_7d, 4),
        "news_sent_delta":     sent_delta,
        "news_sent_vol":       round(sent_vol, 4),
        "news_relevance_avg":  round(relevance_avg, 4),
        "news_esg_score":      round(esg_score, 4),
        "n_articles":          len(recent),
        "sources":             list({a.source for a in recent}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# API PÚBLICA
# ─────────────────────────────────────────────────────────────────────────────

# Mapa de ticker → query para búsqueda (optimizado por universo)
_SEARCH_QUERIES: dict[str, tuple[str, str]] = {
    # (google_query, keyword_para_rss_mx)
    # BMV — búsqueda en español enfocada en BMV
    "ALSEA":    ("Alsea restaurantes BMV Mexico", "Alsea"),
    "WALMEX":   ("Walmex Walmart Mexico BMV", "Walmex"),
    "FEMSAUBD": ("FEMSA Coca-Cola Mexico BMV", "FEMSA"),
    "GMEXICOB": ("Grupo Mexico mineria BMV", "Grupo Mexico"),
    "CEMEXCPO": ("Cemex cemento Mexico BMV", "Cemex"),
    "GFNORTEO": ("GFNorte Banorte banco Mexico BMV", "Banorte"),
    "BIMBOA":   ("Bimbo alimentos Mexico BMV", "Bimbo"),
    "KIMBERA":  ("Kimberly Clark Mexico BMV", "Kimberly"),
    "PINFRA":   ("Pinfra autopistas Mexico BMV", "Pinfra"),
    "ORBIA":    ("Orbia Mexichem Mexico BMV", "Orbia"),
    "GRUMAB":   ("Gruma maiz tortilla Mexico BMV", "Gruma"),
    "ASURB":    ("Asur aeropuertos sureste Mexico BMV", "Asur"),
    "OMAB":     ("OMA aeropuertos Mexico BMV", "OMA"),
    "GAPB":     ("GAP aeropuertos pacifico Mexico BMV", "GAP"),
    "GCARSOA1": ("GCarso Slim Mexico BMV", "Carso"),
    "MEGACPO":  ("Megacable television Mexico BMV", "Megacable"),
    "LABB":     ("Genomma Lab Mexico BMV", "Genomma"),
    "CUERVO":   ("Jose Cuervo tequila Mexico BMV", "Cuervo"),
    "AC":       ("Arca Continental Coca-Cola Mexico BMV", "Arca Continental"),
    "VESTA":    ("Vesta parques industriales Mexico BMV", "Vesta"),
    "BOLSAA":   ("Bolsa Mexicana Valores BMV", "Bolsa Mexicana"),
    "GCC":      ("GCC cemento Mexico norte BMV", "GCC"),
    "HOTEL":    ("Hoteles City Mexico BMV", "City Express"),
    "POSADAS":  ("Posadas hoteles Mexico BMV", "Posadas"),
    "TLEVISA":  ("Televisa media Mexico BMV", "Televisa"),
    "HERDEZ":   ("Herdez alimentos Mexico BMV", "Herdez"),
    "GENTERA":  ("Gentera microfinanzas Mexico BMV", "Gentera"),
    "FUNO11":   ("FUNO fibra uno Mexico REIT BMV", "Fibra Uno"),
    "VISTA":    ("Vista Energy petroleo Argentina BMV", "Vista Energy"),
    "VOLARA":   ("Volaris aerolinea Mexico BMV", "Volaris"),
    "Q":        ("Quálitas seguros Mexico BMV", "Qualitas"),
    # USA SIC
    "AAPL":    ("Apple iPhone earnings stock", "Apple"),
    "MSFT":    ("Microsoft earnings cloud stock", "Microsoft"),
    "NVDA":    ("NVIDIA GPU AI stock", "NVIDIA"),
    "GOOGL":   ("Alphabet Google earnings stock", "Alphabet"),
    "META":    ("Meta Facebook Instagram stock", "Meta"),
    "AMZN":    ("Amazon AWS earnings stock", "Amazon"),
    "TSLA":    ("Tesla electric car earnings stock", "Tesla"),
    "AMD":     ("AMD semiconductor stock", "AMD"),
    "INTC":    ("Intel semiconductor stock", "Intel"),
    "JPM":     ("JPMorgan Chase bank earnings", "JPMorgan"),
    "BAC":     ("Bank of America earnings stock", "Bank of America"),
    "GS":      ("Goldman Sachs bank earnings", "Goldman"),
    "V":       ("Visa payment earnings stock", "Visa"),
    "MA":      ("Mastercard payment earnings stock", "Mastercard"),
    "KO":      ("Coca-Cola earnings dividend stock", "Coca-Cola"),
    "PEP":     ("PepsiCo earnings dividend stock", "PepsiCo"),
    "MCD":     ("McDonald's earnings stock", "McDonald"),
    "JNJ":     ("Johnson Johnson pharma earnings", "Johnson"),
    "PFE":     ("Pfizer pharma earnings stock", "Pfizer"),
    "XOM":     ("ExxonMobil oil earnings stock", "ExxonMobil"),
    "CVX":     ("Chevron oil earnings stock", "Chevron"),
    "AMX":     ("America Movil telecomunicaciones Mexico", "America Movil"),
    "FMX":     ("FEMSA ADR NYSE Mexico", "FEMSA"),
    # ETFs
    "SPY":     ("SP500 ETF market stock", "S&P 500"),
    "QQQ":     ("NASDAQ QQQ ETF tech", "NASDAQ"),
    "GLD":     ("gold price ETF GLD", "gold"),
    "SLV":     ("silver price ETF SLV", "silver"),
    "TLT":     ("treasury bonds TLT ETF", "treasury"),
    "EEM":     ("emerging markets ETF EEM", "emerging markets"),
    "EWZ":     ("Brazil ETF EWZ stock", "Brazil"),
    "FXI":     ("China ETF FXI stock", "China"),
    "XLE":     ("energy sector ETF XLE oil", "energy ETF"),
    "XLF":     ("financial sector ETF XLF bank", "financial ETF"),
    "XLK":     ("technology sector ETF XLK", "tech ETF"),
    # Cripto
    "IBIT":    ("Bitcoin ETF BlackRock IBIT", "Bitcoin"),
    "FBTC":    ("Bitcoin ETF Fidelity FBTC", "Bitcoin"),
    "ETHA":    ("Ethereum ETF iShares ETHA", "Ethereum"),
    "GBTC":    ("Grayscale Bitcoin Trust GBTC", "Bitcoin"),
    "MSTR":    ("MicroStrategy Bitcoin MSTR stock", "MicroStrategy"),
    "COIN":    ("Coinbase crypto exchange earnings", "Coinbase"),
    "MARA":    ("Marathon Digital Bitcoin mining", "Marathon"),
    "RIOT":    ("Riot Platforms Bitcoin mining", "Riot"),
    "CLSK":    ("CleanSpark Bitcoin mining", "CleanSpark"),
    # FIBRAs
    "FIBRAPL": ("Fibra Prologis REIT Mexico industrial", "Prologis"),
    "FINN13":  ("Fibra Inn hotel REIT Mexico", "Fibra Inn"),
}


def _scrape_articles(
    ticker: str,
    yf_symbol: str,
    days_back: int = 7,
    use_gdelt: bool = False,
) -> list[Article]:
    """Descarga artículos de todas las fuentes para un ticker. Sin scoring."""
    query_info = _SEARCH_QUERIES.get(ticker, (f"{ticker} stock", ticker))
    google_q   = query_info[0]
    keyword_mx = query_info[1] if len(query_info) > 1 else ticker
    aliases    = get_entity_aliases(ticker, extra=[keyword_mx])

    articles: list[Article] = []
    is_mx = yf_symbol.endswith(".MX") or ticker in ("AMX", "FMX")
    lang  = "es-MX" if is_mx else "en-US"

    if is_mx:
        articles.extend(
            _dict_to_article(d, ticker)
            for d in fetch_databursatil_articles(ticker, aliases=aliases, max_items=MAX_ARTICLES)
        )

    articles.extend(fetch_yfinance(yf_symbol, ticker))
    articles.extend(fetch_google_news_rss(google_q, ticker, lang=lang))

    if is_mx:
        articles.extend(fetch_elfinanciero_rss(ticker, keyword_mx))
        articles.extend(fetch_expansion_rss(ticker, keyword_mx))

    if os.environ.get("ALPHA_VANTAGE_API_KEY"):
        articles.extend(
            _dict_to_article(d, ticker)
            for d in fetch_alpha_vantage_articles(ticker, max_items=MAX_ARTICLES, aliases=aliases)
        )

    crypto_tickers = {"IBIT", "FBTC", "ETHA", "GBTC", "MSTR", "COIN", "MARA", "RIOT", "CLSK"}
    if ticker in crypto_tickers:
        articles.extend(fetch_cryptopanic(ticker))

    if use_gdelt:
        articles.extend(fetch_gdelt(google_q, ticker, days_back=days_back * 2))

    # Preserve connector output before local deduplication; not a raw HTTP response.
    from src.data.news_medallion import ingest
    for source in sorted({a.source for a in articles}):
        ingest(source, [dict(asdict(a), sentiment=a.av_sentiment)
                        for a in articles if a.source == source])

    # Deduplicar por URL/título
    seen: set[str] = set()
    unique: list[Article] = []
    for a in articles:
        key = a.url or a.title
        if key and key not in seen:
            seen.add(key)
            unique.append(a)

    return unique


def get_news_features(
    ticker: str,
    yf_symbol: str,
    company_name: Optional[str] = None,
    days_back: int = 7,
    use_gdelt: bool = False,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Pipeline completo: descarga noticias de todas las fuentes disponibles,
    corre sentiment (Gemini o keywords) y calcula las 4 features.

    Parameters
    ----------
    ticker      : nombre corto del ticker (e.g. "ALSEA")
    yf_symbol   : símbolo yfinance (e.g. "ALSEA.MX")
    company_name: nombre para búsqueda (si None, usa lookup interno)
    days_back   : ventana de noticias en días
    use_gdelt   : incluir GDELT (más lento, útil para backfill histórico)

    Returns
    -------
    dict con: news_sent_score, news_count_7d, news_sent_delta,
              news_sent_vol, n_articles, sources
    """
    cname = company_name or _SEARCH_QUERIES.get(ticker, (ticker, ticker))[1]

    # Scrape
    articles = _scrape_articles(ticker, yf_symbol, days_back=days_back, use_gdelt=use_gdelt)
    if verbose:
        print(f"  {ticker}: {len(articles)} artículos únicos")

    # Keyword scoring (siempre) + Gemini individual si GEMINI_NEWS_ENABLED=true
    scored = score_with_gemini(articles, cname)

    feat = _compute_features_from_articles(scored, days_back=days_back)
    if feat.get("n_articles", 0) == 0:
        return _neutral_news_features()
    return feat


def batch_news_features(
    tickers: dict[str, str],
    days_back: int = 7,
    use_gdelt: bool = False,
    verbose: bool = True,
    save_cache: bool = True,
) -> pd.DataFrame:
    """
    Calcula news features para todos los tickers de un universo.

    Parameters
    ----------
    tickers   : dict nombre→símbolo yfinance (e.g. {"ALSEA": "ALSEA.MX"})
    days_back : ventana de noticias
    use_gdelt : incluir GDELT (lento, para backfill)
    save_cache: guardar resultado en data/news/YYYY-MM-DD.json

    Returns
    -------
    DataFrame con columnas: ticker, news_sent_score, news_count_7d,
                            news_sent_delta, news_sent_vol, n_articles
    """
    # ── Fase 1: scraping de todas las fuentes ────────────────────────────────
    tickers_articles: dict[str, list[Article]] = {}
    for name, symbol in tickers.items():
        if verbose:
            print(f"  [{name}] {symbol}...", end=" ", flush=True)
        try:
            articles = _scrape_articles(name, symbol, days_back=days_back, use_gdelt=use_gdelt)
            tickers_articles[name] = articles
            if verbose:
                print(f"{len(articles)} arts")
        except Exception as e:
            tickers_articles[name] = []
            if verbose:
                print(f"ERROR: {e}")

    # ── Fase 2: sentiment (AV > keywords) + Gemini batch (si hay API key) ──
    for name, arts in tickers_articles.items():
        for a in arts:
            if a.av_sentiment is not None:
                a.sentiment = round(float(a.av_sentiment), 3)
            else:
                a.sentiment = _score_keywords(a.text())

    gemini_scores = batch_score_gemini(tickers_articles, top_n_headlines=3, days_back=days_back)
    if gemini_scores and verbose:
        print(f"\n  Gemini batch: {len(gemini_scores)} tickers con score NLP\n")

    # ── Fase 3: computar features ─────────────────────────────────────────
    results = []
    for name, arts in tickers_articles.items():
        try:
            feat = _compute_features_from_articles(arts, days_back=days_back)
            # Gemini solo si hay artículos reales en ventana (DATA-001)
            if feat.get("n_articles", 0) > 0 and name in gemini_scores:
                feat["news_sent_score"] = round(gemini_scores[name], 4)
            elif feat.get("n_articles", 0) == 0:
                feat = _neutral_news_features()
            feat["ticker"] = name
            results.append(feat)
            if verbose:
                n    = feat["n_articles"]
                s    = feat["news_sent_score"]
                srcs = ",".join(feat.get("sources", []))
                g    = " (Gemini)" if name in gemini_scores else ""
                print(f"  {name:<15} {n:>3} arts  sent={s:+.2f}{g}  [{srcs}]")
        except Exception as e:
            logger.warning(f"{name}: error en news features — {e}")
            results.append({
                "ticker":              name,
                "news_sent_score":     0.0,
                "news_count_7d":       0.0,
                "news_sent_delta":     0.0,
                "news_sent_vol":       0.0,
                "news_relevance_avg":  0.0,
                "news_esg_score":      0.0,
                "n_articles":          0,
                "sources":             [],
            })

    df = pd.DataFrame(results).set_index("ticker")
    for col in NEWS_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    if save_cache:
        today = datetime.now().strftime("%Y-%m-%d")
        cache_path = NEWS_DIR / f"{today}.json"
        cache_data = df.reset_index().to_dict(orient="records")
        cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2))
        (NEWS_DIR / "latest.json").write_text(
            json.dumps(cache_data, ensure_ascii=False, indent=2)
        )
        from src.data.news_medallion import build_gold
        build_gold(days_back=days_back)
        if verbose:
            print(f"\n  Guardado en: {cache_path}")

    return df


def load_cached_news_features(date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Carga features de noticias del cache en disco.
    Si date=None, carga 'latest.json'.
    """
    if date:
        path = NEWS_DIR / f"{date}.json"
    else:
        path = NEWS_DIR / "latest.json"

    if not path.exists():
        return None

    data = json.loads(path.read_text())
    df   = pd.DataFrame(data)
    if "ticker" in df.columns:
        df = df.set_index("ticker")
    for col in NEWS_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df
