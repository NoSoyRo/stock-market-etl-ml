import pandas as pd
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.news_medallion import clean_news_text
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

# Importamos el motor del profesor
from src.ml.news import (
    Article,
    _score_keywords,
    _scrape_articles,
    _compute_features_from_articles
)

def integrar_noticias():
    print("🧠 Iniciando integración de Sentimiento y TF-IDF...")
    
    # 1. Cargar noticias de Trump desde tu CSV
    ruta_csv = ROOT / "data/news/noticias_trump.csv"
    if not os.path.exists(ruta_csv):
        print("❌ No se encontró noticias_trump.csv")
        return
        
    df_trump = pd.read_csv(ruta_csv).fillna('')
    # The public CSV intentionally has no text column. Silver retains RSS excerpts.
    silver_path = ROOT / "data/news/silver/articles.parquet"
    excerpts = {}
    if silver_path.exists():
        silver = pd.read_parquet(silver_path)
        silver = silver[(silver.source == 'trump') &
                        (silver.available_at <= pd.Timestamp.now(tz='UTC'))]
        latest = silver.sort_values('fetched_at').drop_duplicates('url', keep='last')
        excerpts = latest[latest.text.notna()].set_index('url').text.to_dict()

    
    # 2. Convertir el CSV a objetos Article (lo que espera el profesor)
    articulos_trump = []
    for _, row in df_trump.iterrows():
        try:
            # Convertir la fecha del RSS a timestamp
            dt = parsedate_to_datetime(row['fecha'])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts = int(dt.timestamp())
        except (TypeError, ValueError, OverflowError):
            # A missing publication is not replaced by a made-up publication time.
            continue
            
        title = clean_news_text(row['titulo'])
        summary = clean_news_text(excerpts.get(row['url']) or row.get('texto') or '')
        if summary.casefold() == title.casefold():
            summary = ''
        articulos_trump.append(Article(
            ticker="WALMEX", # Asignamos las noticias de Trump a WALMEX para este ejemplo
            title=title,
            summary=summary,
            url=str(row['url']),
            published_ts=ts,
            source="Trump_RSS"
        ))
        
    print(f"📥 Artículos de Trump cargados: {len(articulos_trump)}")

    # 3. Descargar las noticias normales de WALMEX usando el scraper del profe
    print("🕷️ Descargando noticias generales de WALMEX (esto puede tardar)...")
    articulos_walmex = _scrape_articles("WALMEX", "WALMEX.MX")
    print(f"📥 Artículos generales de WALMEX: {len(articulos_walmex)}")
    
    # 4. Juntar ambas listas
    todos_los_articulos = articulos_walmex + articulos_trump
    print(f"📚 Total de artículos a procesar: {len(todos_los_articulos)}")
    
    # 5. Calcular Sentimiento (usando la función del profe)
    for art in todos_los_articulos:
        # Aplicamos la función de keywords del profe
        art.sentiment = _score_keywords(art.text())
        
    # 6. Aplicar Deduplicación (TF-IDF) y Decaimiento Temporal
    print("🧹 Aplicando Deduplicación (TF-IDF) y Decaimiento Temporal...")
    # _compute_features_from_articles applies deduplication and decay itself.
    
    # 7. Calcular las Features finales (Capa Golden)
    features = _compute_features_from_articles(todos_los_articulos)
    
    print("\n✅ ¡Integración Exitosa! Resultados de la Capa Golden para WALMEX:")
    print(f"Sentimiento final: {features['news_sent_score']}")
    print(f"Total de artículos procesados: {features['n_articles']}")
    print(f"Fuentes utilizadas: {features['sources']}")
    
    # Guardar el resultado para tu Pull Request
    import json
    (ROOT / "data/golden").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "data/golden/walmex_integrado.json", "w", encoding="utf-8") as f:
        json.dump(features, f, indent=4, ensure_ascii=False)
    print("📁 Guardado en: data/golden/walmex_integrado.json")

if __name__ == "__main__":
    integrar_noticias()
