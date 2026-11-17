import feedparser
import pandas as pd
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.news_medallion import clean_news_text

def buscar_noticias_trump():
    print("Iniciando ingesta de noticias de Donald Trump (Impacto en mercados)...")

    # DECISIÓN DE ARQUITECTURA:
    # Usamos Google News RSS con URLs parametrizadas en lugar de la API de X (Twitter).
    # Esto garantiza que el proceso sea 100% automatizable, sin costos de API y escalable.
    url_rss = "https://news.google.com/rss/search?q=Donald+Trump+aranceles+OR+mercados+OR+economia&hl=es-419&gl=MX&ceid=MX:es-419"

    # Parsear el feed RSS
    feed = feedparser.parse(url_rss)

    noticias = []
    for entry in feed.entries:
        # Limpiamos el HTML que trae el resumen de Google News por defecto
        texto_limpio = clean_news_text(entry.get("summary") or entry.title)
        titulo_limpio = clean_news_text(entry.title)
        
        noticias.append({
            "titulo": titulo_limpio,
            "fecha": entry.published,
            "url": entry.link,
            "fuente": "Google News (Trump)",
            "texto": texto_limpio
        })

    if not noticias:
        print("No se encontraron noticias de Trump hoy.")
        return

    # Convertir a DataFrame de Pandas (Este df conserva 'texto' para el motor Medallion)
    df = pd.DataFrame(noticias)

    # Crear la carpeta data/news si no existe
    (ROOT / "data/news").mkdir(parents=True, exist_ok=True)
    ruta_csv = ROOT / "data/news/noticias_trump.csv"

    # Crear una copia del DataFrame SIN la columna 'texto' para exportar el CSV limpio
    df_csv = df.drop(columns=['texto'], errors='ignore')

    # Guardar la tabla local limpia
    df_csv.to_csv(ruta_csv, index=False, encoding="utf-8")

    # --- ESTO ES LO NUEVO DEL PROFE (La Arquitectura Medallion) ---
    # Enviamos los datos (con la columna texto intacta) al motor del profesor
    from src.data.news_medallion import ingest, build_gold
    ingest("trump", noticias, representation="rss_extract")
    build_gold()
    # -------------------------------------------------------------

    print(f"Exito! Se descargaron {len(df)} noticias de Trump.")
    print(f"Guardado en: {ruta_csv}")
    print("\nVista previa de los datos (sin columna de texto):")
    print(df_csv.head(3))

if __name__ == "__main__":
    buscar_noticias_trump()