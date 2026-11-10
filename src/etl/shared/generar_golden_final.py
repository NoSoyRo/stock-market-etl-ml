import json
import pandas as pd
import os
from datetime import datetime, timezone

def generar_tabla_golden():
    print("🥇 Iniciando la construcción de la Tabla Golden Final...")

    # 1. Leer el resultado del Paso 2
    ruta_integracion = "data/golden/walmex_integrado.json"
    if not os.path.exists(ruta_integracion):
        print("❌ No se encontró el archivo de integración del Paso 2.")
        return

    with open(ruta_integracion, "r", encoding="utf-8") as f:
        datos_integrados = json.load(f)

    # 2. Armar el registro siguiendo EXACTAMENTE el esquema de NEWS_GOLDEN.md
    # Nota: Aquí estamos generando 1 fila (WALMEX). Si tuviéramos más tickers, 
    # haríamos un bucle para cada uno.
    
    fecha_hoy = datetime.now().strftime("%Y-%m-%d") # trade_date
    timestamp_actual = datetime.now(timezone.utc).isoformat() # loaded_at

    registro_golden = {
        "ticker": "WALMEX",
        "trade_date": fecha_hoy,
        "universe": "bmv",
        "news_sent_score": datos_integrados.get("news_sent_score", 0.0),
        "news_count_7d": datos_integrados.get("news_count_7d", 0.0),
        "news_sent_delta": datos_integrados.get("news_sent_delta", 0.0),
        "news_sent_vol": datos_integrados.get("news_sent_vol", 0.0),
        "news_relevance_avg": datos_integrados.get("news_relevance_avg", 0.0),
        "news_esg_score": datos_integrados.get("news_esg_score", 0.0),
        "n_articles": datos_integrados.get("n_articles", 0),
        "sources": ", ".join(datos_integrados.get("sources", [])), # Lo convertimos a texto separado por comas
        "loaded_at": timestamp_actual
    }

    # 3. Convertir a DataFrame de Pandas (simulando la tabla de la BD)
    df_golden = pd.DataFrame([registro_golden])

    # 4. Guardar la tabla en Local (JSON y CSV)
    os.makedirs("data/golden", exist_ok=True)
    
    ruta_json = "data/golden/lab_news_feature_daily.json"
    ruta_csv = "data/golden/lab_news_feature_daily.csv"

    # Guardar como JSON (útil para la IA)
    df_golden.to_json(ruta_json, orient="records", indent=4, force_ascii=False)
    
    # Guardar como CSV (útil para revisar en Excel)
    df_golden.to_csv(ruta_csv, index=False)

    print("✅ ¡Tabla Golden generada con éxito!")
    print(f"📁 Guardado en: {ruta_json}")
    print("\nVista previa de la tabla final (Capa Golden):")
    print(df_golden.T) # Imprime la tabla transpuesta para verla mejor en la terminal

if __name__ == "__main__":
    generar_tabla_golden()