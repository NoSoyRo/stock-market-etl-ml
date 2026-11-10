import pandas as pd
import json
import os

def mapear_impacto_trump():
    print("🔗 Iniciando mapeo de noticias de Trump a Tickers afectados...")
    
    # 1. Ruta del CSV que generaste
    ruta_csv = "data/news/noticias_trump.csv"
    
    if not os.path.exists(ruta_csv):
        print("❌ Error: No se encontró el archivo de noticias de Trump.")
        return

    # 2. Cargar los datos
    df = pd.read_csv(ruta_csv)
    
    # 3. El famoso "Mapa de Impacto" (Puedes agregar más palabras y tickers)
    mapa_impacto = {
        "arancel": ["WALMEX", "CEMEXCPO", "GMEXICOB"],
        "economia": ["WALMEX", "FEMSAUBD", "GFNORTEO"],
        "comercio": ["WALMEX", "BIMBOA"],
        "mexico": ["WALMEX", "FEMSAUBD", "AMXB"],
        "acero": ["CEMEXCPO", "GMEXICOB"],
        "petroleo": ["XOM", "CVX"], # Empresas de USA
        "bitcoin": ["BTC", "IBIT"]  # Por si Trump habla de cripto
    }

    noticias_mapeadas = []

    # 4. Recorrer cada noticia y buscar coincidencias
    for index, row in df.iterrows():
        titulo = str(row['titulo']).lower()
        tickers_afectados = set() # Usamos un set para no repetir tickers

        # Buscar si alguna palabra clave está en el título
        for palabra_clave, tickers in mapa_impacto.items():
            if palabra_clave in titulo:
                tickers_afectados.update(tickers)

        # Si la noticia afecta a alguien, la guardamos
        if tickers_afectados:
            noticias_mapeadas.append({
                "titulo": row['titulo'],
                "fecha": row['fecha'],
                "tickers_afectados": list(tickers_afectados)
            })

    # 5. Guardar el resultado en Local
    os.makedirs("data/golden", exist_ok=True)
    ruta_salida = "data/golden/mapeo_trump_golden.json"
    
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        json.dump(noticias_mapeadas, f, indent=4, ensure_ascii=False)

    print(f"✅ ¡Mapeo exitoso!")
    print(f"De {len(df)} noticias, {len(noticias_mapeadas)} fueron asociadas a tickers específicos.")
    print(f"📁 Guardado en: {ruta_salida}")
    
    # Mostrar una vista previa
    print("\nVista previa de las noticias mapeadas:")
    for noticia in noticias_mapeadas[:3]:
        print(f"- {noticia['titulo'][:60]}... -> Afecta a: {noticia['tickers_afectados']}")

if __name__ == "__main__":
    mapear_impacto_trump()