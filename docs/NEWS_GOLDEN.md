# Golden Diario de Noticias Segmentadas (`lab.news_feature_daily`)

La arquitectura local bronce/silver/gold y la integración actual de Trump se describen en [NEWS_MEDALLION.md](NEWS_MEDALLION.md). Las secciones siguientes sobre tablas SQL mantienen su alcance independiente.

## 1. Propósito

El producto **Golden diario de noticias segmentadas** (`lab.news_feature_daily`) almacena en la base de datos `mercado` (esquema `lab` de PostgreSQL 16) las características cuantitativas (*features*) agregadas del flujo noticioso para los universos de inversión configurados en el proyecto (BMV, USA SIC, ETFs, Cripto proxies y FIBRAs).

Este dataset sintetiza el análisis de procesamiento de lenguaje natural (NLP), deduplicación léxica y descuento temporal sobre noticias financieras, proporcionando una fotografía estandarizada y normalizada de cada activo por día de operación bursátil.

---

## 2. Grano y Llave

- **Grano:** Exactamente **una fila por ticker × fecha hábil** (`ticker × trade_date`).
- **Llave Primaria:** `PRIMARY KEY (ticker, trade_date)`.
- **Estrategia de Carga:** Inserción y actualización idempotente (*UPSERT*) mediante:
  ```sql
  ON CONFLICT (ticker, trade_date) DO UPDATE
  ```
  Ejecutar el pipeline de ingesta múltiples veces para una misma fecha actualiza los registros existentes sin duplicar filas ni corromper series de tiempo.

---

## 3. Diccionario de Columnas

| Columna | Tipo SQL | Descripción y Reglas de Negocio |
| :--- | :--- | :--- |
| `ticker` | `text NOT NULL` | Símbolo identificador del activo en el universo (ej. `"WALMEX"`, `"AAPL"`, `"SPY"`). |
| `trade_date` | `date NOT NULL` | Fecha de la sesión de mercado correspondiente a la ventana de noticias. |
| `universe` | `text` | Slug del universo al que pertenece el activo (`"bmv"`, `"usa"`, `"etfs"`, `"cripto"`, `"fibras"`). |
| `news_sent_score` | `double precision` | Puntuación agregada de sentimiento ponderado en escala $[-1.0, +1.0]$. Descuenta artículos repetidos y noticias antiguas. |
| `news_count_7d` | `double precision` | Volumen mediático efectivo en la ventana móvil de 7 días, comprimido logarítmicamente: $\ln(1 + \sum w_i)$. Evita distorsiones por coberturas sindicadas masivas. |
| `news_sent_delta` | `double precision` | Cambio del sentimiento respecto a la ventana de 7 días previa ($\mathrm{sent}_{t} - \mathrm{sent}_{t-7d}$). Detecta giros o quiebres repentinos de narrativa. |
| `news_sent_vol` | `double precision` | Desviación estándar ponderada del sentimiento de los artículos recientes. Mide incertidumbre o desacuerdo entre medios de prensa. |
| `news_relevance_avg` | `double precision` | Puntuación media ponderada de relevancia léxica de los titulares frente a los alias corporativos de la empresa $[0.0, 1.0]$. |
| `news_esg_score` | `double precision` | Puntuación de sentimiento restringida a artículos con temática ambiental, social o de gobernanza corporativa (ASG / ESG). |
| `n_articles` | `integer` | Conteo entero bruto de artículos recopilados dentro de la ventana de 7 días. |
| `sources` | `text` | Lista de proveedores que aportaron artículos en la ventana, separada por comas (ej. `"expansion,google_news"`). |
| `loaded_at` | `timestamptz NOT NULL` | Marca de tiempo UTC de inserción o última actualización (`DEFAULT now()`). |

---

## 4. Diferencia entre `lab.news_daily` y `lab.news_feature_daily`

El esquema `lab` mantiene una separación rigurosa entre datos no estructurados en crudo y métricas analíticas agregadas:

| Criterio | `lab.news_daily` | `lab.news_feature_daily` |
| :--- | :--- | :--- |
| **Grano** | **1 fila por artículo individual** | **1 fila por ticker × trade_date** |
| **Llave** | `id serial PRIMARY KEY` | `PRIMARY KEY (ticker, trade_date)` |
| **Contenido** | Titular (`title`), URL, fuente y sentimiento de cada noticia en crudo. | Las 6 features cuantitativas (`NEWS_FEATURE_COLS`), volumen comprimido y metadatos agregados. |
| **Cardinalidad** | 0 a $N$ artículos por ticker y día. | Máximo 1 registro por ticker en cada fecha. |
| **Comportamiento** | Inserción acumulativa de notas de prensa. | *UPSERT* idempotente. |
| **Objetivo** | Trazabilidad documental y lectura directa de titulares. | Alimentación cuantitativa de modelos de Machine Learning y semáforos de agentes. |

---

## 5. Consumidores del Golden

1. **Capa Semántica y Agentes de Consulta (`ver_noticias`):**
   Permite al agente consultar de forma instantánea el sentimiento numérico, volumen y alertas de un activo sin necesidad de recalcular scraping o procesar texto en tiempo de ejecución.
2. **Modelos Cuantitativos y SuperModel:**
   Inyección directa de `NEWS_FEATURE_COLS` en el ensamble de Machine Learning (LightGBM, WoE y Regresión Logística) para generar señales multi-horizonte.
3. **Agentes de Validación y Riesgo (`Gemini Validator`):**
   Detección de condiciones de veto (ej. regla `LR-003: noticias contradicen momentum técnico` cuando `news_sent_score < -0.25` o `news_alerta = True`).

---

## 6. Ingesta temática de noticias sobre Donald Trump

### Propósito y Alcance
La ingesta temática implementada en [scripts/ingesta_trump.py](../scripts/ingesta_trump.py) es un proceso experimental especializado en monitorear el flujo de noticias internacionales y macroeconómicas relacionadas con **Donald Trump**, aranceles, política comercial, mercados y economía.

A diferencia del pipeline general de noticias que indexa por ticker o activo individual, este proceso extrae información temática agregada utilizando el servicio público de **Google News RSS** parametrizado con consultas booleanas y filtros geográficos y de idioma para México (`hl=es-419`, `gl=MX`, `ceid=MX:es-419`).

### Ejecución
El script se ejecuta directamente desde la raíz del proyecto mediante el entorno virtual:

```bash
python scripts/ingesta_trump.py
```

### Salida Local y Esquema de Datos
El resultado de la extracción se persiste localmente en la ruta:

`data/news/noticias_trump.csv`

> [!IMPORTANT]
> El archivo `data/news/noticias_trump.csv` se versiona junto con los datos de las capas medallion para conservar en el repositorio la información descargada.

El dataset exportado contiene las siguientes columnas:
- `titulo`: Titular completo de la nota de prensa según lo entrega el feed RSS.
- `fecha`: Fecha y hora de publicación original en formato RFC 822 (ej. `Mon, 26 Oct 2026 14:23:56 GMT`).
- `url`: Enlace proporcionado por Google News RSS (puede redirigir al medio original).
- `fuente`: Identificador de la fuente temática (`"Google News (Trump)"`).
El CSV ya no contiene `texto`. El resumen RSS limpio de HTML se conserva en bronce y silver; cuando falta, se usa el titular como respaldo. Los artículos con resumen distinto al titular se clasifican como `excerpt`, y los demás como `headline`.

### Estado dentro de la Arquitectura Golden

La ingesta ya escribe las capas locales bronce y silver y calcula indicadores temáticos en gold; véase [NEWS_MEDALLION.md](NEWS_MEDALLION.md). Esto no implica una carga en las tablas SQL descritas aquí.
- **Nivel de madurez:** Esta ingesta es de carácter **temático y experimental**.
- **Persistencia en base de datos:** Actualmente **no** carga directamente en `lab.news_daily` ni en `lab.news_feature_daily`.
- **Criterio de integración:** No debe presentarse como parte completa del producto Golden hasta que sea formalmente integrada al pipeline diario, asociada a tickers afectados (o un identificador macroeconómico transversal), normalizada léxicamente y deduplicada mediante TF-IDF y decaimiento temporal.

### Métricas de Ejecución en Vivo
Durante una ejecución en vivo específica de validación se obtuvieron las siguientes métricas (cabe señalar que tanto los 100 artículos como la cobertura temporal corresponden a dicha corrida puntual y pueden cambiar en ejecuciones posteriores al tratarse de una fuente en vivo):
- **Artículos descargados:** **100 noticias** (muestra de la ejecución en vivo puntual; puede cambiar en futuras consultas).
- **Cobertura temporal observada:** Desde `2026-07-20 07:00:00+00:00` hasta `2026-10-27 00:28:47+00:00` (específica de esta ejecución; variable según las notas disponibles en vivo).
- **Calidad de campos:** 0 títulos vacíos, 0 URLs vacías y 0 duplicados exactos por URL.

### Limitaciones Conocidas
1. **Dependencia de Red y Rate Limiting:** Requiere conexión a Internet saliente y está sujeta a la disponibilidad y límites de tasa del endpoint RSS de Google News.
2. **Contenido Dinámico y Volátil:** El conjunto de notas cambia dinámicamente según la actualidad política y comercial.
3. **Profundidad de Contenido:** El canal RSS provee titulares y resúmenes fragmentados, no el artículo periodístico completo.
4. **Duplicación de Narrativas:** Diferentes agencias y medios pueden reportar el mismo evento con redacciones ligeramente distintas (sindicación de prensa), requiriendo deduplicación semántica previa a su uso como señal predictiva.
