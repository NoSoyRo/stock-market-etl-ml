# Servidor MCP de Noticias Segmentadas

## 1. ¿Qué es el MCP de Noticias Segmentadas?

El **Servidor MCP de Noticias Segmentadas** es una interfaz basada en el estándar abierto [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) que permite a modelos de lenguaje (LLMs) y agentes conversacionales (como Claude Desktop, Antigravity, Cursor, etc.) consultar el estado informativo y de sentimiento de noticias financieras preparado por el pipeline Medallion del proyecto.

El servidor expone herramientas semánticas de sólo lectura que traducen indicadores cuantitativos y series temporales a un formato comprensible, seguro y serializable en JSON.

---

## 2. ¿Qué capa consume? Separación de responsabilidades

El servidor MCP respeta estrictamente la arquitectura por capas del proyecto:

```text
┌─────────────────────────────────────────────────────────────────┐
│                       Agentes / Clientes LLM                    │
│                 (Claude Desktop, Cursor, Chatbots)              │
└────────────────────────────────┬────────────────────────────────┘
                                 │ Protocolo MCP (stdio / JSON-RPC)
┌────────────────────────────────▼────────────────────────────────┐
│             Servidor MCP (src/mcp/server.py)                    │
│      - ver_noticias          - resumen_noticias                 │
│      - comparar_noticias     - listar_tickers_noticias          │
└────────────────────────────────┬────────────────────────────────┘
                                 │ Métodos de dominio
┌────────────────────────────────▼────────────────────────────────┐
│           Capa Semántica (src/semantic/service.py)              │
│      - Normalización de entidades y fechas                      │
│      - Métricas de negocio (interpretation.py / metrics.json)   │
│      - Reglas de negocio y comprobación de integridad           │
└────────────────────────────────┬────────────────────────────────┘
                                 │ Lectura de Parquet
┌────────────────────────────────▼────────────────────────────────┐
│                        Capa Gold (Medallion)                    │
│    - data/news/gold/today_features.parquet                      │
│    - data/news/gold/historical_features.parquet                 │
│    - Parquet Silver: data/news/silver/articles.parquet          │
│    (Sincronizada con PostgreSQL lab.news_feature_daily)         │
└─────────────────────────────────────────────────────────────────┘
```

### Límites de responsabilidad:
*   **Bronze / Silver:** Ingesta cruda, normalización de titulares y deduplicación.
*   **Gold:** Cálculo de agregaciones numéricas de sentimiento y volumen por ticker y fecha.
*   **Capa Semántica:** Traduce columnas técnicas a métricas de negocio, valida cortes temporales y produce respuestas tipificadas con advertencias.
*   **MCP:** Expone las herramientas hacia agentes externos mediante el protocolo estándar MCP sin ejecutar scraping, sin descargas y sin recalcular pipelines.

---

## 3. Diferencia entre la Capa Semántica y el MCP

| Aspecto | Capa Semántica (`src/semantic`) | Servidor MCP (`src/mcp`) |
|---|---|---|
| **Propósito** | Definir el significado de las métricas de negocio, validar integridad y resolver evidencia histórica. | Exponer herramientas según el estándar JSON-RPC de MCP para agentes y LLMs. |
| **Consumidores** | Código Python interno: CLI (`scripts/query_news_semantic.py`), agentes internos y el servidor MCP. | Agentes externos: Claude Desktop, Cursor, Antigravity, MCP Inspector. |
| **Dependencia** | Puramente de dominio (Pandas, PyArrow, modelos dataclass). | Protocolo MCP (`mcp>=2.3.0`). |
| **Entrada/Salida** | Métodos Python que devuelven `dict[str, Any]` validados con `SemanticResponse`. | Herramientas MCP con firmas y esquemas tipificados (`@server.tool`). |

---

## 4. Herramientas expuestas y parámetros

El servidor expone cuatro herramientas principales:

### 1. `ver_noticias`
Consulta las métricas semánticas y el estado informativo más reciente (o en una fecha histórica específica) para un ticker.

*   **Parámetros:**
    *   `ticker` (`str`, **obligatorio**): Clave de pizarra o símbolo de la empresa (ej. `"WALMEX"`, `"AC"`, `"BOLSAA"`).
    *   `fecha` (`str`, *opcional*): Fecha en formato ISO `YYYY-MM-DD`. Si se omite, consulta la tabla de hoy (`today_features.parquet`).
    *   `limite` (`int`, *opcional*): Límite de control para metadatos o fuentes.

### 2. `resumen_noticias`
Resume la evolución temporal de las métricas de noticias para un ticker durante un periodo, calculando la tendencia del sentimiento y el volumen observado.

*   **Parámetros:**
    *   `ticker` (`str`, **obligatorio**): Clave de pizarra de la empresa.
    *   `fecha_inicio` (`str`, *opcional*): Fecha inicial en formato `YYYY-MM-DD`. Si se omite, se infiere del rango de cobertura disponible.
    *   `fecha_fin` (`str`, *opcional*): Fecha final en formato `YYYY-MM-DD`. Si se omite, se infiere del rango de cobertura disponible.

### 3. `comparar_noticias`

Antes de generar rankings, valida que las filas compartan fecha UTC, `window_days`
y `scoring_method`. Si difieren o no se conocen, responde `incompatible_metrics`
sin rankings. Las filas de la comparativa y `ver_noticias` incluyen ventana y
método. El resumen histórico también rechaza agregaciones de ventanas o métodos
mezclados.
Compara métricas de dos o más activos en una fecha específica, ordenándolos por sentimiento y volumen ponderado.

*   **Parámetros:**
    *   `tickers` (`list[str]` o `str`, **obligatorio**): Lista de claves de pizarra (ej. `["WALMEX", "FEMSAUBD", "ALSEA"]`) o texto separado por comas (`"WALMEX, FEMSAUBD, ALSEA"`).
    *   `fecha` (`str`, *opcional*): Fecha de corte en formato `YYYY-MM-DD`. Si se omite, evalúa las métricas de hoy.

### 4. `listar_tickers_noticias`
Informa el universo de cobertura actual: tickers disponibles, temas, rango de fechas analizadas, fuentes y fecha del último cálculo.

*   **Parámetros:** Ninguno obligatorio.

---

## 5. Ejemplos de entrada y salida

### Ejemplo 1: `ver_noticias`

**Entrada:**
```json
{
  "ticker": "AC"
}
```

**Salida:**
```json
{
  "ok": true,
  "status": "ok",
  "ticker": "AC",
  "fecha_utc": "2026-10-31",
  "metricas": {
    "sentimiento_general": 0.0,
    "volumen_informativo_ponderado": 1.7891,
    "relevancia_promedio": 1.0,
    "cantidad_articulos": 30,
    "cambio_sentimiento": 0.0,
    "dispersion_sentimiento": 0.0,
    "sentimiento_esg": null
  },
  "fuentes": [
    "databursatil"
  ],
  "resumen_semantico": "Resultado numéricamente neutral según palabras clave; no demuestra neutralidad económica.",
  "cobertura_esg": "unknown",
  "fecha_calculo_utc": "2026-10-31T00:05:22.546824+00:00",
  "antiguedad_calculo_horas": 19.3,
  "advertencias": [
    "La cobertura ESG es desconocida hasta revisar los titulares."
  ]
}
```

### Ejemplo 2: `resumen_noticias`

El total de artículos se resuelve desde silver y se deduplica por `article_id` entre
las ventanas recientes de las fechas observadas. No es la suma de los conteos
diarios ni exige que la publicación caiga dentro del rango de fechas solicitado.
Si no se puede resolver toda la evidencia, se devuelve `null` con advertencia.
El ejemplo de abajo es ilustrativo; el total anterior obtenido sumando días ya no
es válido y debe recalcularse con la evidencia disponible.

**Entrada:**
```json
{
  "ticker": "BOLSAA",
  "fecha_inicio": "2026-06-30",
  "fecha_fin": "2026-10-31"
}
```

**Salida:**
```json
{
  "ok": true,
  "status": "ok",
  "ticker": "BOLSAA",
  "periodo": {
    "fecha_inicio": "2026-06-30",
    "fecha_fin": "2026-10-31",
    "dias_con_datos": 16
  },
  "evolucion_sentimiento": {
    "promedio": 0.0245,
    "inicial": 0.125,
    "final": 0.0,
    "cambio_en_periodo": -0.125,
    "tendencia": "desfavorable / a la baja"
  },
  "resumen_volumen": {
    "promedio_volumen_ponderado": 1.842,
    "articulos_totales_observados": null,
    "criterio_conteo": "Artículos únicos de las ventanas recientes de las fechas observadas; no suma diaria ni publicaciones estrictamente dentro del periodo."
  },
  "serie_temporal": [
    {
      "fecha_utc": "2026-06-30",
      "sentimiento": 0.125,
      "volumen_ponderado": 1.9859,
      "articulos": 16,
      "interpretacion": "Balance positivo según el método de palabras clave."
    }
  ],
  "advertencias": [
    "Los días sin fila se conservan como huecos; n_articles no se suma entre ventanas superpuestas."
  ]
}
```

### Ejemplo 3: `comparar_noticias`

**Entrada:**
```json
{
  "tickers": ["AC", "BOLSAA", "UNKNOWN_TICKER"]
}
```

**Salida:**
```json
{
  "ok": true,
  "status": "ok",
  "fecha_consulta": null,
  "total_solicitados": 3,
  "total_encontrados": 2,
  "total_no_disponibles": 1,
  "ranking_por_sentimiento": [
    "BOLSAA",
    "AC"
  ],
  "ranking_por_volumen": [
    "BOLSAA",
    "AC"
  ],
  "comparativa": [
    {
      "ticker": "AC",
      "status": "ok",
      "sentimiento_general": 0.0,
      "volumen_informativo_ponderado": 1.7891,
      "cantidad_articulos": 30,
      "relevancia_promedio": 1.0
    },
    {
      "ticker": "BOLSAA",
      "status": "ok",
      "sentimiento_general": 0.0,
      "volumen_informativo_ponderado": 1.7891,
      "cantidad_articulos": 30,
      "relevancia_promedio": 1.0
    }
  ],
  "no_disponibles": [
    {
      "ticker": "UNKNOWN_TICKER",
      "status": "not_found",
      "motivo": "No hay una fila gold guardada para ticker=UNKNOWN_TICKER."
    }
  ]
}
```

### Ejemplo 4: `listar_tickers_noticias`

**Entrada:**
```json
{}
```

**Salida:**
```json
{
  "ok": true,
  "status": "ok",
  "total_tickers": 43,
  "total_temas": 1,
  "tickers_disponibles": [
    "AC",
    "ALFAA",
    "ALSEA",
    "AMXL",
    "AMZN",
    "ASURB",
    "BIMBOA",
    "BOLSAA",
    "CEMEXCPO"
  ],
  "temas_disponibles": [
    "trump"
  ],
  "rango_fechas_utc": {
    "first": "2026-06-30",
    "last": "2026-10-31"
  },
  "fuentes_detectadas": [
    "databursatil",
    "elfinanciero",
    "expansion",
    "gdelt",
    "google_news",
    "trump"
  ],
  "ultimo_calculo_utc": "2026-10-31T00:05:22.546824+00:00"
}
```

---

## 6. Comandos para iniciar el servidor

### 6.1. Ejecución estándar (stdio)

```bash
# Desde la raíz del repositorio
.venv/bin/python scripts/run_news_mcp.py
```

O como módulo Python:
```bash
PYTHONPATH=. .venv/bin/python -m src.mcp.server
```

### 6.2. Opciones de línea de comandos

```bash
.venv/bin/python scripts/run_news_mcp.py --help

Opciones:
  --transport {stdio,sse,streamable-http}  Protocolo de transporte (default: stdio)
  --news-root RUTA                        Ruta alternativa al directorio de datos
```

---

## 7. Cómo probarlo con un cliente o inspector MCP

### Opción A: MCP Inspector oficial (recomendado para desarrollo)

El inspector permite visualizar interactivamente el servidor, explorar las herramientas y probar llamadas:

```bash
npx @modelcontextprotocol/inspector .venv/bin/python scripts/run_news_mcp.py
```

### Opción B: Configuración en Claude Desktop

Agrega la configuración en `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "noticias_segmentadas": {
      "command": "/home/alexis/proyectosFinalMineria/stock-market-etl-ml/.venv/bin/python",
      "args": [
        "/home/alexis/proyectosFinalMineria/stock-market-etl-ml/scripts/run_news_mcp.py"
      ],
      "env": {
        "PYTHONPATH": "/home/alexis/proyectosFinalMineria/stock-market-etl-ml"
      }
    }
  }
}
```

### Opción C: Suite de pruebas automatizadas

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_news_mcp.py -v
```

---

## 8. Limitaciones actuales

1. **Solo lectura:** El servidor MCP no modifica ni escribe en los datasets Parquet ni en la base PostgreSQL.
2. **Dependencia de la actualización Gold:** Si la tabla `today_features.parquet` no ha sido generada por el pipeline diario de ingestión (`scripts/gcp_daily_job.py` o `build_news_medallion.py`), el servidor reportará estado `stale`.
3. **No realiza inferencias predictivas:** Un score positivo refleja balance de palabras clave según el modelo NLP configurado, pero no garantiza movimiento bursátil favorable.
4. **Ventana temporal:** El historial soporta consultas de hasta 366 días continuos por solicitud.
5. **Compatibilidad offline:** La lectura se realiza directamente sobre Parquet Gold local, por lo que no requiere conectividad activa a la base remota PostgreSQL para funcionar en entornos de desarrollo o pruebas.


## 9. Herramientas del dashboard y agente de noticias

El Observatorio descrito en [NEWS_DASHBOARD.md](NEWS_DASHBOARD.md) consume además:

- `consultar_tema_noticias(topic, fecha=None)`: consulta temas sin asignarlos a empresas.
- `consultar_evidencia_noticias(entity_type, entity_id, fecha, limit=10, offset=0)`:
  recupera artículos, enlaces y procedencia del resultado solicitado.
- `consultar_historial_noticias(entity_type, entity_id, inicio, fin)`: serie de una
  empresa o tema, preservando los estados y reglas del servicio semántico.

Estas herramientas delegan directamente en `NewsSemanticService`. El agente del
chat abre una sesión MCP real por stdio para invocarlas; las gráficas del panel
usan directamente la misma capa semántica a través de Flask.
