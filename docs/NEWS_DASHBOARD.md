# Dashboard de noticias con agente de IA

El Observatorio reúne indicadores, histórico, evidencia y un chat con un agente especializado. Se ejecuta localmente en Flask y lee los Parquet existentes; no necesita una conexión SQL ni descarga noticias al consultar.

## Iniciar

Desde la raíz, con el entorno virtual del proyecto y sus dependencias instaladas:

```bash
venv/bin/python scripts/run_news_dashboard.py
```

Abrir `http://127.0.0.1:8050/`.

Opciones:

```bash
venv/bin/python scripts/run_news_dashboard.py --port 8051
venv/bin/python scripts/run_news_dashboard.py --news-root /ruta/a/data/news
```

También se integra en el dashboard anterior de algoritmos genéticos mediante `/news/` cuando se ejecuta `src/dashboard/app.py`. Ese dashboard anterior conserva sus requisitos de SQL Server; el comando independiente de arriba no los requiere.

## Interfaz

- Selección por empresa o tema; los temas no se atribuyen automáticamente a empresas.
- Consulta del día UTC o de una fecha histórica explícita.
- Sentimiento, artículos recientes, cambio frente a la ventana anterior, fuentes, ventana y método.
- Gráfica de sentimiento con periodos configurables. No une huecos de más de un día ni observaciones con ventanas o métodos diferentes.
- Artículos paginados, enlaces, resumen disponible y fechas. Los fallos de integridad se muestran explícitamente.
- Estado de datos ausentes, desactualizados o no evaluados, sin fabricar valores neutrales.
- Chat con conversación breve y enlaces recuperados como evidencia.

«Actualizar vista» vuelve a consultar las capas guardadas. No descarga ni recalcula noticias. Para actualizar las fuentes se usan los comandos de ingesta habituales.

## Arquitectura

```text
Panel visual → API Flask → NewsSemanticService → Gold / Silver
Chat         → API Flask → NewsAgent → cliente MCP stdio
                                      → servidor MCP → NewsSemanticService
```

Archivos principales:

| Archivo | Responsabilidad |
|---|---|
| `src/dashboard/news.py` | Blueprint, consultas JSON y control de solicitudes del chat. |
| `src/dashboard/templates/news.html` | Estructura de la página. |
| `src/dashboard/static/news.css` | Diseño adaptable a escritorio y móvil. |
| `src/dashboard/static/news.js` | Filtros, gráfica SVG, evidencia y conversación. |
| `src/agent/news.py` | Agente de noticias sobre la clase `Agent` existente. |
| `scripts/run_news_dashboard.py` | Arranque independiente sin SQL. |

## Configurar el chat

El panel funciona sin activar un modelo. El agente admite dos proveedores y elige
**Gemini por defecto**. Su adaptador está en `src/agent/news_chat.py`; el agente
inversor anterior conserva su conexión Anthropic independiente.

### Gemini

En `.env`:

```dotenv
NEWS_CHAT_PROVIDER=gemini
GEMINI_API_KEY=tu_clave_de_google
# Opcional: elegir un modelo habilitado en tu cuenta.
GEMINI_CHAT_MODEL=gemini-3.8-flash
```

Se reutiliza `GEMINI_API_KEY` si ya existe en el proyecto. `GEMINI_CHAT_MODEL`
configura únicamente este chat, sin cambiar los modelos de scoring de noticias.
El adaptador usa el endpoint de compatibilidad de Google con el formato común de
mensajes y llamadas a herramientas; conserva los metadatos devueltos en las llamadas
entre turnos. No necesita instalar el SDK de otro proveedor para usar Gemini.

Referencia de implementación: [compatibilidad de la API de Gemini](https://ai.google.dev/gemini-api/docs/openai).

### Anthropic como alternativa

Para cambiar de proveedor, editar `.env` y reiniciar el dashboard:

```dotenv
NEWS_CHAT_PROVIDER=anthropic
ANTHROPIC_API_KEY=tu_clave_de_anthropic
# Opcional: indicar un modelo habilitado en tu cuenta.
# ANTHROPIC_MODEL=tu_modelo
```

Este modo reutiliza el adaptador existente en `src/agent/chat.py`. Las dos claves
pueden coexistir, pero se utiliza solo la del proveedor seleccionado. No hay
fallback automático ni llamadas duplicadas a dos proveedores. Un error de API,
cuota o credenciales se informa sin intentar consumir la cuenta alternativa.

La selección es del servidor; no se envían claves al navegador ni se permite elegir
un proveedor arbitrario desde una solicitud del chat. El indicador de disponibilidad
muestra el proveedor configurado y si su clave está presente, pero no valida por
adelantado cuotas, permisos o disponibilidad del modelo. Sin clave, la interfaz
indica que la IA está pendiente de configuración y no simula respuestas.

El dashboard, el MCP, la capa semántica y los Parquet son los mismos para ambos
proveedores. No se necesita la clave de DataBursátil para leerlos.

Por cada consulta de chat se abre una sesión MCP por `stdio` con el intérprete del mismo entorno virtual y se cierra al terminar. El servidor hereda la raíz de datos configurada. No requiere iniciar manualmente otro servidor ni abrir un puerto MCP.

Herramientas disponibles:

- `ver_noticias`, `resumen_noticias`, `comparar_noticias`, `listar_tickers_noticias`.
- `consultar_tema_noticias`: indicadores para `trump`, `market` u otro tema disponible.
- `consultar_evidencia_noticias`: artículos que respaldan el corte solicitado.
- `consultar_historial_noticias`: serie de una empresa o tema.

El agente tiene un máximo de seis rondas y seis llamadas por ronda. El backend limita la pregunta y el historial, admite un análisis simultáneo por proceso y no acepta mensajes de sistema enviados por el navegador. Un fallo del modelo o MCP produce un error explícito. Los titulares y resúmenes se tratan como datos no confiables, no como instrucciones.

## Ejecución y alcance

El servidor inicia en loopback con debug desactivado. Es una herramienta local de análisis; no implementa autenticación multiusuario ni cuotas persistentes. Para un despliegue accesible a otros usuarios hace falta un servidor WSGI, autenticación y controles de acceso y consumo. Esta implementación no publica los datos en un servicio externo.

La aplicación consulta gold del día según UTC y conserva sus reglas de disponibilidad. Los resultados dependen de la cobertura real. La IA puede equivocarse: las fuentes enlazadas permiten verificar sus respuestas. No coloca órdenes ni utiliza las reglas de cartera del agente inversor.

## Pruebas

```bash
venv/bin/python -m pytest tests/test_news_chat.py tests/test_news_dashboard.py tests/test_news_semantic.py tests/test_news_mcp.py tests/test_investor_agent.py -q
```

La prueba de conversación usa un cliente y un servidor MCP reales por `stdio` con un modelo simulado, sin consumir la API externa. Se verifican rutas, validación de entradas, ausencia de clave, lectura sin modificaciones, paso de contexto y enlaces de evidencia. Una validación con el modelo real requiere una clave configurada y genera consumo en esa cuenta.

## Diagnóstico de errores del proveedor

Un `httpx.ReadTimeout` al llamar a Google significa que el proveedor o la ruta de
red no entregó respuesta antes del límite; el grupo de errores de AnyIO/MCP puede
envolver esa excepción sin que el fallo provenga de los Parquet o herramientas.
El backend extrae la causa y responde `provider_timeout` (HTTP 504), con un mensaje
claro en el chat. No reintenta automáticamente ni cambia de proveedor.

Gemini utiliza `reasoning_effort=low` por defecto para evitar el presupuesto de
razonamiento predeterminado del modelo en consultas del panel. Se puede ajustar:

```dotenv
GEMINI_CHAT_REASONING_EFFORT=low
GEMINI_CHAT_TIMEOUT_SECONDS=90
```

El timeout de lectura admite entre 10 y 120 segundos. La conexión tiene un límite
de 10 segundos. Reducir razonamiento puede mejorar latencia, pero no garantiza
resolver un fallo de red o del proveedor. Reiniciar el dashboard tras cambiar `.env`.

Se distinguen `provider_connection_error`, `provider_auth_error`,
`provider_model_error` y `provider_rate_limit` para conexión, permisos, modelo y
cuota. Los cuerpos de error HTTP de Gemini no se muestran ni se registran para
no exponer entradas o credenciales.
