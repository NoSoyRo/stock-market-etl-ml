# MCP server standalone — stock-market

Servidor MCP real (protocolo oficial Model Context Protocol, transporte
`streamable-http`) que envuelve los agentes de dominio ya existentes en
`src/agents_mcp/domains/` para que una app de Claude (claude.ai / Claude
Desktop, como "custom connector") se conecte por URL y les haga preguntas
en lenguaje natural sobre el proyecto stock-market.

Esto es distinto del sistema multiagente in-process
(`src/agents_mcp/orchestrator.py` + `create_sdk_mcp_server`): ese corre
dentro de un proceso Python que ya está llamando a Claude. Este servidor
expone las mismas capacidades vía HTTP, para que cualquier cliente MCP
(incluido claude.ai) se conecte desde fuera.

## Tools expuestas

Una tool por dominio, cada una recibe `pregunta: str` en lenguaje natural:

- `preguntar_mercado` — precios OHLCV, FX.
- `preguntar_fundamentales` — estados financieros, ratios, reportes.
- `preguntar_noticias` — noticias y sentimiento de mercado.
- `preguntar_features_ml` — features, modelos ML, pronóstico y portafolio
  (pronóstico/portafolio dependen de que el dominio ya tenga esas tools —
  ver TODO en `server.py`).
- `preguntar_lab` — experimentos / esquema Postgres de lab.
- `preguntar_evaluacion` — evaluación de modelos / métricas A-B.

Cada tool importa su `_agent` de forma perezosa (dentro de la función), así
que si un dominio tiene un error de import, solo falla esa tool al
invocarse — no tumba el servidor completo.

## Correr local

```bash
cd /Users/rod/Projects/stock-market
source venv/bin/activate
export MCP_AUTH_TOKEN="un-secreto-cualquiera-para-pruebas"
python -m mcp_server.server
```

Por default escucha en `0.0.0.0:${PORT:-8080}` (8080 = default de Cloud Run).
El endpoint MCP queda en `http://localhost:8080/mcp`.

## Autenticación

Todas las requests (salvo `/health` y `/healthz`) deben incluir:

```
Authorization: Bearer <MCP_AUTH_TOKEN>
```

El token correcto viene de la variable de entorno `MCP_AUTH_TOKEN`:

- Si **no está seteada**: el servidor arranca igual, pero usa un token
  default claramente marcado como inseguro
  (`INSECURE-DEFAULT-TOKEN-CAMBIAME`, ver `mcp_server/auth.py`) e imprime un
  `WARNING` en stderr. Sirve para no trabar pruebas rápidas locales.
- Para forzar que el servidor se niegue a arrancar sin un token real
  (recomendado antes de exponerlo fuera de tu laptop), exporta además
  `MCP_REQUIRE_AUTH_TOKEN=1`.

Una request sin el header correcto recibe `401 Unauthorized` antes de
llegar al protocolo MCP (el middleware vive en `mcp_server/auth.py` y se
monta sobre la app ASGI de Starlette que regresa
`mcp.streamable_http_app()`).

## Probar manualmente

El protocolo MCP sobre `streamable-http` usa JSON-RPC 2.0 y espera
`Accept: application/json, text/event-stream`, así que un `curl` plano con
un body cualquiera no basta para una sesión MCP completa, pero sirve para
confirmar que la autenticación responde:

```bash
# Sin token -> 401
curl -i http://localhost:8080/mcp

# Con token correcto -> pasa la autenticación (puede dar 400 si el body
# no es un JSON-RPC válido, pero ya no es 401)
curl -i -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer un-secreto-cualquiera-para-pruebas" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "curl-test", "version": "0"}}}'
```

Para una prueba real end-to-end (listar tools, llamarlas, etc.), usa un
cliente MCP de verdad en vez de curl plano:

```bash
# Inspector interactivo del SDK oficial de mcp
mcp dev mcp_server/server.py
```

o configura este servidor como "custom connector" en Claude Desktop /
claude.ai apuntando a `http://<host>:<port>/mcp` con el header
`Authorization: Bearer <token>`.

## Nota sobre la versión del SDK `mcp`

El repo trae instalado `mcp==2.3.0` (viene como dependencia de
`claude-agent-sdk`, que ya usan los agentes in-process). En esa serie,
`mcp.server.fastmcp.FastMCP` fue renombrado a
`mcp.server.mcpserver.MCPServer`. `mcp_server/server.py` usa `MCPServer`
por eso — la API relevante (`@mcp.tool()`, `mcp.streamable_http_app()`,
`mcp.run(...)`) es equivalente a la de `FastMCP` en `mcp<2`. Si el venv
baja de versión a `mcp<2`, basta cambiar el import de vuelta a
`from mcp.server.fastmcp import FastMCP`.

## Capa semántica de evaluación de múltiples estrategias

El servidor incorpora siete herramientas adicionales para consultar
directamente el producto Gold de evaluación y riesgo.

Estas herramientas no requieren invocar un modelo de lenguaje,
volver a entrenar modelos ni ejecutar nuevos backtests.

### Arquitectura de integración

    Resultados históricos de estrategias
                    |
                    v
            Bronze: integridad
                    |
                    v
            Silver: validación
                    |
                    v
            Gold: consolidación
                    |
                    v
            Capa semántica
               /         \
              v           v
       Agente interno   MCP externo
              |
              v
          Orquestador

### Producto de datos

El producto consultado es:

    data/eval/medallion/gold/evaluacion_multiples_estrategias.json

Para regenerarlo desde los resultados existentes, ejecutar
desde la raíz del repositorio:

    python -m scripts.medallion_estrategias

La generación requiere que los archivos originales y homologados
de las estrategias estén disponibles.

### Herramientas disponibles

| Herramienta | Argumentos | Descripción |
|---|---|---|
| resumen_estrategias | Ninguno | Resumen general |
| consultar_estrategia | nombre | Métricas y configuración |
| comparar_estrategias | criterio | Ranking de estrategias |
| consultar_benchmark | Ninguno | Métricas de NAFTRAC |
| consultar_validacion | nombre | Criterios de aprobación |
| consultar_calidad | Ninguno | Calidad y trazabilidad |
| consultar_decision | Ninguno | Decisión global |

### Identificadores de estrategias

- topk_2
- topk_5
- topk_10
- topk_5_ranking
- sma_50_200
- rsi_14

### Criterios de comparación

- rendimiento_neto
- sharpe_neto
- cumplimiento

### Ejemplos de consultas

Resumen general:

    {"name": "resumen_estrategias", "arguments": {}}

Consultar una estrategia:

    {"name": "consultar_estrategia", "arguments": {"nombre": "sma_50_200"}}

Comparar por Sharpe neto:

    {"name": "comparar_estrategias", "arguments": {"criterio": "sharpe_neto"}}

Consultar la validación de RSI:

    {"name": "consultar_validacion", "arguments": {"nombre": "rsi_14"}}

Consultar la decisión global:

    {"name": "consultar_decision", "arguments": {}}

Estos objetos corresponden a los parámetros de una llamada
MCP tools/call, no a solicitudes JSON-RPC completas.

### Estructura de las respuestas

Las consultas exitosas incluyen:

- Identificador de consulta.
- Nombre del producto de datos.
- Versión del esquema.
- Fecha de generación.
- Periodo de evaluación.
- Configuración común.
- Fuente de información.
- Resultado solicitado.
- Aviso sobre las limitaciones financieras.

### Integridad y calidad

Antes de responder, la capa semántica comprueba:

- Existencia y estructura del producto Gold.
- Integridad de los archivos Bronze y Silver.
- Hashes SHA-256 de los archivos correspondientes.
- Consistencia entre métricas, rankings y decisiones.
- Correspondencia del contenido Gold con su reconstrucción
  a partir de Silver.

Si la información es inválida, la consulta debe fallar en lugar
de proporcionar resultados inconsistentes.

Las consultas son de solo lectura.

### Integración con los agentes existentes

Las siete herramientas se incorporan al agente de Evaluación,
conservando sus tres herramientas originales.

El orquestador puede acceder a ellas mediante el agente
de Evaluación existente.

El servidor MCP externo también permite invocarlas directamente,
sin pasar por un modelo de lenguaje.

### Resultado histórico de referencia

- Estrategias evaluadas: 6.
- Estrategias aprobadas: 0.
- Estrategias no aprobadas: 6.
- Decisión global: NO_BET_LIVE.

SMA presenta el mayor rendimiento neto y RSI presenta el mayor
Sharpe neto entre las estrategias evaluadas.

Ninguna estrategia supera el rendimiento neto del benchmark.

Un producto de datos válido no implica que una estrategia
esté aprobada para operar con capital real.

### Pruebas automatizadas

Desde la raíz del repositorio:

    python -m pytest -v \
        tests/test_medallion_estrategias.py \
        tests/test_evaluacion_semantica.py \
        tests/test_mcp_estrategias.py \
        tests/test_agents_mcp_evaluacion_lab.py

Estas pruebas no requieren consumir créditos de una API de IA.

### Alcance de la integración

Se ha comprobado localmente:

- Registro de herramientas.
- Consultas directas al producto Gold.
- Integración con el agente de Evaluación.
- Registro y permisos del orquestador.
- Consultas MCP mediante HTTP.
- Autenticación mediante Bearer Token.
- Manejo de errores.
- Detección de alteraciones en el producto de datos.

No se ha validado un despliegue en producción ni una conversación
completa mediante un modelo de lenguaje.

Para pruebas locales, evitar exponer el puerto del servidor a
Internet y utilizar un token de autenticación seguro.

### Documentación complementaria

Consultar [Evaluación y riesgo](../docs/evaluacion_riesgo.md)
para conocer la metodología, resultados, criterios de aprobación
y limitaciones de las estrategias.
