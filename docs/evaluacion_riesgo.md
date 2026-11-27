# Evaluación y riesgo: múltiples estrategias de inversión

## 1. Objetivo

Evaluar y comparar múltiples estrategias de inversión utilizando datos
históricos de mercado, condiciones homogéneas de simulación, métricas
financieras, controles de calidad y criterios objetivos de aprobación.

Los resultados se integran en una arquitectura Medallion y se exponen
mediante una capa semántica basada en Model Context Protocol (MCP).

## 2. Fuentes de información

- Yahoo Finance: precios históricos OHLCV.
- Banco de México (SIE): tasas de interés históricas.
- NAFTRACISHRS.MX: instrumento de referencia para el benchmark.
- Archivos Parquet y JSON almacenados localmente.

Los datos de mercado se someten a controles de calidad para detectar
precios inválidos, inconsistencias OHLCV, registros faltantes y
volúmenes no válidos para la ejecución de operaciones.

Las tasas de interés se incorporan con un desfase temporal para reducir
el riesgo de utilizar información que todavía no estaba disponible.

## 3. Estrategias evaluadas

| Estrategia | Tipo | Descripción |
|---|---|---|
| Top-K 2 | Machine Learning | Selección de hasta dos instrumentos |
| Top-K 5 | Machine Learning | Selección de hasta cinco instrumentos |
| Top-K 10 | Machine Learning | Selección de hasta diez instrumentos |
| Top-K 5 Ranking | Machine Learning | Selección exploratoria por clasificación |
| SMA 50/200 | Tradicional | Cruce de medias móviles |
| RSI 14 | Tradicional | Indicador de fuerza relativa |

Las estrategias Top-K originales utilizan un umbral de probabilidad
de 0.56.

La variante Top-K 5 Ranking elimina ese umbral con fines exploratorios.
No constituye una validación independiente ni una mejora demostrada
del modelo.

## 4. Metodología

### Machine Learning

- Separación cronológica de entrenamiento y evaluación.
- Purga de observaciones cuyos horizontes cruzan la frontera temporal.
- Selección de variables utilizando información de entrenamiento.
- Combinación de modelos de regresión logística y LightGBM.
- Calibración de probabilidades.
- Evaluación de capacidad predictiva mediante AUC.

### Estrategias tradicionales

- SMA con ventanas de 50 y 200 observaciones.
- RSI con periodo de 14 observaciones.
- Umbrales RSI de 30 y 70.
- Señales calculadas con información disponible al cierre.
- Ejecución a partir de la apertura de la siguiente sesión válida,
  conforme a las reglas de cada simulación.
- Distribución inicial equitativa del capital entre instrumentos.

## 5. Condiciones comunes de evaluación

| Parámetro | Valor |
|---|---|
| Capital inicial | 100,000 MXN |
| Inicio de evaluación | 2023-05-10 |
| Fin de evaluación | 2026-10-30 |
| Sesiones de evaluación | 876 |
| Frecuencia de métricas | Diaria |
| Periodos de anualización | 252 |
| Comisión por operación | 0.25% |
| Deslizamiento por operación | 0.05% |
| Costo total de entrada y salida | 0.60% |
| Drawdown máximo permitido | 30% |
| Benchmark | NAFTRAC Buy and Hold |

Las estrategias se homologan utilizando el mismo calendario,
capital inicial, moneda, costos y referencia de mercado.

## 6. Métricas financieras

- Rendimiento bruto.
- Rendimiento neto.
- Ratio de Sharpe bruto y neto.
- Maximum Drawdown.
- Ratio de Calmar.
- Proporción de rendimientos positivos.
- Capital final.
- Rendimiento excedente respecto al benchmark.
- AUC para estrategias de Machine Learning.

El ratio de Sharpe convencional no debe confundirse con el
Deflated Sharpe Ratio utilizado en otros módulos del proyecto.

## 7. Resultados históricos

| Estrategia | Rendimiento neto | Sharpe neto | Max Drawdown | Capital final MXN |
|---|---:|---:|---:|---:|
| SMA 50/200 | 12.46% | 0.4213 | -8.05% | 112,463.84 |
| RSI 14 | 11.03% | 0.5008 | -7.57% | 111,029.44 |
| Top-K 2 | 0.00% | 0.0000 | 0.00% | 100,000.00 |
| Top-K 5 | 0.00% | 0.0000 | 0.00% | 100,000.00 |
| Top-K 10 | 0.00% | 0.0000 | 0.00% | 100,000.00 |
| Top-K 5 Ranking | -45.25% | -0.8734 | -52.41% | 54,745.13 |
| Benchmark NAFTRAC | 24.54% | 0.4803 | -16.23% | 124,535.33 |

Las cifras corresponden a simulaciones históricas, no a ganancias
obtenidas mediante operaciones reales.

SMA presenta el mayor rendimiento neto entre las estrategias evaluadas.

RSI presenta el mayor ratio de Sharpe neto entre las estrategias
evaluadas.

Ninguna estrategia supera el rendimiento neto del benchmark.

Las tres estrategias Top-K originales permanecieron en efectivo al
no encontrar oportunidades que cumplieran sus condiciones de entrada.

La ausencia de operaciones no constituye evidencia de rentabilidad.

## 8. Criterios de aprobación

Las estrategias se someten a los siguientes criterios:

1. Sharpe neto positivo.
2. Rendimiento neto positivo.
3. Drawdown dentro del límite establecido.
4. Costos de operación considerados.
5. Rendimiento superior al benchmark.
6. Cantidad suficiente de periodos de evaluación.
7. Separación temporal válida, cuando corresponda.
8. AUC superior a 0.50, cuando corresponda.

Los criterios relacionados con entrenamiento y AUC no se aplican
a estrategias tradicionales.

Las estrategias sin operaciones y las variantes exploratorias
no se consideran aprobadas.

### Resultado de validación

- Estrategias evaluadas: 6.
- Estrategias aprobadas: 0.
- Estrategias no aprobadas: 6.
- Decisión global: NO_BET_LIVE.

La decisión evita recomendar la ejecución con capital real de
estrategias que no cumplen los criterios establecidos.

## 9. Arquitectura Medallion

### Bronze

Conserva los archivos originales de evaluación y genera un manifiesto
con su procedencia, tamaño y hash SHA-256.

### Silver

Valida:

- Integridad de los archivos Bronze.
- Calendarios de evaluación.
- Capital inicial y final.
- Curvas de capital y rendimientos diarios.
- Costos de operación.
- Consistencia del benchmark.
- Métricas y resultados de comparación.
- Aplicabilidad de los criterios de aprobación.

Genera un reporte de calidad y registra las advertencias detectadas.

### Gold

Consolida las estrategias, métricas, rankings, benchmark, criterios
de validación, decisión global y trazabilidad.

Producto generado:

data/eval/medallion/gold/evaluacion_multiples_estrategias.json

El producto se reconstruye a partir de Silver para comprobar su
consistencia antes de responder consultas semánticas.

Un producto Gold válido no implica que sus estrategias estén aprobadas.

## 10. Capa semántica MCP

Las herramientas semánticas consultan el producto Gold validado.

| Herramienta | Función |
|---|---|
| resumen_estrategias | Resumen general |
| consultar_estrategia | Información de una estrategia |
| comparar_estrategias | Rankings comparativos |
| consultar_benchmark | Información de NAFTRAC |
| consultar_validacion | Criterios y motivos de aprobación o rechazo |
| consultar_calidad | Calidad y trazabilidad |
| consultar_decision | Decisión global |

Las herramientas se integran en el agente de Evaluación y en el
servidor MCP externo.

Las consultas son de solo lectura y no requieren volver a entrenar
los modelos.

La capa semántica rechaza información inconsistente y evita sustituir
silenciosamente el producto Gold por resultados anteriores.

## 11. Archivos principales

- scripts/backtest_topk.py
- scripts/backtest_traditional.py
- scripts/backtest_benchmark.py
- scripts/homologate_topk.py
- scripts/consolidate_strategies.py
- scripts/validate_strategies.py
- scripts/medallion_estrategias.py
- scripts/repair_market_data.py
- src/data/market_quality.py
- src/agent/trade_execution.py
- src/agent/traditional_backtest.py
- src/agent/traditional_portfolio.py
- src/agents_mcp/domains/evaluacion_semantica.py
- src/agents_mcp/domains/evaluacion_tools.py
- src/agents_mcp/domains/evaluacion.py
- mcp_server/server.py

## 12. Pruebas automatizadas

Las pruebas comprueban:

- Validación y comparación de estrategias.
- Ejecución de estrategias tradicionales.
- Homologación de resultados.
- Integridad Bronze, Silver y Gold.
- Detección de alteraciones de métricas.
- Consultas de la capa semántica.
- Registro e integración de herramientas MCP.
- Manejo controlado de errores.

Las pruebas locales de integración no requieren consumir créditos
de una API de inteligencia artificial.

## 13. Limitaciones

- Los resultados históricos no garantizan rendimientos futuros.
- El universo de instrumentos puede presentar sesgo de supervivencia.
- No se incluyen dividendos ni intereses sobre efectivo.
- Los costos de operación son supuestos de simulación.
- Las posiciones abiertas se valoran con los precios disponibles.
- Existen observaciones faltantes y valoraciones desactualizadas.
- El AUC del modelo es cercano a 0.50.
- La variante Ranking requiere validación independiente.
- Los hashes permiten comprobar consistencia, pero no protegen
  contra la modificación coordinada de todas las fuentes.
- No se ha validado un despliegue en producción.
- No se ha validado una conversación completa mediante un LLM.

## 14. Conclusión

La integración permite evaluar estrategias bajo condiciones
comparables, identificar riesgos y comunicar resultados mediante
una capa semántica respaldada por datos verificables.

Aunque las estrategias tradicionales presentan rendimientos
positivos, ninguna supera el rendimiento del benchmark.

Por lo tanto, se conserva la decisión NO_BET_LIVE hasta disponer
de evidencia suficiente para justificar una estrategia de inversión.

## 15. Integración con el primer avance de Evaluación y Riesgo

El desarrollo de múltiples estrategias complementa el producto
de datos presentado en el primer avance del equipo.

Se conservan los siguientes componentes:

- Esquema SQL: `sql/04_eval_schema.sql`.
- Contrato de datos: `contracts/evaluacion_riesgo.yaml`.
- Exportador inicial: `scripts/export_evaluacion.py`.
- Pruebas del exportador: `tests/test_evaluacion.py`.

La documentación original se conserva íntegramente en:

[evaluacion_riesgo_primer_avance.md](evaluacion_riesgo_primer_avance.md).

El exportador inicial procesa los resultados históricos del
backtest original. Sus métricas no deben confundirse con las
obtenidas en la evaluación de múltiples estrategias.

El producto Gold de múltiples estrategias constituye una
ampliación del alcance de evaluación mediante condiciones
homogéneas, controles de calidad y consultas semánticas MCP.

La integración directa del nuevo producto Gold con el esquema
SQL Server no ha sido implementada ni validada.

Para reproducir los resultados de múltiples estrategias,
consultar:

[REPRODUCIBILIDAD_EVALUACION.md](REPRODUCIBILIDAD_EVALUACION.md).

## 16. Verificación final de integración

La versión integrada del proyecto fue verificada sobre el
commit `237df46` en dos entornos de ejecución independientes.

| Verificación | Resultado |
|---|---|
| Pruebas automatizadas seleccionadas | 144/144 |
| Capas Medallion de Evaluación | 3/3 |
| Entradas Bronze verificadas mediante SHA-256 | 10/10 |
| Estrategias evaluadas | 6 |
| Errores de calidad | 0 |
| Advertencias de calidad | 9 |
| Benchmark validado | Sí |
| Funciones semánticas verificadas | 7/7 |
| Herramientas MCP registradas | 14/14 |
| Herramientas de Evaluación verificadas mediante HTTP | 7/7 |
| Igualdad entre respuestas HTTP y directas | 7/7 |
| Integridad de Gold después de las consultas | Verificada |

Adicionalmente, se ejecutó y auditó el ETL de Mercado en
una copia aislada, incluyendo Bronze, Silver, Gold y
la base de datos SQLite.

Las pruebas HTTP comprobaron la autenticación Bearer,
la inicialización del protocolo MCP y la ejecución de
las siete herramientas de Evaluación.

La verificación corresponde al alcance seleccionado
de Evaluación e integración. No constituye una validación
completa de todos los dominios del repositorio ni de un
despliegue en producción.

La decisión financiera permanece en `NO_BET_LIVE`.

Los procedimientos, resultados detallados y limitaciones
se encuentran en:

[REPRODUCIBILIDAD_EVALUACION.md](REPRODUCIBILIDAD_EVALUACION.md).
