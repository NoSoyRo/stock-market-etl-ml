# Agente hibrido de validacion de estrategias

## Objetivo

Implementar un agente que formule y responda preguntas de
validacion sobre estrategias financieras a partir de los
resultados de un backtest.

El sistema combina reglas cuantitativas de Python con
inteligencia artificial mediante Gemini.

## Componentes

### StrategyValidator

Realiza ocho validaciones deterministas:

1. Sharpe neto positivo.
2. Retorno neto positivo.
3. Drawdown dentro del limite configurado.
4. Presencia de comisiones y deslizamiento.
5. Rendimiento superior al benchmark NAFTRAC.
6. Separacion temporal entre entrenamiento y prueba.
7. Cantidad suficiente de periodos evaluados.
8. AUC dentro del rango valido.

Cada validacion incluye pregunta, estado y evidencia.

Los estados posibles son:
- cumple
- no_cumple
- pendiente

La estrategia se aprueba solamente si las ocho
validaciones cumplen.

### StrategyAI

Utiliza Gemini para generar tres preguntas adicionales
sobre riesgos, metodologia y robustez.

Cada pregunta incluye:
- pregunta
- respuesta
- estado
- evidencia

Si no existe informacion suficiente, Gemini debe
responder que no puede determinarse y utilizar
el estado pendiente.

Las respuestas de Gemini son orientativas y no
modifican la decision cuantitativa.

## Ejecucion

Desde la raiz del repositorio:

### Sin inteligencia artificial

    python scripts/validate_strategies.py

### Con Gemini

Configurar previamente la variable GEMINI_API_KEY.

    python scripts/validate_strategies.py --ai

### Modelo personalizado

    python scripts/validate_strategies.py --ai --ai-model gemini-3.8-flash

### Limite de drawdown personalizado

    python scripts/validate_strategies.py --max-drawdown 0.25

## Entrada y salida

Entrada:
data/eval/backtest_topk.json

Salida:
data/eval/strategy_validation.json

El reporte contiene las validaciones cuantitativas,
sus evidencias, el resumen y la decision final.

Al activar --ai se agrega validacion_ia con las
preguntas, respuestas, estados y evidencias de Gemini.

## Manejo de errores

El agente puede funcionar sin Gemini.

Si no existe GEMINI_API_KEY, la validacion
cuantitativa se mantiene disponible.

Ante errores temporales 429 o 503 se realizan
hasta tres intentos.

Si Gemini falla, se conserva la decision
cuantitativa y se informa el error.

## Pruebas unitarias

Ejecutar:

    python -m unittest discover -s tests -p "test_strategy_*.py" -v

Se implementaron 12 pruebas:

- 8 pruebas para StrategyValidator.
- 4 pruebas para StrategyAI.

Las pruebas de inteligencia artificial utilizan
respuestas simuladas y no requieren consumir la API.

## Resultados de ejemplo

Con data/eval/backtest_topk.json:

- Validaciones que cumplen: 4.
- Validaciones que no cumplen: 4.
- Validaciones pendientes: 0.
- Estrategia aprobada: False.

Gemini genero tres preguntas adicionales con
sus respectivas respuestas y evidencias.

## Limitaciones

La validacion no garantiza rentabilidad futura.

La separacion de fechas no demuestra por si sola
la ausencia de fuga de informacion.

Un AUC dentro del rango valido no implica
necesariamente buen poder predictivo.

Las respuestas de Gemini deben revisarse y no
constituyen recomendaciones de inversion.

Nunca almacenar claves de API en el repositorio.
