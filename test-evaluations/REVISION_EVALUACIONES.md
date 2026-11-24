# Revisión: lectura de evaluaciones y P&L de los agentes

- **Rama:** `test-evaluations`
- **Primera revisión:** 2026-11-07, sobre `main` en `f9a9899`
- **Segunda revisión:** 2026-11-12, después de traer el `main` de `NoSoyRo/stock-market-etl-ml` en `913f03b` (ver [Segunda revisión](#segunda-revisión-2026-11-12))
- **Alcance:** comprobar que los agentes de dominio `evaluacion` y `lab` (`src/agents_mcp/domains/`) leen bien sus archivos y muestran ganancias y pérdidas, y señalar archivos faltantes o con datos incorrectos.

## Cómo se probó

- Se ejecutaron las herramientas `query_evaluacion` (con `pregunta_tipo` = `resumen`, `ganancias` y `pnl`), `schema_evaluacion` y `query_lab`. Se llamó a sus handlers directamente, sin LLM, dentro de un contenedor con Python 3.11 y `requirements.txt` (pandas 3.1.1, numpy 2.4.6).
- Los archivos `data/eval/forecast_demo/llm_sim_{1m,2m,3m}.json` se recalcularon a partir de `daily_returns`: retorno, max drawdown, Sharpe anual y capital final.
- Se corrió `pytest` completo.

## Resultado general

Los archivos de simulación son consistentes por dentro, y el agente `evaluacion` los lee y reporta sus números correctamente. Lo que no hace el agente es mostrar ganancias y pérdidas en dinero. Además hay archivos faltantes o desactualizados, y algunos datos no cuadran entre sí.

### Números verificados (libro de $50 USD)

| Simulación | Periodo | Libro USD | Capital final | SPY | NAFTRAC | Libro MXN |
|---|---|---|---|---|---|---|
| 1m | 2026-09-30 → 10-30 | **+2.57%** | $51.28 | -0.91% | +4.15% | +7.16% |
| 2m | 2026-08-31 → 10-30 | **-3.16%** | $48.42 | -1.48% | -3.57% | +7.20% |
| 3m | 2026-07-30 → 10-30 | **-0.17%** | $49.92 | +1.89% | -4.11% | +7.90% |

- Retorno, drawdown, Sharpe y capital final recalculados desde `daily_returns` coinciden con lo guardado. En 2m el Sharpe difiere 0.001 por redondeo.
- `atribucion_usd` suma la ganancia o pérdida en dólares (por ejemplo, 1m: 3.17 − 1.69 − 0.20 = +1.28).
- La diferencia USD/MXN es igual para todos los activos dentro de cada periodo. Viene solo del tipo de cambio.
- Los archivos son UTF-8 válido.

## Problemas

### Muestra de ganancias y pérdidas

1. **`query_evaluacion` no muestra P&L en dinero.** Solo reporta porcentajes, Sharpe y Deflated Sharpe. No incluye `final_equity_usd`, `costos_usd`, `atribucion_usd` ni los resultados en MXN, aunque están en los archivos.
2. **Con `pregunta_tipo` = `ganancias` o `pnl` el agente no devuelve resultados.** Solo los tipos de `evaluacion.py:141` (`resumen`, `ultimo_backtest`, `sims`, `llm_sim`) incluyen las simulaciones. Cualquier otro tipo responde únicamente el conteo de la bitácora.
3. **Ninguna herramienta lee `lab.outcome`.** Por eso la ganancia o pérdida de los planes reales ya calificados no aparece en ningún agente.

### Archivos faltantes o desactualizados

4. **Falta `data/agent/book.json`.** `query_lab` responde "No encontré data/agent/book.json", así que no hay estado del libro (efectivo y posiciones) en local. Parece que solo lo genera el job de GCP (`scripts/gcp_daily_job.py:59`).
5. **`data/lab/entry_plans.jsonl` no corresponde a la regla vigente.**
   - Tiene 3 planes del 2026-10-09: INTC, FMX y GENTERA.
   - La regla que describe `query_lab` dice que el satélite son emisoras BMV o FIBRAs, pero INTC y FMX cotizan en EE.UU.
   - FMX y GENTERA tienen `p_sube` de 0.41 y 0.39.
   - Entraron el 2026-10-12 con horizonte de 20 días hábiles, así que todavía no hay resultado; se califican a partir del 2026-11-09.

### Datos que no cuadran

6. **N (número de pruebas) no coincide en tres lugares.**
   - Las simulaciones usan `n_trials = 15`. Ese valor sale de `--prior-trials` (`scripts/backtest_core_satellite.py:336`) y no está registrado en ningún archivo, porque las corridas `llm` no se anotan en la bitácora.
   - `data/eval/trial_log.jsonl` tiene 4 filas de la familia `libro50_24m`, pero solo 2 hashes distintos.
   - `query_evaluacion` reporta "4 filas" porque cuenta filas (`evaluacion.py:85`). En cambio, `src/ml/trial_stats.py:78` cuenta hashes distintos, que es el N correcto (2).
7. **"Deflated Sharpe 0.32" está escrito a mano** en `schema_evaluacion` (`evaluacion.py:68`). No sale de ningún archivo, y las filas de `trial_log.jsonl` no guardan `dsr`.
8. **Las simulaciones se presentan como resultados reales.**
   - `evaluacion.py:149` dice "Resultados reales del agente LLM de producción".
   - `lab.py:156` dice "Decisión real más reciente".
   - Ambos textos se refieren a los archivos `llm_sim_*.json`, que son simulaciones.
   - Además, las simulaciones 1m y 3m toman decisiones distintas el mismo día, 2026-09-30: NAFTRAC 80% contra NAFTRAC 60% + SPY 30%.
9. **Posición de un día en la simulación 3m.** PINFRA entra con 20% el 2026-07-30 y sale el 2026-07-31. Eso choca con `MIN_HOLD_DAYS = 5` de `src/agent/policy.py:11`. Falta confirmar si esa regla aplica a la política `llm` de `core_satellite`.

## Tests

122 pasan y 8 fallan.

- **7 en `tests/test_graph_ising.py`:** `ValueError: underlying array is read-only` en `np.fill_diagonal` (`src/ml/graph_ising.py:104`). `requirements.txt` pide `pandas>=2.1.0`, que hoy instala pandas 3.x, y en esa versión el array que sale del DataFrame es de solo lectura. Se arregla con una copia (`corr = corr.copy()`) o fijando `pandas<3`.
- **1 en `tests/test_lstm_seq.py:80`:** `predict_proba` devuelve `NaN` a propósito en las filas sin secuencia completa (`src/ml/lstm_seq.py:286`), pero el test exige que todos los valores estén en [0, 1]. El test quedó desactualizado.

## Siguientes pasos sugeridos

1. Agregar a `query_evaluacion` el P&L en dólares (capital final, costos, atribución), aceptar `ganancias`/`pnl` como tipos de pregunta y mostrar MXN.
2. Agregar una herramienta que lea `lab.outcome`, o su respaldo local, para los planes reales.
3. Contar N por hashes distintos en `query_evaluacion` y anotar las corridas `llm` en la bitácora.
4. Calcular el Deflated Sharpe del texto de `schema_evaluacion` a partir de los datos, o quitarlo.
5. Etiquetar las simulaciones como simulaciones.
6. Decidir si `entry_plans.jsonl` y `book.json` deben estar en el repo o generarse en local.

## Segunda revisión (2026-11-12)

Se repitió la misma prueba después de hacer merge del `main` del repo original (`913f03b`, 41 commits nuevos) en esta rama. Se usó el mismo entorno: contenedor con Python 3.11 y `requirements.txt` (pandas 3.1.1, numpy 2.4.6).

### Evaluaciones y P&L: sin cambios

El merge no tocó `src/agents_mcp/domains/evaluacion.py`, `lab.py` ni los archivos de `data/eval/` y `data/lab/`. Las salidas de las herramientas y los números recalculados son iguales a los de la primera revisión:

| Simulación | Retorno | Max DD | Sharpe | Capital final | Suma `atribucion_usd` | DSR |
|---|---|---|---|---|---|---|
| 1m | +2.57% | -1.62% | 2.584 | $51.28 | +1.28 | 0.161 |
| 2m | -3.16% | -4.40% | -1.909 (guardado -1.908) | $48.42 | -1.58 | 0.005 |
| 3m | -0.17% | -3.80% | -0.030 | $49.92 | -0.09 | 0.038 |

Los problemas 1 a 9 siguen abiertos:

- `ganancias` y `pnl` siguen devolviendo solo el conteo de la bitácora.
- `book.json` sigue faltando.
- `query_evaluacion` sigue contando 4 filas en lugar de 2 hashes.
- El Deflated Sharpe de 0.32 sigue escrito a mano.
- Las simulaciones se siguen presentando como resultados reales.
- La posición de PINFRA de un día sigue en la simulación 3m.

### Tests

230 pasan, 13 fallan y 1 se omite. La primera revisión tenía 122 pasando y 8 fallando; el merge agregó tests nuevos.

- **Siguen los 8 fallos anteriores:** 7 en `tests/test_graph_ising.py` (array de solo lectura con pandas 3) y 1 en `tests/test_lstm_seq.py:80` (`NaN` en `predict_proba`).
- **4 nuevos en `tests/test_news_mcp.py`, que dependen de la fecha:**
  - `test_ver_noticias_nonexistent_ticker`
  - `test_comparar_noticias_multiple_tickers`
  - `test_comparar_noticias_comma_separated_and_missing`
  - `test_server_with_isolated_mock_gold`
  - **Causa:** `src/semantic/service.py:98` responde `stale` sin datos cuando la tabla Gold del día no es de la fecha UTC actual. Los datos del repo son del 2026-10-30, y el mock del test usa 2026-10-31.
  - **Efecto:** `ver_noticias` responde `ok: true` con todas las métricas en `null`, incluso para un ticker que no existe (`src/mcp/server.py:92` trata `stale` como éxito). `comparar_noticias` responde `incompatible_metrics` porque la fecha y la ventana vienen vacías.
  - Esto es un error del servidor, no solo de los tests: un `stale` sin datos no debería reportarse como éxito. Los tests deberían fijar el reloj en lugar de depender del día en que se corren.
- **1 nuevo en `tests/test_news_medallion.py:224` (`test_empty_bronze_has_stable_schema`):** al leer el parquet, `fetched_at` regresa como `datetime64[ms, UTC]`, pero el frame esperado es `datetime64[ns, UTC]`. Viene del cambio de resolución de fechas en pandas 3. Se arregla igual que `graph_ising`, fijando `pandas<3` o normalizando la resolución en el test.
- **Omitido:** `tests/test_news_pyspark.py`, porque `pyspark` está en `requirements-spark.txt` y no en `requirements.txt`.
