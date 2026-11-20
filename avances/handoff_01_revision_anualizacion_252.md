# Handoff 01 · Revisar las anualizaciones con √252 que quedan

Para: quien lo implemente · Tamaño: S–M (medio día) · Depende de: nada (parte de `main` con el PR #5 ya mergeado)

## 1. Contexto en 1 minuto

En el backtest top-K cada periodo dura 5 días hábiles, pero `Metrics.portfolio_metrics_from_returns` ignoraba `periods_per_year` y anualizaba con `√252` (retornos diarios). Resultado: todo Sharpe salía inflado ×2.24 y todo Calmar ×5. Eso ya se arregló en `src/ml/shared/evaluators/metrics.py` (commit "fix(M1a)"; tests en `tests/test_metrics_annualization.py`).

Regla para decidir si un Sharpe está bien anualizado:

> `Sharpe anual = media(r) / std(r) × √(periodos por año)`, y el "periodo" debe ser **la frecuencia de los retornos que entran**: diario → 252; semanal → 52; periodos de 5 días hábiles → 252/5 = 50.4.

Quedan otros sitios con `252` en el código. Esta tarea es **auditarlos uno por uno**, corregir los que estén mal y dejar documentados los que están bien.

## 2. Qué hay que revisar (ya auditado a nivel de lectura; confirma antes de tocar)

| # | Sitio | Qué recibe | Veredicto previo | Qué hacer |
|---|---|---|---|---|
| 1 | `src/ml/shared/evaluators/backtester.py:192` (`Metrics.calculate_dual_metrics(gross, net)` sin `periods_per_year`) | Retornos **diarios** consecutivos (`_simulate_directional_returns`: `np.diff(close)/close[:-1]`, target `close`) | √252 es **correcto** | Hacerlo explícito: `periods_per_year=252` + comentario de por qué, y un test que lo fije. No cambia números. |
| 2 | `src/ml/shared/data_loaders/feature_engineering.py:527-546` (`calculate_sharpe_ratio`) | Lo que le pasen; ya respeta `periods_per_year` | Correcto; **nadie lo llama** (grep) | Nada, o decidir si es código muerto. |
| 3 | `src/ml/shared/optimizers/fitness_evaluator.py:140` (`np.sqrt(252) * returns.mean()/returns.std()`, con `returns = np.diff(y_train)/y_train[:-1]`) | `y_train` viene de `splits['y_train']` en `src/ml/pipelines/train_pipeline.py`, tras `_preprocess` (probablemente **escalado** y/o secuencias de varios pasos) | **Dudoso**: no es solo un problema de 252, ese "retorno" puede no ser un retorno | Confirma qué es `y_train` cuando llega ahí. Si está escalado o es 2-D, el Sharpe no tiene sentido financiero: propón (a) quitarlo del fitness del GA, o (b) pasar retornos reales. Si no estás seguro, no lo cambies: déjalo documentado en la sección "Resultado" y pregunta. |
| 4 | `scripts/run_meta.py:100` (`rets_net.mean()/std × √(252/HORIZON)`) | Retornos **por operación** (cada fila = una señal; ~3,100 operaciones en ~3 años, varias por semana, solapadas) | **Mal anualizado**: tratar cada operación como un periodo de 5 días no refleja cuántas operaciones hay al año | Reportar el Sharpe **por operación sin anualizar** (y decirlo en el nombre de la clave), o anualizar con operaciones/año reales. Si el script se vuelve a correr, `data/meta_results.json` cambia: no lo regeneres sin avisar. |
| 5 | `scripts/run_purged_cv.py:124` (mismo patrón por señal × √(252/HORIZON)) | Retornos de señales solapadas, varias por fecha | Mismo problema que el 4 | `dsr_from_fold_sharpes` (línea ~205) des-anualiza dividiendo por √(252/horizonte) y **depende** de esta convención: si cambias la anualización de la línea 124, ajusta también esa función y sus tests (`tests/test_dsr_caller.py`). |
| 6 | `Metrics.calculate_all_metrics` (en `metrics.py`) | — | No expone `periods_per_year`; usa 252 | Añadir parámetro opcional con default 252 (sin cambiar el comportamiento actual). |

Ya correctos (solo confirmar, sin cambios): `src/agent/bmv_report.py:290` y `src/agent/core_satellite.py:145` (vol anualizada con retornos diarios), `src/ml/trial_stats.py` (`deflated_sharpe` recibe `daily_returns`; el caller `scripts/backtest_core_satellite.py` pasa diarios), `src/ml/vol_target.py` (`TRADING_DAYS=252` con retornos diarios), `scripts/experiments.py:87` (`ppy = 252/horizon`).

## 3. Cómo trabajar

1. Rama nueva desde `main` actualizado (p. ej. `fix/anualizacion-restante`).
2. Para cada sitio: lee el código, confirma qué retornos entran (**frecuencia real**) y apúntalo. No cambies nada hasta confirmarlo.
3. Un commit por sitio corregido. Cada corrección lleva un test que falle antes y pase después (datos sintéticos, sin red ni `data/` real).
4. Tests: `python -m pytest tests/<tu_archivo>.py -q`, y al final la suite completa. **Fallan 7 tests de `tests/test_graph_ising.py` en `main` por un problema de numpy ("underlying array is read-only"): no son de esta tarea, ignóralos.**

## 4. Aceptación

- Para cada fila de la tabla, una línea en tu PR: "frecuencia de los retornos = X, anualización correcta = Y, acción tomada = Z".
- Sitios 1 y 6: sin cambio de números (tests de regresión).
- Sitios 4 y 5: o corregidos con test, o dejados explícitamente como "no se toca porque…".
- La suite completa no empeora (salvo los 7 de `test_graph_ising.py`).

## 5. Fuera de alcance

Regenerar `data/eval/*.json` o `data/meta_results.json`; cambiar costos o el backtest top-K; reescribir el GA (solo decidir/documentar el sitio 3).
