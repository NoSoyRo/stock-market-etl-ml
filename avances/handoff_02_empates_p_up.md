# Handoff 02 · Desempatar el ranking de `p_up` en el backtest top-K

Para: quien lo implemente · Tamaño: S–M (medio día) · Depende de: nada (parte de `main` con el PR #5 ya mergeado)

## 1. Contexto en 1 minuto

`p_up` sale de `EnsembleModel.predict_proba`, que pasa la probabilidad por una **calibración isotónica**. La isotónica es escalonada: muchas filas reciben exactamente el mismo `p_up`. En el backtest top-K, cada semana se ordena por `p_up` y se toman las 5 primeras; con empates, **quién queda en el top-5 depende del orden de las filas** (de `sort_values`), no de la señal.

Medido: en 150 de 172 fechas del test, el 5.º y el 6.º puesto tenían el mismo `p_up`. Efecto práctico: la selección es parcialmente arbitraria, y la **rotación** depende del orden de las columnas. (Un estudio previo vio rotación 0.51 con desempate "por orden de columnas" contra 0.82–0.83 con desempate aleatorio; el 0.83 es el comparable.)

Ya existe la herramienta para arreglarlo (commit "fix(M13b)", `src/ml/ensemble.py`):

- `break_ties(p_cal, p_raw)`: desempata las mesetas de `p_cal` usando el score **sin calibrar** `p_raw`, de forma determinista y **sin cambiar el orden entre valores distintos** de `p_cal`.
- `EnsembleModel.predict_proba_tiebroken(df, feature_cols)`: `predict_proba` + `break_ties`. Pensado **para rankear, no para leer probabilidades**.
- `predict_proba` por defecto no cambia.

Lo que falta es **usarlo** en `scripts/backtest_topk.py`.

## 2. Qué hay que hacer

Archivos: `scripts/backtest_topk.py` (principal), un test nuevo, y los scripts que consumen su panel (solo para revisar, ver 2.4).

### 2.1 Diseño (decidido, para no darle vueltas)

- **No sobrescribir `p_up`.** `p_up` se usa como probabilidad: AUC (`y_prob=test["p_up"]`, línea ~329), umbral `min_p_up` (línea ~286) y reportes. Si lo desplazas, esas métricas cambian sin querer.
- **Agregar una columna nueva** `p_rank = model.predict_proba_tiebroken(test, FEATURE_COLS)` junto a `p_up`, en `build_test_panel` (donde hoy están `p_test = model.predict_proba(test, FEATURE_COLS)` y `test["p_up"] = p_test`, líneas ~429-430). Haz lo mismo para `p_rel` si ese modelo también es un `EnsembleModel` calibrado (revisa: si usa otro modelo, aplica la misma idea o déjalo anotado).
- **Rankear por `p_rank`**, pero aplicar el umbral `min_p_up` sobre `p_up`. Hoy `_score_column("p_up")` (línea ~195) devuelve `"p_up"` y se usa para ordenar (línea ~285) y filtrar (línea ~286-287). Separa las dos cosas: ordenar por `p_rank` cuando exista, filtrar por `p_up`.
- Flag nuevo `tiebreak: bool = True` en `run_topk_backtest` / `train_and_backtest` y `--no-tiebreak` en el CLI, para poder reproducir el comportamiento anterior.

### 2.2 Tests (escríbelos primero)

Nuevo archivo `tests/test_topk_ties.py`, datos sintéticos, sin red:

1. Panel con empates exactos en `p_up` pero distintos `p_rank`: el top-K elige por `p_rank`.
2. **Invariancia al orden de las filas**: barajar las filas del panel no cambia los tickers elegidos.
3. `min_p_up` sigue filtrando por `p_up` (un ticker con `p_up` bajo el umbral no entra aunque su `p_rank` sea alto).
4. `tiebreak=False` reproduce el orden anterior.

### 2.3 Medir el efecto (antes/después)

Con semilla fija (`SEED = 42` ya existe en `backtest_topk.py`), mismo universo (`bmv`), corre el backtest con `tiebreak=False` y `tiebreak=True` y reporta en tu PR una tabla corta:

| | tiebreak=False | tiebreak=True |
|---|---|---|
| Rotación media del top-5 | | |
| Sharpe bruto / neto (con rotación real) | | |
| IC por fecha de `p_up` / de `p_rank` | | |
| Fechas con empate en el puesto 5-6 | | |

Para el IC por fecha usa `src/ml/shared/evaluators/cross_section.py` (`ic_by_date`, `ic_summary`). Para el costo por rotación, `src/ml/shared/evaluators/costs.py`. Explica en 3 líneas si el cambio mueve la lectura o es ruido (el IC es de ~0.04 con error estándar de ~0.015: diferencias menores que eso no son señal).

### 2.4 Quién más usa el panel (solo revisar)

`scripts/run_m1_report.py`, `scripts/run_rotation_study.py` y `scripts/run_m3c_before_after.py` construyen su panel con `build_test_panel`. Hoy `run_rotation_study.py` desempata con un jitter aleatorio de 1e-9 (a propósito, para no sesgar la rotación). Revisa que seguir funcionando con la columna nueva no los rompa (sus tests pasan) y **no los modifiques**: anótalo en tu PR si conviene migrarlos después.

## 3. Cómo trabajar

1. Rama nueva desde `main` actualizado (p. ej. `fix/desempate-p-up`).
2. Tests primero, luego el código; un commit para el código y otro para los tests si prefieres, pero el PR debe quedar verde.
3. Tests: `python -m pytest tests/test_topk_ties.py tests/test_ensemble_ties.py tests/test_m1_report.py tests/test_clean_panel.py -q`, y luego la suite completa. **Fallan 7 tests de `tests/test_graph_ising.py` en `main` por un problema de numpy ("underlying array is read-only"): no son de esta tarea.**
4. **No regeneres** `data/eval/portfolio_metrics.json`, `data/eval/backtest_topk.json` ni `data/eval/m1_report.*`: cambiarían en miles de líneas. Escribe tus corridas de comparación con `--out` a un archivo nuevo (por ejemplo `data/eval/backtest_topk_tiebreak.json`) o solo pega la tabla en el PR.

## 4. Aceptación

- Con `tiebreak=True`, la selección del top-K es **idéntica aunque se barajen las filas** (test 2.2.2).
- `p_up`, el AUC y el filtro `min_p_up` no cambian de valor (test 2.2.3; el AUC con y sin flag coincide).
- Con `--no-tiebreak` se reproduce el comportamiento anterior.
- Tabla antes/después del punto 2.3 en el PR, con tu lectura en 3 líneas.
- Suite completa sin regresiones nuevas (salvo los 7 de `test_graph_ising.py`).

## 5. Fuera de alcance

Cambiar la calibración isotónica en sí o quitarla; tocar `src/ml/super_model.py` o el flujo de producción (`signal_pipeline`, `master_ranking`): el SuperModel tiene su propia fusión y no usa este panel; revisar si sufre el mismo problema es otra tarea. Cambiar defaults de otros scripts.
