# Pre-registro · N5 eventos de EPS (reversión con/sin anuncio, calendario, PEAD)

Escrito el 2026-11-12 **antes de calcular ningún IC ni retorno con estos grupos**. Rama `build/n-series`. Cambios posteriores van como desviación en el reporte, no se edita este texto.

## Datos y reglas comunes
- Eventos: `data/eps/eps_clean.parquet` (limpio por N1), solo los **37 tickers `.MX`**; los demás tickers del universo de precios (sin EPS) se **excluyen**, no se imputan. Universo de precios = el de `run_reversal_study.load_universe(use_mask=True)` (máscara M3a, ≥1500 filas, sin NAFTRAC/FIBRA*) ∩ tickers con EPS.
- Un anuncio es **conocido** desde `available_from` (día hábil siguiente a `announced`).
- Retorno: Open t+1 → Close t+6 (`f_open_close` de `reversal.forward_returns`, en fracciones). IC de Spearman por fecha, paso 5 (semanal sin solape), `ic_summary` (bloque 4, semilla 0). Costo 0.30%/lado.
- **Elegir en 2008-2017, probar UNA vez en 2018-2026.** Aquí nada se afina: ventana, mínimos y reglas quedan fijos abajo; 2008-2017 solo se reporta (chequeo de signo).

## (a) Reversión con / sin anuncio
- "Con evento" en t: algún anuncio con `available_from ≤ t` y dentro de los **últimos 5 días de la malla de precios** (posiciones t−4…t). "Sin evento": ninguno. Score = −ret_5d. IC por grupo con **mín. 6 nombres por grupo y fecha**; se documenta cuántas fechas quedan.
- **Éxito:** IC(sin evento) > +0.03 con t > 2 en 2018-2026 **y** IC(con evento) ≤ 0 en 2018-2026 **y** la diferencia IC(sin) − IC(con), en fechas con ambos grupos, tiene IC95% por bootstrap de bloques con límite inferior > 0.
- **Descarte automático:** IC(sin evento) < 0 en 2023-2026, aunque pase lo demás.
- Baselines: reversión simple (todos los EPS), equiponderado y top-5 aleatorio sobre los mismos nombres (informativos; top-5 bruto por grupo vs EW del grupo, 2018-2026).
- Sanidad: reversión simple en los 37 MX 2008-2026 debe quedar cerca de +0.04 (`reversal_study_mask.json`: +0.0424, 32 tickers).

## (b) Calendario esperado
- **Fecha esperada causal** = cada anuncio pasado `a` genera una fecha esperada `a + 364 días` (ajustada a día hábil), conocida desde `available_from` de `a`. Se mide el acierto (|error| en días hábiles ≤ 1, ≤ 2, ≤ 5) contra el anuncio real del trimestre (emparejando con el anuncio de 300–430 días antes).
- **(b1) Volatilidad.** Regresores `earn_in5`, `earn_in21`, `earn_days` con la fecha esperada (tolerancia ±2 días hábiles, `event_features`). `vol_regressors.evaluate_regressor` contra HAR (6 folds purgados, mismo objetivo Parkinson, solo tickers con EPS). **Éxito:** en h=5 **o** h=21, Δskill agrupado ≥ 0.01 con IC95% inferior > 0 (se cuentan los 2 ensayos). Se reporta también `skill_in_out`.
- **(b2) Filtro de riesgo.** Cada semana (paso 5): equiponderado de los nombres elegibles vs el mismo excluyendo los de **anuncio esperado con ventana (t−2, t+8] días hábiles** (el periodo mantenido t+1…t+6 más tolerancia ±2). Ambos brazos sobre la misma muestra (nombre-fecha con calendario disponible), neto con rotación real a 0.30%/lado. **Éxito en 2018-2026:** ΔSharpe semanal (filtrado − sin filtrar) con IC95% por bootstrap de bloques de semanas (bloque 4) > 0 **y** Δmedia neta ≥ −0.05% por semana (misma media), **y** mismo signo de ΔSharpe en 2008-2017.
- Referencia (no entra en el criterio): el mismo filtro con el anuncio **real** (oráculo, no operable).

## (c) PEAD (solo diagnóstico, sin criterio de cartera)
- Evento = anuncio; entrada **Close** del primer día de la malla ≥ `available_from`, salida +5 días de malla. Quintil de sorpresa **dentro de cada trimestre calendario** (≥ 10 eventos). Se reporta retorno medio por quintil, spread Q5−Q1 y Q5 largo-solo vs 0.60% de costo, para 2008-2017 y 2018-2026, **con todos los eventos y con el subconjunto de Open limpio** (Open > 0 y no nulo en los 7 días de malla alrededor del evento, sin máscara).

## Ensayos
Familia `n5_eps_events`: una fila en `trial_log` por variante (a: sin, con, diferencia, simple; b1: h5, h21; b2; c: todos, Open limpio), con métricas de 2018-2026. Se suman en el texto los ~15 cortes exploratorios del agente 02.
