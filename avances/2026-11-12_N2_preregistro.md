# Pre-registro N2 · reglas congeladas para prueba prospectiva

Escrito el 2026-11-12, **antes de calcular nada** del flujo residual ni de semanas posteriores al corte. Rama `build/n-series`. Cualquier cambio posterior se anota como desviación en el reporte, no se edita este texto. Los JSON de `data/eval/preregistros/` guardan la definición + hash de este archivo, del código y de los datos.

**Corte (congelado):** 2026-11-12. Solo cuentan **fechas de señal ≥ 2026-11-13** (estrictamente futuras). Todo lo anterior —incluida la ventana 2026-06→10, ya vista— es contexto, no evidencia.

## Regla 1 · `flujo_residual` (idea 3.1)
- **Hipótesis:** el empuje de las casas en 5 días se revierte la semana siguiente. Signo esperado del IC: **negativo**.
- **Universo:** tickers MX del mapa `hechos_ticker_map.json` con confianza `alta` que además están en el panel BMV (la lista exacta se guarda en el JSON al congelar).
- **Señal (fecha t = último día hábil de la semana):**
  - `imb_5d` = media de `flow_imbalance` diario (definición de `flow_features.py`) en los 5 días hábiles que terminan en t; exige ≥ 4 de 5 días presentes en `hechos` (**no se rellenan días**).
  - `ret_5d` = Close(t)/Close(t−5) − 1.
  - `resid` = residuo, por fecha, de la regresión cruzada de `imb_5d` sobre `ret_5d` (mín. 15 nombres con dato).
- **Objetivo:** retorno 5d con entrada **Open t+1** → Close t+6 (misma convención que OOS v2).
- **Medición principal:** IC Spearman por fecha de `resid` vs retorno 5d; `t` con `ic_summary` (bloques de 4 semanas).
- **Cartera (secundaria):** largo las 5 de menor `resid`, rebalanceo semanal, pesos iguales, costo 0.30%/lado con rotación real; contra equiponderado del mismo universo a la misma frecuencia.
- **Éxito (sin ajustes):** con **≥ 26 semanas** nuevas: IC medio **≤ −0.04** y **t ≤ −2** y neto de cartera > neto EW.
- **Abandono:** con ≥ 26 semanas, IC medio **> −0.02** → se archiva sin ajustar.
- **Inconcluso:** entre −0.04 y −0.02, o t > −2: se sigue hasta **52 semanas**; ahí el criterio de éxito sube a **t ≤ −2.4** (segunda mirada); si no cumple, se archiva.
- **Miradas intermedias:** a las 13 semanas solo se revisa **fuga/errores** (|IC| > 0.10 = sospecha de error). No se declara éxito antes de 26.
- **Ensayos:** 1 por evaluación en `log_trial`, familia `n2_flujo`; solo al evaluar, no al congelar.
- **Límites:** ~13 semanas de `hechos` por trimestre; los días faltantes de `hechos` reducen n. N efectivo de nombres ≈ 12-15 (N1).

## Regla 2 · `reglas_v1` (de `2026-11-09_oos_v2_preregistro.md`)
- **Qué se congela:** E-frozen, E-refit, R-frozen, R-refit y REV tal como están definidas en ese archivo (carteras, panel, costos, semillas 42), **sin reajustar nada**.
- **Re-evaluación:** cada **13 semanas** desde el corte (primera: señales 2026-11-13 → 2027-02-12), sobre la ventana acumulada de fechas ≥ 2026-11-13, con el mismo criterio S1/S2 (REV: S1/S2′) y la misma clasificación (se sostiene / inconcluso / falla grosera) del preregistro de OOS v2.
- **Ensayos:** cada re-evaluación son 5 filas `log_trial` (familia `oos_v2_prosp`).
- **Límite:** con 13 semanas solo se ven fracasos groseros; la decisión de fondo exige muchas más (ver potencia).

## Potencia (aritmética, no medición)
Con la std semanal del IC de `oos_v2.json` (media de las 5 reglas ≈ 0.174; las de E-frozen ≈ 0.165): semanas para SE = IC (t≈1) y para |t| ≥ 2 (n = (2·σ/IC)²), IC 0.03 y 0.01: ver tabla calculada en cada JSON (con 0.174: ≈ 34 / 135 y ≈ 304 / 1217; con 0.165: ≈ 31 / 121 y ≈ 272 / 1089). Para el flujo la std real es desconocida; se usa esta como proxy. Con 26 semanas solo se detectan efectos ≥ ~0.065 (t=2): el criterio de éxito (−0.04) es **optimista** en potencia; se acepta y se dice al reportar.
