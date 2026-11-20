# Pre-registro · N3 captura del ex-dividendo

Escrito el 2026-11-12 **antes de calcular ningún retorno de evento**. Rama `build/n-series`. Cambios posteriores van como desviación en el reporte, no se edita este texto.

## Evento y retorno
- **Evento** = ex-fecha `d` de cada dividendo en `data/dividends/<ticker>.parquet`: los 36 de `load_universe_tickers("bmv")` + el resto de archivos `*.MX`, 2004→2026. `d` se asigna al primer día hábil del índice de precios ≥ `d` (posición `p`).
- **Ejecución al CIERRE** (no al Open): compra en `Close[p−5]`, venta en `Close[p]`. Los precios del repo no están ajustados por dividendo, así que la caída ex es real.
- **Retorno bruto** = `(Close[p] + div) / Close[p−5] − 1`. **Neto** = bruto − 0.60% (0.30%/lado), con retención de **10%** sobre el dividendo (supuesto sin verificar: `(Close[p] + 0.9·div)/Close[p−5] − 1 − 0.006`); también se reporta **sin retención**. Ambos marcados.
- **Yield** = `div / Close[p−5]`. Se descarta el evento si el dividendo falta o yield > 15% (extraordinarios; se listan aparte, p. ej. HERDEZ 2026-04).
- **Versión A (largo-solo)**: retorno absoluto del evento (la operable). **Versión B (cubierta)**: A menos el retorno del equiponderado del universo `bmv` en el mismo periodo (aísla el efecto; no incluye costo de la cobertura).

## Dos formas de conocer el evento
1. **Fecha conocida (oráculo)**: usa la ex-fecha real y el yield realizado. Sirve para replicar el efecto (sanidad: A/B bruto ≈ +0.85% por evento / +0.93% vs mercado del panel, t ≈ 6 sobre ~91 trimestres) y como cota superior.
2. **Fecha causal (la operable)**: el parquet no trae fecha de declaración (se revisa y se anota). Regla: cada ex-fecha pasada `h` genera una operación prevista en `P = h + 364 días` (mismo día de la semana, luego primer día hábil ≥), con compra en `P−5` y venta en `P`. Solo usa información ≤ `P−5` (la propia `h`). El retorno de esa operación incluye cualquier dividendo cuya ex-fecha **real** caiga en `(P−5, P]`; si la fecha se falla, la operación puede no cobrar nada (se cuenta igual). El yield causal = dividendo de `h` / `Close[P−5]`. Se mide acierto (± 2 días hábiles) y el retorno según acierto/fallo.

## Corte
- Parámetros elegidos en **2008-2017** (ventana −5, fija; umbral del cuartil alto de yield = percentil 75 de los yields de 2008-2017, congelado). Prueba **una sola vez** en **2018-2026** (por fecha de venta). 2004-2007 solo para historia y la sanidad.
- n honesto = **trimestres** (media por trimestre de la fecha de venta; t = media / (desv/√n_trimestres)). No se usan ventanas distintas de −5 (si se probara otra, sería ensayo extra y se declara).

## Criterio de éxito (fijado antes)
En el **cuartil alto de yield**, **versión A con fecha conocida causalmente** (forma 2, yield causal), neto de 0.30%/lado y retención 10%:
1. media neta por evento **> +0.30%** en 2018-2026, **y**
2. **t > 2** sobre medias por trimestre en 2018-2026, **y**
3. **mismo signo (media neta > 0)** en 2008-2017.

- **Se sostiene** = 1, 2 y 3 en la forma causal.
- **No operable** = el oráculo (forma 1) pasa 1-3 pero la causal no (el efecto existe pero la fecha no se conoce a tiempo).
- **No se sostiene** = el oráculo tampoco pasa.
- **Inconcluso** = cumple 1 y 3 pero 2 queda entre t 1.5 y 2, o el n por trimestre < 20 en el cuartil.

## Reportes adicionales (no entran en el criterio)
Eventos por año y capacidad (eventos/año y por ticker); cuánto de la media viene de los 5 mejores eventos (media sin ellos); cortes por yield (cuartiles), por año y por trimestre; con/sin retención.

## Ensayos
Familia `n3_exdiv`, una fila en `trial_log` por variante × versión (variante ∈ {oráculo, causal} × {todos, cuartil alto}; versión ∈ {A, B}), con las métricas de 2018-2026.
