# Avance: panel de ideas de ML y primera ronda de construcción

Fecha: 2026-11-05 · Rama: `build/m-series` (21 commits sobre `main`) · Alcance: sistema de señales para la BMV

## 1. Resumen en 30 segundos

El sistema predecía "¿sube en 5 días?" y no lograba ventaja (AUC 0.50–0.54, Sharpe neto negativo). Montamos un panel de 7 agentes que propuso 55 ideas, las priorizamos en 14 paquetes (M1–M14) y construimos y probamos la mayoría con sesiones separadas.

**Conclusión honesta:** con la evidencia actual el sistema **no tiene una ventaja demostrada**. Lo que sí quedó firme es la infraestructura para medir bien, un piso simple de comparación y varios errores corregidos que hacían ver los números mejor o peor de lo que eran.

## 2. Qué hicimos

1. **Panel de ideas.** 7 agentes (etiquetado, features, modelos, no supervisado, finanzas, validación, datos) escribieron ideas con hipótesis, riesgo de fuga, costo y forma de validarlas.
2. **Verificación propia.** Las afirmaciones de los agentes sobre el código se revisaron leyendo el código y recalculando sobre `data/`. Eso destapó dos errores que ningún agente vio (ver 3.1).
3. **Construcción por partes.** 21 handoffs de construcción, cada uno con archivos propios para poder correr en paralelo sin pisarse, y cada uno con su reporte. La suite de pruebas pasó de 114 a 325 pasan (1 falla conocida de LSTM, previa al proyecto).

## 3. Hallazgos

### 3.1 Errores de medición (lo más importante)
- **Sharpe y Calmar mal anualizados.** `portfolio_metrics_from_returns` ignoraba `periods_per_year` y usaba √252 con retornos de 5 días: todo Sharpe salía inflado ×2.24 y todo Calmar ×5. Con la corrección, el backtest guardado pasa de Sharpe bruto 1.53 / neto −1.79 / NAFTRAC 0.82 a 0.68 / −0.80 / 0.37. El "rompecabezas" de AUC 0.5065 con Sharpe 1.5 era en buena parte este error.
- **Costo fijo.** El backtest restaba 0.6% cada semana sin mirar cuánto cambió el portafolio. La rotación real del top-5 es 83%, y el costo de equilibrio (el costo por lado con el que el neto llega a 0) es 0.185%, contra el 0.30% base.
- **AUC agrupado.** Mezcla "qué día es" con "qué acción es". Ahora se mide con IC por fecha (correlación semanal entre el ranking y el retorno posterior).
- **Tres reglas de entrada distintas** (cierre a cierre, cierre actual, Open siguiente). Entrar al Open siguiente reduce el IC casi a la mitad.
- **Datos:** sin dividendos en los precios, panel de noticias de un solo ticker (WALMEX, 909 artículos), fecha de disponibilidad de fundamentales fija en +70 días, y filas basura (POSADASA con 88% de días sin volumen; Open≈0 en algunas filas de 2011).

### 3.2 Qué señales sobreviven
| Idea | Resultado |
|---|---|
| Reversión de 5 días | IC +0.056 a +0.060 cierre a cierre y +0.041 a +0.044 entrando al Open siguiente. Decae: ~0.07 antes de 2015, ~0.03 después. La puerta de decisión no se cerró (3 de 5 bloques sin máscara; 4 de 5 con ella, pero el último en 0.0302 contra un umbral de 0.03). |
| Modelo actual (`p_up` del ensemble) | IC +0.042 con el panel sucio; **+0.019 (ya no significativo) con el panel limpio**. Percentil 90 frente al top-5 aleatorio. Alfa contra el equiponderado +8.9% con t=1.24. |
| Piso simple | Ridge sobre ranks: IC +0.032 (+0.037 con panel limpio). **Ningún otro brazo le gana** (xs_lgbm +0.022, rankers +0.008 a +0.015, anti-ruido +0.033 empata). |
| Rotar menos | La celda 10d / K=10 / salida 2K rota 15–16%, equilibrio 1.1–1.2% por lado, Sharpe neto 0.80–0.87 (t del exceso 2.6–3.1). Es la mejor evidencia que tenemos, pero con 18 ensayos por regla y un solo tramo de 3 años. |
| Cambiar el objetivo (M10) | Ninguna variante supera al baseline (IC +0.0373). Escalar por riesgo empeora un poco. |

### 3.3 Lo que sí funciona: la volatilidad
- **Vol targeting.** En el equiponderado, con objetivo 10%, el MDD baja de −35.1% a −14.7% con casi el mismo Sharpe (0.88 → 0.85). El beneficio es de drawdown y no de retorno. HAR como estimador sale peor que EWMA (MDD −22.5%) por sesgo de re-exponenciar.
- **Mejorar el modelo de vol no se logró.** Ninguno de 15 regresores (OHLCV, calendario de eventos, macro) llega a +0.01 de skill sobre HAR (+0.30 a 5d, +0.19 a 21d). Lo único visible: HAR mejora ~3 puntos dentro de la semana de FOMC, que es ~10% de las filas.

### 3.4 Riesgo y sizing
- **Kelly:** con pagos fijos (3.32% / 2.44%) apuesta ~3.3% del capital incluso con p=0.50; con pagos estimados del train apuesta 0.
- **Stop por σ̂ y veto de cola:** sin mejora robusta. Toda configuración sale negativa neta porque el costo (0.6%) supera el drift.
- **HMM de régimen:** separa por volatilidad, no por dirección; "bear" y "lateral" son casi lo mismo.

## 4. Qué se construyó (en el repo)
- `metrics.py`: anualización correcta · `cross_section.py`: IC por fecha, baselines, bootstrap por bloques · `costs.py`: rotación y costo real · `quality.py`: máscara de calidad · `total_return.py`: dividendos · `labeling.py`/`targets.py`: opciones `next_open`, desempate y días hábiles.
- Estudios reproducibles en `scripts/` con resultados en `data/eval/` (reversión, rotación, vol targeting, vol regressors, calendario, stops, bake-off, panel limpio, rediseño de objetivo).
- `trials.py` (registro de ensayos), `power.py` (poder estadístico), DSR/PSR corregidos, archivador PIT de noticias (primer snapshot de 556 titulares).

## 5. Proceso y costo
- Agentes del panel: 7 en esfuerzo `high`, ~480k tokens en total. Construcción: sesiones separadas con contexto mínimo por handoff.
- Cada commit trae solo los archivos de su tarea, sin línea de coautor. El commit base se reescribió para quitarla.

## 6. Limitaciones y cuidado al leer los números
- Muchas cifras vienen de cálculos cortos con 26–35 tickers, cierre a cierre o sin control de supervivencia; las mediciones de reversión comparten los mismos datos de Yahoo, así que no son pruebas independientes.
- Los resultados de portafolio salen de un solo tramo de test (2023-03 a 2026-06, 172 semanas); un MDD de ±10 puntos es ruido.
- La lista de tickers es la de hoy (sesgo de supervivencia probable).
- El reentrenamiento cambia algunas cifras entre corridas (AUC 0.5065 contra 0.5133): se fijó semilla 42 desde M1d.

## 7. Pendiente (se atiende a continuación)
**Correcciones:** otros `√252` (`backtester.py:192`, `feature_engineering.py:546`, `fitness_evaluator.py:140`); fuga de 1 día en fundamentales (`BDay(1)`); empates de `p_up` en `backtest_topk.py`; `n_trials` del DSR en `run_purged_cv.py`; marca `empty` por 429 en `news_history.py`; test roto de LSTM; README desactualizado.
**Verificaciones:** dividendo extraordinario de HERDEZ (abril 2026, 15 MXN), alcance real de GDELT, acceso a Postgres `lab.news_daily`, cron local diario del archivador de noticias.
**Decisiones:** qué hacer con el sistema (investigación y riesgo, o paper trading de la regla Ridge + histéresis); qué defaults de producción cambiar (`use_mask`, Kelly estimado, factor de régimen HMM, Ising); cómo cerrar la rama.

