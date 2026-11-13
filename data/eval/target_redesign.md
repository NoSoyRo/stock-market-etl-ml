# M10 · Rediseño del objetivo (Ridge limpio)

Generado: 2026-11-05T20:02:57 · 31 tickers · 117106 filas · 24 features · 5 folds purgados · semilla 42

Criterio (fijado antes): gana si supera a A (IC crudo, diferencia pareada) por ≥ 2 EE en ≥ 3 de 5 folds

Features de mercado quitadas en E: ['rn_wti_ret5', 'rn_sp500_ret5', 'rn_usdmxn_ret5'] · constantes en ≥90 % de fechas (diagnóstico): ninguna

## IC por variante (step=5; IC95% por bootstrap de bloques)

| var | descripción | IC crudo | t | IC95% | IC en σ̂ | Δ vs A (EE) | folds ganados | gana |
|---|---|---|---|---|---|---|---|---|
| A | baseline: percentil de fwd_ret | +0.0373 | 3.61 | [+0.017, +0.058] | +0.0354 | — | —/5 | — |
| B | percentil de fwd_ret/σ̂ | +0.0309 | 3.01 | [+0.011, +0.052] | +0.0282 | -0.0064 (0.0023) | 0/5 | no |
| C | percentil de fwd_ret − media de la fecha | +0.0373 | 3.61 | [+0.017, +0.058] | +0.0354 | +0.0000 (0.0000) | 0/5 | idéntica a A |
| D | como A, entrena solo con |fwd_ret| ≥ 0.5σ̂√5 | +0.0380 | 3.67 | [+0.017, +0.059] | +0.0387 | +0.0007 (0.0025) | 0/5 | no |
| E | como A, sin 3 features de mercado | +0.0365 | 3.54 | [+0.016, +0.057] | +0.0348 | -0.0008 (0.0005) | 0/5 | no |

## IC crudo por fold

| var | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|
| A | +0.034 | +0.052 | +0.048 | +0.048 | +0.000 |
| B | +0.029 | +0.051 | +0.056 | +0.049 | -0.010 |
| C | +0.034 | +0.052 | +0.048 | +0.048 | +0.000 |
| D | +0.041 | +0.049 | +0.043 | +0.041 | +0.006 |
| E | +0.033 | +0.053 | +0.048 | +0.048 | -0.000 |

## Portafolio (M5: 10d, K=10, salida 20; costo 0.30 %/lado)

| var | cobertura train | rotación | bruto %/rebal | equilibrio %/lado | Sharpe bruto | Sharpe neto | exceso t |
|---|---|---|---|---|---|---|---|
| A | 100% | 0.17 | 0.136 | 0.399 | 0.34 | 0.08 | 0.37 |
| B | 100% | 0.16 | 0.140 | 0.428 | 0.36 | 0.11 | 0.48 |
| C | 100% | 0.17 | 0.136 | 0.399 | 0.34 | 0.08 | 0.37 |
| D | 60% | 0.18 | 0.160 | 0.447 | 0.39 | 0.13 | 0.79 |
| E | 100% | 0.17 | 0.136 | 0.401 | 0.34 | 0.08 | 0.38 |

**Veredicto:** empate: se queda A (baseline)
Ensayos: 5
