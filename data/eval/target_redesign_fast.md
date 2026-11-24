# M10 · Rediseño del objetivo (Ridge limpio)

Generado: 2026-11-05T20:01:52 · 16 tickers · 69205 filas · 24 features · 5 folds purgados · semilla 42

Criterio (fijado antes): gana si supera a A (IC crudo, diferencia pareada) por ≥ 2 EE en ≥ 3 de 5 folds

Features constantes por fecha detectadas: ['rn_wti_ret5', 'rn_sp500_ret5', 'rn_usdmxn_ret5']

## IC por variante (step=5; IC95% por bootstrap de bloques)

| var | descripción | IC crudo | t | IC95% | IC en σ̂ | Δ vs A (EE) | folds ganados | gana |
|---|---|---|---|---|---|---|---|---|
| A | baseline: percentil de fwd_ret | +0.0555 | 4.06 | [+0.029, +0.084] | +0.0558 | — | —/5 | — |
| B | percentil de fwd_ret/σ̂ | +0.0520 | 3.70 | [+0.026, +0.081] | +0.0515 | -0.0035 (0.0036) | 0/5 | no |
| C | percentil de fwd_ret − media de la fecha | +0.0555 | 4.06 | [+0.029, +0.084] | +0.0558 | +0.0000 (0.0000) | 0/5 | idéntica a A |
| D | como A, entrena solo con |fwd_ret| ≥ 0.5σ̂√5 | +0.0471 | 3.51 | [+0.021, +0.075] | +0.0498 | -0.0084 (0.0038) | 0/5 | no |
| E | como A, sin features constantes por fecha | +0.0558 | 4.08 | [+0.029, +0.084] | +0.0560 | +0.0003 (0.0003) | 0/5 | no |

## IC crudo por fold

| var | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|
| A | +0.020 | +0.086 | +0.095 | +0.029 | +0.048 |
| B | +0.021 | +0.077 | +0.100 | +0.037 | +0.024 |
| C | +0.020 | +0.086 | +0.095 | +0.029 | +0.048 |
| D | +0.022 | +0.082 | +0.070 | +0.024 | +0.038 |
| E | +0.020 | +0.086 | +0.096 | +0.029 | +0.048 |

## Portafolio (M5: 10d, K=10, salida 20; costo 0.30 %/lado)

| var | cobertura train | rotación | bruto %/rebal | equilibrio %/lado | Sharpe bruto | Sharpe neto | exceso t |
|---|---|---|---|---|---|---|---|
| A | 100% | 0.00 | 0.122 | 41.122 | 0.36 | 0.35 | -0.52 |
| B | 100% | 0.00 | 0.128 | 37.717 | 0.35 | 0.35 | -0.41 |
| C | 100% | 0.00 | 0.122 | 41.122 | 0.36 | 0.35 | -0.52 |
| D | 61% | 0.00 | 0.119 | 40.347 | 0.33 | 0.33 | -0.60 |
| E | 100% | 0.00 | 0.122 | 41.122 | 0.36 | 0.35 | -0.52 |

**Veredicto:** empate: se queda A (baseline)
Ensayos: 5
