# M3c · Panel limpio: antes / después

Semilla 42 (ensemble, desempate M5, bootstrap); brazos M9 con semillas 0–4. Universo bmv, todos los tickers. Ensayos: 16.

| Métrica | Sin máscara | Con máscara | Nota |
|---|---|---|---|
| IC p_up (test) | +0.042 | +0.019 | EE 0.015 |
| AUC medio por fecha p_up | 0.5133 | 0.5098 |  |
| Top-5 Sharpe bruto | +0.76 | +0.75 |  |
| Top-5 Sharpe neto @0.30% | -0.47 | -0.39 |  |
| Top-5 rotación / equilibrio %/lado | 0.83 / 0.185 | 0.84 / 0.198 | base 0.30 |
| IC xs_lgbm (M9) | +0.022 | +0.021 | EE 0.010 |
| IC Ridge (M9) | +0.032 | +0.037 | EE 0.010 |
| IC mejor anti-ruido (l4_m300 / l4_m1000) | +0.033 | +0.036 | elegido sobre test |
| M5 10d/K10/2K · rotación | 0.16 | 0.15 | tope 0.30 |
| M5 · bruto %/sem · equilibrio %/lado | 0.361 / 1.133 | 0.364 / 1.231 | base 0.30 |
| M5 · Sharpe neto@0.30 · exceso t vs EW | 0.80 · 2.56 | 0.87 · 3.07 |  |
| M5 · pasa la puerta | True | True |  |
| max |fwd_ret| en test (%) | 41.0 | 41.0 |  |

**Filas descartadas** (reglas independientes; solo filas que habrían entrado al panel): train 1362 filas / 31 tickers; test 149 / 3.
- train: mask_invalid 1359 filas/31 tick., open_nonpos 5 filas/1 tick., fwd_ret_outlier 5 filas/1 tick.
- test: mask_invalid 149 filas/3 tick., open_nonpos 0 filas/0 tick., fwd_ret_outlier 0 filas/0 tick.
- Solo el guardián (no la máscara): 3 filas (KIMBERA).

**Conclusión**
El IC de p_up cambia -0.023 (EE 0.015) y el del Ridge +0.005 (EE 0.010); el Sharpe neto del top-5 pasa de -0.47 a -0.39; la celda de M5 sigue pasando la puerta de rotación/equilibrio. Algún cambio supera un error estándar o altera la puerta: no es un cambio de lectura automático; ver el reporte para la interpretación.
