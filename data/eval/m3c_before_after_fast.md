# M3c · Panel limpio: antes / después

Semilla 42 (ensemble, desempate M5, bootstrap); brazos M9 con semillas 0–4. Universo bmv, 16 tickers. Ensayos: 16.

| Métrica | Sin máscara | Con máscara | Nota |
|---|---|---|---|
| IC p_up (test) | +0.035 | +0.039 | EE 0.026 |
| AUC medio por fecha p_up | 0.5049 | 0.5125 |  |
| Top-5 Sharpe bruto | +0.46 | +0.85 |  |
| Top-5 Sharpe neto @0.30% | -0.54 | -0.17 |  |
| Top-5 rotación / equilibrio %/lado | 0.69 / 0.138 | 0.67 / 0.250 | base 0.30 |
| IC xs_lgbm (M9) | +0.043 | +0.040 | EE 0.012 |
| IC Ridge (M9) | +0.060 | +0.056 | EE 0.014 |
| IC mejor anti-ruido (l4_m300 / l4_m300) | +0.063 | +0.062 | elegido sobre test |
| M5 10d/K10/2K · rotación | 0.00 | 0.00 | tope 0.30 |
| M5 · bruto %/sem · equilibrio %/lado | -0.034 / -3.966 | 0.139 / 26.005 | base 0.30 |
| M5 · Sharpe neto@0.30 · exceso t vs EW | -0.10 · -2.46 | 0.40 · 0.44 |  |
| M5 · pasa la puerta | False | True |  |
| max |fwd_ret| en test (%) | 28.8 | 28.8 |  |

**Filas descartadas** (reglas independientes; solo filas que habrían entrado al panel): train 288 filas / 16 tickers; test 0 / 0.
- train: mask_invalid 285 filas/16 tick., open_nonpos 5 filas/1 tick., fwd_ret_outlier 5 filas/1 tick.
- test: mask_invalid 0 filas/0 tick., open_nonpos 0 filas/0 tick., fwd_ret_outlier 0 filas/0 tick.
- Solo el guardián (no la máscara): 3 filas (KIMBERA).

**Conclusión**
El IC de p_up cambia +0.004 (EE 0.026) y el del Ridge -0.004 (EE 0.014); el Sharpe neto del top-5 pasa de -0.54 a -0.17; la celda de M5 cambia su veredicto. Algún cambio supera un error estándar o altera la puerta: la lectura debe revisarse con estos números.
