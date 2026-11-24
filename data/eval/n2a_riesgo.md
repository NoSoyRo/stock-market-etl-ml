# N2-A motor de riesgo (OOS 6 folds, rebalanceo mensual, costo 0.30%/lado)

Folds (F1 antiguo .. F6 reciente): [['2017-07-21', '2019-02-07'], ['2019-02-08', '2020-08-25'], ['2020-08-26', '2022-03-10'], ['2022-03-11', '2023-09-22'], ['2023-09-25', '2025-04-15'], ['2025-04-16', '2026-11-03']]. Objetivo vol 10%. Bootstrap por bloques {'B': 1000, 'block': 21}.
Pre-registro completo en el docstring de scripts/n2a_riesgo.py.

## 1. Metricas agrupadas OOS (neto de costos)

| config | vol | MDD | CVaR5 d | Calmar | Sharpe | ret anual | exposicion | turnover/anio |
|---|---|---|---|---|---|---|---|---|
| ew | 15.2% | -34.5% | -2.17% | 0.17 | 0.44 | 5.7% | 1.00 | 0.71 |
| invvol | 14.5% | -33.9% | -2.06% | 0.16 | 0.43 | 5.4% | 1.00 | 0.77 |
| mv_sample | 12.4% | -31.5% | -1.71% | 0.13 | 0.39 | 4.1% | 1.00 | 2.36 |
| mv_lw | 12.4% | -31.5% | -1.72% | 0.13 | 0.38 | 4.1% | 1.00 | 2.23 |
| mv_oas | 12.4% | -31.6% | -1.72% | 0.13 | 0.38 | 4.1% | 1.00 | 2.28 |
| mv_cc | 12.5% | -31.4% | -1.72% | 0.14 | 0.41 | 4.5% | 1.00 | 1.96 |
| hrp_sample | 13.0% | -33.7% | -1.85% | 0.11 | 0.35 | 3.8% | 1.00 | 1.93 |
| hrp_lw | 13.2% | -33.2% | -1.87% | 0.13 | 0.39 | 4.4% | 1.00 | 1.75 |
| rp_lw | 13.9% | -33.9% | -1.98% | 0.15 | 0.43 | 5.2% | 1.00 | 0.93 |
| ew|ewma | 10.4% | -19.3% | -1.45% | 0.11 | 0.25 | 2.1% | 0.77 | 3.97 |
| ew|fhs5 | 9.3% | -19.3% | -1.30% | 0.08 | 0.20 | 1.5% | 0.70 | 4.05 |
| ew|lgbm5 | 10.4% | -21.2% | -1.44% | 0.07 | 0.20 | 1.6% | 0.76 | 6.35 |
| ew|fhs21 | 8.7% | -19.4% | -1.23% | 0.06 | 0.17 | 1.1% | 0.66 | 3.93 |
| ew|lgbm21 | 10.3% | -31.9% | -1.49% | -0.01 | 0.01 | -0.5% | 0.73 | 6.06 |
| hrp_lw|ewma | 10.0% | -18.8% | -1.38% | 0.10 | 0.24 | 1.9% | 0.85 | 4.53 |
| hrp_lw|fhs5 | 9.2% | -18.1% | -1.28% | 0.07 | 0.18 | 1.3% | 0.79 | 4.87 |
| hrp_lw|lgbm5 | 10.0% | -18.4% | -1.35% | 0.13 | 0.30 | 2.5% | 0.84 | 6.20 |
| hrp_lw|fhs21 | 9.0% | -18.3% | -1.25% | 0.06 | 0.16 | 1.0% | 0.77 | 4.86 |
| hrp_lw|lgbm21 | 10.0% | -27.3% | -1.42% | 0.01 | 0.06 | 0.2% | 0.81 | 6.66 |
| mv_lw|ewma | 9.9% | -20.0% | -1.33% | 0.10 | 0.25 | 2.0% | 0.87 | 4.77 |
| mv_lw|fhs5 | 9.5% | -19.7% | -1.28% | 0.08 | 0.21 | 1.6% | 0.84 | 5.03 |
| mv_lw|lgbm5 | 10.1% | -19.0% | -1.35% | 0.14 | 0.31 | 2.7% | 0.89 | 5.49 |
| mv_lw|fhs21 | 9.3% | -19.8% | -1.26% | 0.07 | 0.19 | 1.3% | 0.82 | 4.94 |
| mv_lw|lgbm21 | 10.2% | -26.8% | -1.39% | 0.06 | 0.21 | 1.7% | 0.87 | 6.39 |
| hrp_lw_tilt0.25 | 13.2% | -32.2% | -1.88% | 0.13 | 0.37 | 4.2% | 1.00 | 3.52 |
| hrp_lw_tilt0.5 | 13.3% | -31.9% | -1.90% | 0.12 | 0.34 | 3.7% | 1.00 | 6.13 |

## 2. Por fold: MDD / Sharpe

| config | F1 | F2 | F3 | F4 | F5 | F6 |
|---|---|---|---|---|---|---|
| ew | -19.4% / -0.52 | -33.9% / -0.02 | -7.4% / 1.49 | -18.6% / 0.55 | -11.3% / 0.28 | -12.7% / 1.10 |
| mv_sample | -15.9% / -0.59 | -24.9% / -0.05 | -6.7% / 0.92 | -12.0% / 0.95 | -9.9% / 0.48 | -10.5% / 0.76 |
| mv_lw | -16.3% / -0.57 | -25.1% / -0.05 | -6.7% / 0.92 | -12.3% / 0.94 | -10.6% / 0.42 | -10.3% / 0.79 |
| hrp_lw | -18.1% / -0.49 | -29.6% / -0.10 | -6.5% / 1.17 | -15.4% / 0.83 | -12.9% / 0.19 | -9.9% / 1.07 |
| rp_lw | -18.6% / -0.53 | -31.7% / -0.06 | -6.3% / 1.34 | -16.5% / 0.70 | -10.2% / 0.32 | -10.5% / 1.12 |
| ew|ewma | -15.3% / -0.64 | -15.1% / 0.03 | -6.4% / 1.03 | -14.9% / 0.19 | -11.0% / 0.07 | -11.7% / 0.83 |
| ew|fhs5 | -15.0% / -0.74 | -13.1% / -0.05 | -6.4% / 0.95 | -13.1% / 0.18 | -10.1% / 0.07 | -10.8% / 0.81 |
| ew|lgbm5 | -15.8% / -0.88 | -14.0% / -0.24 | -6.3% / 1.05 | -14.6% / 0.34 | -11.8% / 0.02 | -11.7% / 0.79 |
| hrp_lw|ewma | -13.3% / -0.50 | -14.1% / -0.07 | -5.2% / 0.84 | -13.6% / 0.55 | -13.5% / -0.06 | -10.2% / 0.72 |
| hrp_lw|fhs5 | -12.5% / -0.56 | -12.6% / -0.16 | -5.1% / 0.80 | -12.5% / 0.49 | -12.8% / -0.10 | -10.0% / 0.66 |
| hrp_lw|lgbm5 | -12.1% / -0.61 | -12.0% / -0.00 | -5.9% / 0.94 | -14.3% / 0.60 | -13.7% / -0.06 | -10.6% / 0.79 |
| hrp_lw|lgbm21 | -17.1% / -0.93 | -16.1% / -0.65 | -5.0% / 0.93 | -13.9% / 0.69 | -13.4% / -0.12 | -10.9% / 0.63 |

## 3. Estres (MDD ventana / retorno ventana)

| config | 2020 | 2022 | 2008 (sin fold, sin parametros) |
|---|---|---|---|
| ew | -32.4% / -14.5% | -18.6% / -1.5% | -41.0% / -5.7% |
| mv_sample | -24.6% / -7.3% | -12.0% / 5.6% | -34.3% / 1.2% |
| mv_lw | -24.7% / -7.6% | -12.3% / 5.4% | -34.1% / 1.1% |
| hrp_lw | -28.7% / -12.2% | -15.4% / 2.5% | -36.0% / -1.2% |
| rp_lw | -30.6% / -13.3% | -16.5% / 1.0% | -36.4% / -2.8% |
| ew|ewma | -13.2% / -7.0% | -14.9% / -5.6% | -17.0% / -5.4% |
| ew|fhs5 | -11.5% / -6.1% | -13.1% / -5.0% | -12.2% / -3.3% |
| ew|lgbm5 | -11.2% / -6.9% | -14.6% / -3.0% | - |
| hrp_lw|ewma | -12.6% / -6.3% | -13.6% / -2.0% | -15.6% / -3.3% |
| hrp_lw|fhs5 | -11.2% / -5.7% | -12.5% / -2.1% | -12.3% / -2.1% |
| hrp_lw|lgbm5 | -9.6% / -3.7% | -14.3% / -0.7% | - |
| hrp_lw|lgbm21 | -14.3% / -11.1% | -13.9% / -0.7% | - |

## 4. Comparaciones vs baseline (folds que mejoran de 6 / diferencia agrupada [IC90 bootstrap]); POSITIVO segun regla

| estrategia | baseline | MDD | CVaR | Sharpe | positivo en |
|---|---|---|---|---|---|
| ew | mv_sample | 0/6 / -3.04% [-15.41%, 3.14%] | 0/6 / -0.45% [-0.62%, -0.32%] | 4/6 / 0.05 [-0.21, 0.34] | ninguno |
| invvol | ew | 5/6 / 0.65% [-0.46%, 3.22%] | 6/6 / 0.10% [0.08%, 0.13%] | 2/6 / -0.01 [-0.06, 0.04] | cvar5 |
| invvol | mv_sample | 0/6 / -2.39% [-12.52%, 3.46%] | 0/6 / -0.35% [-0.49%, -0.24%] | 3/6 / 0.04 [-0.18, 0.29] | ninguno |
| mv_sample | ew | 6/6 / 3.04% [-3.14%, 15.41%] | 6/6 / 0.45% [0.32%, 0.62%] | 2/6 / -0.05 [-0.34, 0.21] | cvar5 |
| mv_lw | ew | 6/6 / 2.98% [-2.94%, 14.85%] | 6/6 / 0.44% [0.32%, 0.61%] | 2/6 / -0.06 [-0.33, 0.19] | cvar5 |
| mv_lw | mv_sample | 2/6 / -0.06% [-0.99%, 0.61%] | 2/6 / -0.01% [-0.02%, 0.00%] | 3/6 / -0.01 [-0.03, 0.02] | ninguno |
| mv_oas | ew | 6/6 / 2.88% [-3.22%, 14.97%] | 6/6 / 0.45% [0.32%, 0.61%] | 2/6 / -0.06 [-0.33, 0.20] | cvar5 |
| mv_oas | mv_sample | 2/6 / -0.16% [-0.69%, 0.35%] | 2/6 / -0.01% [-0.01%, -0.00%] | 3/6 / -0.00 [-0.02, 0.01] | ninguno |
| mv_cc | ew | 5/6 / 3.15% [-2.37%, 15.85%] | 6/6 / 0.45% [0.31%, 0.63%] | 3/6 / -0.03 [-0.30, 0.22] | cvar5 |
| mv_cc | mv_sample | 2/6 / 0.11% [-2.84%, 3.61%] | 3/6 / -0.00% [-0.04%, 0.02%] | 3/6 / 0.02 [-0.08, 0.11] | ninguno |
| hrp_sample | ew | 5/6 / 0.84% [-2.65%, 8.03%] | 6/6 / 0.31% [0.24%, 0.40%] | 2/6 / -0.09 [-0.25, 0.06] | cvar5 |
| hrp_sample | mv_sample | 1/6 / -2.20% [-9.19%, 2.19%] | 1/6 / -0.14% [-0.23%, -0.06%] | 3/6 / -0.03 [-0.19, 0.13] | ninguno |
| hrp_lw | ew | 5/6 / 1.36% [-1.60%, 7.94%] | 6/6 / 0.29% [0.23%, 0.37%] | 2/6 / -0.05 [-0.19, 0.08] | cvar5 |
| hrp_lw | mv_sample | 2/6 / -1.69% [-8.63%, 2.68%] | 1/6 / -0.16% [-0.26%, -0.08%] | 3/6 / 0.00 [-0.16, 0.17] | ninguno |
| rp_lw | ew | 6/6 / 0.63% [-0.31%, 4.96%] | 6/6 / 0.19% [0.15%, 0.23%] | 3/6 / -0.01 [-0.09, 0.07] | cvar5 |
| rp_lw | mv_sample | 2/6 / -2.41% [-10.97%, 3.20%] | 0/6 / -0.27% [-0.39%, -0.16%] | 3/6 / 0.04 [-0.15, 0.26] | ninguno |
| ew|fhs5 | ew|ewma | 6/6 / 0.02% [0.37%, 3.17%] | 6/6 / 0.15% [0.13%, 0.17%] | 0/6 / -0.05 [-0.08, -0.02] | mdd,cvar5 |
| ew|fhs21 | ew|ewma | 6/6 / -0.12% [0.53%, 4.65%] | 6/6 / 0.22% [0.20%, 0.25%] | 1/6 / -0.08 [-0.12, -0.04] | mdd,cvar5 |
| ew|lgbm5 | ew|ewma | 4/6 / -1.94% [-4.71%, 2.81%] | 4/6 / 0.02% [-0.05%, 0.08%] | 2/6 / -0.05 [-0.17, 0.06] | ninguno |
| ew|lgbm21 | ew|ewma | 4/6 / -12.66% [-16.60%, 0.21%] | 4/6 / -0.04% [-0.20%, 0.08%] | 2/6 / -0.25 [-0.42, -0.07] | ninguno |
| hrp_lw|ewma | ew|ewma | 5/6 / 0.52% [-3.85%, 5.73%] | 6/6 / 0.07% [0.04%, 0.10%] | 2/6 / -0.01 [-0.16, 0.13] | cvar5 |
| hrp_lw|fhs5 | ew|ewma | 5/6 / 1.14% [-3.26%, 6.64%] | 6/6 / 0.17% [0.14%, 0.20%] | 2/6 / -0.07 [-0.22, 0.07] | cvar5 |
| hrp_lw|fhs21 | ew|ewma | 5/6 / 1.01% [-3.41%, 7.03%] | 6/6 / 0.20% [0.17%, 0.24%] | 2/6 / -0.09 [-0.24, 0.05] | cvar5 |
| hrp_lw|lgbm5 | ew|ewma | 5/6 / 0.84% [-3.98%, 8.72%] | 6/6 / 0.10% [0.04%, 0.16%] | 2/6 / 0.04 [-0.15, 0.23] | cvar5 |
| hrp_lw|lgbm21 | ew|ewma | 3/6 / -8.05% [-12.89%, 2.82%] | 4/6 / 0.03% [-0.07%, 0.11%] | 1/6 / -0.19 [-0.39, 0.01] | ninguno |
| mv_lw|ewma | ew|ewma | 5/6 / -0.74% [-6.21%, 11.18%] | 5/6 / 0.12% [0.07%, 0.17%] | 3/6 / -0.01 [-0.29, 0.26] | cvar5 |
| mv_lw|fhs5 | ew|ewma | 5/6 / -0.37% [-6.02%, 11.48%] | 6/6 / 0.17% [0.12%, 0.22%] | 3/6 / -0.04 [-0.33, 0.23] | cvar5 |
| mv_lw|fhs21 | ew|ewma | 5/6 / -0.49% [-6.08%, 11.59%] | 5/6 / 0.19% [0.14%, 0.25%] | 3/6 / -0.06 [-0.35, 0.21] | cvar5 |
| mv_lw|lgbm5 | ew|ewma | 5/6 / 0.31% [-5.48%, 13.02%] | 5/6 / 0.11% [0.04%, 0.17%] | 2/6 / 0.06 [-0.24, 0.35] | cvar5 |
| mv_lw|lgbm21 | ew|ewma | 3/6 / -7.50% [-9.95%, 9.38%] | 4/6 / 0.06% [-0.02%, 0.12%] | 2/6 / -0.04 [-0.35, 0.26] | ninguno |
| hrp_lw_tilt0.25 | hrp_lw | 3/6 / 0.98% [-1.91%, 1.65%] | 2/6 / -0.00% [-0.02%, 0.02%] | 2/6 / -0.02 [-0.06, 0.03] | ninguno |
| hrp_lw_tilt0.5 | hrp_lw | 3/6 / 1.27% [-4.59%, 2.63%] | 2/6 / -0.03% [-0.07%, 0.01%] | 1/6 / -0.05 [-0.14, 0.03] | ninguno |
| ew|ewma | ew | 6/6 / 15.24% [-3.20%, 21.12%] | 6/6 / 0.72% [0.42%, 1.12%] | 1/6 / -0.19 [-0.40, 0.06] | cvar5 |
| hrp_lw|ewma | hrp_lw | 4/6 / 14.40% [-4.53%, 18.37%] | 6/6 / 0.49% [0.25%, 0.85%] | 1/6 / -0.15 [-0.35, 0.09] | cvar5 |

## 5. Diebold-Mariano sobre varianza realizada (r^2 bruto escalado; dif<0 => menos varianza que el baseline)

| comparacion | dif media | t | p |
|---|---|---|---|
| ew vs mv_sample | 3.06e-05 | 6.70 | 0.000 |
| invvol vs ew | -8.18e-06 | -7.67 | 0.000 |
| invvol vs mv_sample | 2.24e-05 | 6.22 | 0.000 |
| mv_sample vs ew | -3.06e-05 | -6.70 | 0.000 |
| mv_lw vs ew | -3.05e-05 | -6.79 | 0.000 |
| mv_lw vs mv_sample | 1.30e-07 | 0.68 | 0.497 |
| mv_oas vs ew | -3.05e-05 | -6.77 | 0.000 |
| mv_oas vs mv_sample | 1.10e-07 | 0.93 | 0.351 |
| mv_cc vs ew | -2.98e-05 | -6.29 | 0.000 |
| mv_cc vs mv_sample | 8.70e-07 | 1.18 | 0.236 |
| hrp_sample vs ew | -2.47e-05 | -8.46 | 0.000 |
| hrp_sample vs mv_sample | 5.91e-06 | 2.90 | 0.004 |
| hrp_lw vs ew | -2.24e-05 | -8.36 | 0.000 |
| hrp_lw vs mv_sample | 8.23e-06 | 3.85 | 0.000 |
| rp_lw vs ew | -1.49e-05 | -9.21 | 0.000 |
| rp_lw vs mv_sample | 1.58e-05 | 5.15 | 0.000 |
| ew|fhs5 vs ew|ewma | -8.74e-06 | -17.58 | 0.000 |
| ew|fhs21 vs ew|ewma | -1.27e-05 | -18.21 | 0.000 |
| ew|lgbm5 vs ew|ewma | -2.13e-07 | -0.13 | 0.895 |
| ew|lgbm21 vs ew|ewma | -5.59e-07 | -0.21 | 0.832 |
| hrp_lw|ewma vs ew|ewma | -3.08e-06 | -5.12 | 0.000 |
| hrp_lw|fhs5 vs ew|ewma | -9.09e-06 | -12.98 | 0.000 |
| hrp_lw|fhs21 vs ew|ewma | -1.10e-05 | -14.34 | 0.000 |
| hrp_lw|lgbm5 vs ew|ewma | -3.36e-06 | -2.76 | 0.006 |
| hrp_lw|lgbm21 vs ew|ewma | -3.61e-06 | -2.37 | 0.018 |
| mv_lw|ewma vs ew|ewma | -3.99e-06 | -3.58 | 0.000 |
| mv_lw|fhs5 vs ew|ewma | -6.98e-06 | -6.24 | 0.000 |
| mv_lw|fhs21 vs ew|ewma | -8.49e-06 | -7.47 | 0.000 |
| mv_lw|lgbm5 vs ew|ewma | -2.02e-06 | -1.47 | 0.143 |
| mv_lw|lgbm21 vs ew|ewma | -1.56e-06 | -1.05 | 0.293 |

## 6. Pronostico de cola 5 % (agrupado OOS)

| serie / h / modelo | pinball | cobertura (todo) | hits no solap. | Kupiec p | CC p | DM vs normal (t) | folds mejor que normal |
|---|---|---|---|---|---|---|---|
| ew|h5|norm5 | 0.00244 | 6.5% | 30/466 | 0.172 | 0.289 | - | - |
| ew|h5|fhs5 | 0.00247 | 4.6% | 21/466 | 0.619 | 0.883 | 0.57 | 2 |
| ew|h5|lgbm5 | 0.00235 | 5.6% | 27/466 | 0.443 | 0.653 | -1.60 | 5 |
| ew|h21|norm21 | 0.00588 | 5.9% | 8/111 | 0.315 | 0.521 | - | - |
| ew|h21|fhs21 | 0.00609 | 3.8% | 4/111 | 0.478 | 0.216 | 0.98 | 2 |
| ew|h21|lgbm21 | 0.00617 | 5.4% | 7/111 | 0.543 | 0.613 | 1.43 | 2 |
| hrp_lw|h5|norm5 | 0.00208 | 5.6% | 27/466 | 0.443 | 0.653 | - | - |
| hrp_lw|h5|fhs5 | 0.00211 | 4.8% | 25/466 | 0.721 | 0.889 | 0.72 | 2 |
| hrp_lw|h5|lgbm5 | 0.00198 | 5.1% | 27/466 | 0.443 | 0.653 | -1.58 | 4 |
| hrp_lw|h21|norm21 | 0.00506 | 5.1% | 8/111 | 0.315 | 0.521 | - | - |
| hrp_lw|h21|fhs21 | 0.00520 | 3.8% | 6/111 | 0.847 | 0.578 | 1.04 | 2 |
| hrp_lw|h21|lgbm21 | 0.00531 | 5.1% | 7/111 | 0.543 | 0.613 | 2.09 | 1 |

## 7. Veredicto

Comparaciones POSITIVAS segun la regla pre-registrada: [('invvol', 'ew', ['cvar5']), ('mv_sample', 'ew', ['cvar5']), ('mv_lw', 'ew', ['cvar5']), ('mv_oas', 'ew', ['cvar5']), ('mv_cc', 'ew', ['cvar5']), ('hrp_sample', 'ew', ['cvar5']), ('hrp_lw', 'ew', ['cvar5']), ('rp_lw', 'ew', ['cvar5']), ('ew|fhs5', 'ew|ewma', ['mdd', 'cvar5']), ('ew|fhs21', 'ew|ewma', ['mdd', 'cvar5']), ('hrp_lw|ewma', 'ew|ewma', ['cvar5']), ('hrp_lw|fhs5', 'ew|ewma', ['cvar5']), ('hrp_lw|fhs21', 'ew|ewma', ['cvar5']), ('hrp_lw|lgbm5', 'ew|ewma', ['cvar5']), ('mv_lw|ewma', 'ew|ewma', ['cvar5']), ('mv_lw|fhs5', 'ew|ewma', ['cvar5']), ('mv_lw|fhs21', 'ew|ewma', ['cvar5']), ('mv_lw|lgbm5', 'ew|ewma', ['cvar5']), ('ew|ewma', 'ew', ['cvar5']), ('hrp_lw|ewma', 'hrp_lw', ['cvar5'])].


### Lectura (regla pre-registrada aplicada; ver tablas 1-6)

- Ningun resultado es POSITIVO en MDD ni en Sharpe: el unico "positivo" es CVaR 5 % y viene de MENOR VOL/exposicion (los pesos de riesgo pagan Sharpe/Calmar; las escalas bajan retorno). No es mejora gratis.
- Shrinkage (LW/OAS/cc) = covarianza muestral: dif. de varianza realizada no significativa (DM p 0.24-0.50), 2/6 folds. HRP y risk parity tienen MAS varianza realizada que min-var muestral (DM p<0.01) y no mejoran su MDD/CVaR; frente a 1/N bajan vol pero Sharpe neto <= 1/N.
- Escala EWMA vs 1/N: MDD 6/6 folds (-34.5% -> -19.3%) pero Sharpe 0.44 -> 0.25 con costo 0.30%/lado (el hallazgo previo "mismo Sharpe" era con 0.05%). El IC bootstrap de MDD incluye 0 (el bootstrap por bloques rompe la dependencia de trayectoria: el MDD agrupado depende de un solo tramo, 2020).
- FHS 5/21d vs EWMA: MDD y CVaR mejoran 6/6 folds y en 2020/2022, pero con exposicion media menor (0.70/0.66 vs 0.77), Sharpe peor (6/6 folds) y Calmar menor: es mas conservador, no mas eficiente.
- LightGBM cuantil: pinball algo mejor en h5 (DM t -1.6, no significativo, 4-5/6 folds), peor en h21; no mejora el MDD vs EWMA y 2020 sigue siendo el unico tramo que decide el MDD agrupado. Cobertura Kupiec/CC aceptable en todos (no rechaza 5 %).
- Tilt Ridge-rank sobre HRP: turnover 1.75 -> 3.5/6.1 por anio, Sharpe y MDD sin mejora (2-3/6 folds).
- 2008 queda fuera de los folds (TRAIN del panel); solo se reporta para estrategias sin parametros ajustados.
