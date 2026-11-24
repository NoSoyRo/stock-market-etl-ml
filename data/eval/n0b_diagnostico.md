# N0-B diagnostico (WoE+Logistic, 6 folds)

Base OOS: AUC 0.5165, IC 0.0167, t NW 2.60, alfa largo neto -0.526%

## 1. Persistencia por emisora-ano

340 celdas emisora-ano (35 emisoras). AUC medio 0.5277 (sd entre celdas 0.0629); IC medio 0.0558.

| metrica | rho medio t->t+1 | p perm. emisoras | p perm. desplaz. (ruido) | rho agrupado | p |
|---|---|---|---|---|---|
| auc | 0.014 | 0.410 | 0.377 | 0.007 | 0.900 |
| ic | -0.027 | 0.688 | 0.709 | -0.043 | 0.453 |

Predecibilidad vs liquidez (Spearman):

| par | rho | p | n |
|---|---|---|---|
| auc~adv | 0.170 | 0.002 | 340 |
| auc~nop | -0.189 | 0.000 | 340 |
| auc~cs | -0.122 | 0.025 | 340 |
| auc_por_emisora~adv | 0.464 | 0.005 | 35 |
| auc_por_emisora~nop | -0.403 | 0.016 | 35 |
| auc_por_emisora~cs | -0.121 | 0.488 | 35 |
| ic~adv | 0.183 | 0.001 | 340 |
| ic~nop | -0.172 | 0.001 | 340 |
| ic~cs | -0.118 | 0.029 | 340 |
| ic_por_emisora~adv | 0.518 | 0.001 | 35 |
| ic_por_emisora~nop | -0.396 | 0.019 | 35 |
| ic_por_emisora~cs | -0.089 | 0.612 | 35 |

Por ano: 2017: AUC 0.571 IC 0.160; 2018: AUC 0.517 IC 0.024; 2019: AUC 0.548 IC 0.087; 2020: AUC 0.508 IC 0.012; 2021: AUC 0.535 IC 0.073; 2022: AUC 0.504 IC 0.003; 2023: AUC 0.507 IC 0.018; 2024: AUC 0.520 IC 0.048; 2025: AUC 0.508 IC 0.021; 2026: AUC 0.563 IC 0.121

## 2. Cortes pre-registrados

Bonferroni alfa = 0.0063 (m=8). Tamano: SIN DATO: no hay capitalizacion ni acciones en circulacion en el repo (fundamentals.parquet solo tiene margenes); eje omitido, Bonferroni se mantiene con m=8

| celda | n | IC | t NW | p | alfa neto % | folds IC+ | folds alfa+ | cuenta |
|---|---|---|---|---|---|---|---|---|
| liquidez_ADV:T1_iliquido | 24356 | 0.0035 | 0.34 | 0.7317 | -0.568 | 3/6 | 0/6 | False |
| liquidez_ADV:T2 | 25030 | 0.0011 | 0.11 | 0.9103 | -0.654 | 3/6 | 0/6 | False |
| liquidez_ADV:T3_liquido | 25872 | 0.0350 | 3.27 | 0.0011 | -0.482 | 4/6 | 0/6 | False |
| volatilidad_mercado:alta | 24255 | 0.0232 | 2.13 | 0.0329 | -0.546 | 6/6 | 0/6 | False |
| volatilidad_mercado:baja | 51003 | 0.0134 | 1.73 | 0.0832 | -0.545 | 5/6 | 0/6 | False |

Complementarios (no cuentan): liq_sinop:T3_liquido: IC 0.0117 t 1.06 alfa -0.517%; liq_cs:T3_liquido: IC 0.0295 t 2.91 alfa -0.579%; liq_sinop:T1_iliquido: IC -0.0013 t -0.13 alfa -0.621%; liq_cs:T1_iliquido: IC 0.0133 t 1.35 alfa -0.520%

## 3. Adversarial validation (2008-2023 vs 2024-2026)

AUC adversarial 0.8686. Top features (ganancia): cs_ret20_rank 0.21, cs_ret5_rank 0.20, cs_vol_rank 0.14, mkt_trend 0.10, usdmxn_vs_ma20 0.06, cs_rsi_rank 0.06

| variante | AUC | IC | t NW | alfa largo neto % | AUC 2024+ | IC 2024+ | t 2024+ |
|---|---|---|---|---|---|---|---|
| todas | 0.5165 | 0.0167 | 2.60 | -0.526 | 0.5184 | 0.0034 | 0.29 |
| drop_top1 (adv 0.844) | 0.5165 | 0.0167 | 2.60 | -0.526 | 0.5184 | 0.0034 | 0.29 |
| drop_top3 (adv 0.737) | 0.5151 | 0.0160 | 2.58 | -0.575 | 0.5196 | 0.0082 | 0.73 |
| drop_top5 (adv 0.769) | 0.5114 | 0.0125 | 2.00 | -0.607 | 0.5153 | 0.0085 | 0.74 |
| importance_weighting | 0.5139 | 0.0164 | 2.64 | -0.528 | 0.5130 | 0.0029 | 0.25 |

Quitadas: drop_top1: ['cs_ret20_rank']; drop_top3: ['cs_ret20_rank', 'cs_ret5_rank', 'cs_vol_rank']; drop_top5: ['cs_ret20_rank', 'cs_ret5_rank', 'cs_vol_rank', 'mkt_trend', 'usdmxn_vs_ma20']

## 4. Monotonia por decil de score

### decil_seccional_por_fecha

| decil | n | exceso % | ret % | hit | IC dentro |
|---|---|---|---|---|---|
| 1 | 6851 | -0.159 | -0.030 | 0.472 | 0.010 |
| 2 | 7069 | -0.101 | 0.022 | 0.484 | 0.016 |
| 3 | 7453 | -0.009 | 0.115 | 0.485 | 0.020 |
| 4 | 7683 | 0.055 | 0.186 | 0.496 | 0.038 |
| 5 | 7997 | 0.014 | 0.139 | 0.494 | 0.020 |
| 6 | 7019 | 0.002 | 0.130 | 0.483 | 0.017 |
| 7 | 7622 | 0.009 | 0.134 | 0.494 | 0.025 |
| 8 | 7518 | 0.048 | 0.173 | 0.495 | 0.037 |
| 9 | 7001 | 0.033 | 0.161 | 0.496 | 0.025 |
| 10 | 9039 | 0.073 | 0.195 | 0.502 | 0.035 |

Monotonia rho 0.77 (p 0.009); d10-d9 = 0.050% (t NW 0.60, d10>d9 en 5/6 folds); d10-d1 = 0.231%

### decil_pooled_por_fold

| decil | n | exceso % | ret % | hit | IC dentro |
|---|---|---|---|---|---|
| 1 | 7528 | -0.051 | -0.044 | 0.462 | -0.011 |
| 2 | 7525 | -0.120 | -0.079 | 0.461 | 0.006 |
| 3 | 7523 | 0.090 | 0.216 | 0.494 | -0.024 |
| 4 | 7526 | -0.027 | 0.089 | 0.487 | -0.007 |
| 5 | 7526 | 0.118 | 0.224 | 0.495 | -0.027 |
| 6 | 7522 | -0.093 | 0.085 | 0.482 | -0.025 |
| 7 | 7526 | -0.106 | 0.059 | 0.486 | -0.021 |
| 8 | 7523 | 0.029 | 0.294 | 0.512 | 0.001 |
| 9 | 7525 | 0.053 | 0.369 | 0.514 | 0.014 |
| 10 | 7528 | 0.106 | 0.044 | 0.512 | -0.063 |

Monotonia rho 0.37 (p 0.293); d10-d9 = 0.013% (t NW 0.07, d10>d9 en 2/6 folds); d10-d1 = 0.157%

