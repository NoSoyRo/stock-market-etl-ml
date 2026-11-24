# N4 combinacion pre-registrada (Ridge-rank h=21 [+ rev1d], top-10 buffer, costo por emisora, capa de vol)

Diseno fijo en el docstring de `scripts/n4_combinacion.py` (escrito antes de correr). Alfa = exceso % por rebalanceo vs 1/N del universo elegible (relativo, benchmark sin costo, como N2-B). Folds: F1 = mas reciente. Config con capa: alfa aritmetico aproximado (ver docstring).
- Verificacion de la replica del portafolio contra `n2b_costos.run`: {'n2b_alpha_net': -0.1355095482381622, 'n4_alpha_net': -0.1355095482381622, 'n2b_n': 113, 'n4_n': 113}
- N de ensayos del log: inicio 341, final 350. Rangos de folds: {'1': ['2025-04-16', '2026-10-09'], '2': ['2023-09-25', '2025-04-15'], '3': ['2022-03-11', '2023-09-22'], '4': ['2020-08-26', '2022-03-10'], '5': ['2019-02-08', '2020-08-25'], '6': ['2017-07-21', '2019-02-07']}
- IC por fecha del score (todas las fechas OOS): S1: 0.0334 (t NW 3.64), 2024+ 0.0166; S2: 0.0308 (t NW 4.64), 2024+ 0.0175

## Tabla clave (costo x1 salvo indicacion)

| config | IC (t) | bruto % | costo % | alfa neto % | t | folds+ | 2024+ % | boot95 % | MDD | Sharpe neto | DSR | (a) | (b) | (c) | (d) | todo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_sinCapa_x1 | 0.0334 (3.64) | 0.274 | 0.409 | -0.136 | -0.70 | 2/6 | -0.295 | [-0.504, 0.240] | -0.382 | 0.26 | 0.000 | True | False | False | False | False |
| S2_sinCapa_x1 | 0.0308 (4.64) | 0.403 | 0.440 | -0.037 | -0.19 | 3/6 | -0.087 | [-0.332, 0.254] | -0.386 | 0.29 | 0.001 | True | False | False | False | False |
| S1_conCapa_x1 | 0.0334 (3.64) | 0.274 | 0.409 | -0.289 | -2.25 | 2/6 | -0.371 | [-0.524, -0.054] | -0.194 | 0.00 | 0.000 | True | False | False | False | False |
| S2_conCapa_x1 | 0.0308 (4.64) | 0.403 | 0.440 | -0.279 | -2.50 | 1/6 | -0.316 | [-0.472, -0.091] | -0.193 | -0.03 | 0.000 | True | False | False | False | False |
| S1_sinCapa_x05 | 0.0334 (3.64) | 0.274 | 0.267 | 0.007 | 0.03 | 4/6 | -0.152 | [-0.373, 0.358] | -0.369 | 0.36 | 0.002 | True | True | False | False | False |
| S1_sinCapa_x2 | 0.0334 (3.64) | 0.274 | 0.582 | -0.308 | -1.59 | 2/6 | -0.460 | [-0.707, 0.072] | -0.406 | 0.15 | 0.000 | True | False | False | False | False |
| S2_sinCapa_x05 | 0.0308 (4.64) | 0.403 | 0.282 | 0.121 | 0.60 | 3/6 | 0.083 | [-0.171, 0.435] | -0.385 | 0.40 | 0.011 | True | False | True | False | False |
| S2_sinCapa_x2 | 0.0308 (4.64) | 0.403 | 0.639 | -0.236 | -1.18 | 3/6 | -0.291 | [-0.534, 0.063] | -0.390 | 0.16 | 0.000 | True | False | False | False | False |

MDD/Sharpe: serie diaria neta (retorno absoluto, largo, resto en efectivo), solo dias en posicion; 'conCapa' = con vol targeting EWMA 10 %.

## Referencias

- 1/N (mismo rebalanceo y costo x1): ret mensual neto 0.536 %, turnover 0.100; diario {'n': 2325, 'ret_ann': 0.057407086440753874, 'vol': 0.14882195437313528, 'mdd': -0.3371778087523437, 'sharpe': 0.44972576121456476}; con capa {'n': 2325, 'ret_ann': 0.014041044308642059, 'vol': 0.10403417244036188, 'mdd': -0.17522699026778166, 'sharpe': 0.1860653962456706}.
- Score aleatorio (200 perm.): {'n_perm': 200, 'alpha_net_media': -0.4303047974505199, 'alpha_net_sd': 0.19553442691016498, 'alpha_net_p95': -0.1343130803352872, 'bruto_media': 0.02691741020524327, 'bruto_sd': 0.1945200702722874, 'bruto_p95': 0.3234559272950396, 'turnover_media': 0.43936479842674536, 'percentil_S1_neto': 94.5, 'percentil_S2_neto': 97.5, 'percentil_S1_bruto': 88.0, 'percentil_S2_bruto': 97.5}
- Referencia costo fijo 0.30 % S1_sinCapa_x1: {'alpha_net_pct': 0.04870068260359357, 't': 0.24921920699532724, 'folds_pos': 4, 'alpha_2024p_pct': -0.09063836402216291, 'boot95_pct': [-0.32494309358811146, 0.4153242707914913]}
- Referencia costo fijo 0.30 % S2_sinCapa_x1: {'alpha_net_pct': 0.15530966266348495, 't': 0.7701613282834229, 'folds_pos': 3, 'alpha_2024p_pct': 0.1373075655618048, 'boot95_pct': [-0.16316902173728812, 0.46301514505286906]}

## Estres (MDD / Sharpe, serie diaria neta)

| config | 2020 MDD | 2020 SR | 2022 MDD | 2022 SR |
|---|---|---|---|---|
| 1/N sin capa | -0.321 | -1.06 | -0.165 | 0.02 |
| 1/N con capa | -0.137 | -1.84 | -0.146 | -0.57 |
| S1_sinCapa_x1 | -0.350 | -1.01 | -0.209 | -0.02 |
| S2_sinCapa_x1 | -0.379 | -0.93 | -0.213 | -0.30 |
| S1_conCapa_x1 | -0.136 | -1.77 | -0.167 | -0.52 |
| S1_conCapa_x1 (misma cartera sin capa) | -0.350 | -1.01 | -0.209 | -0.02 |
| S2_conCapa_x1 | -0.132 | -1.46 | -0.178 | -0.86 |
| S2_conCapa_x1 (misma cartera sin capa) | -0.379 | -0.93 | -0.213 | -0.30 |

## Folds por separado (alfa neto % / IC / MDD diario)

| config | F1 | F2 | F3 | F4 | F5 | F6 |
|---|---|---|---|---|---|---|
| S1_sinCapa_x1 | 0.36 / 0.003 / -0.15 | -0.86 / 0.042 / -0.22 | -0.59 / 0.075 / -0.21 | -0.02 / 0.040 / -0.10 | -0.12 / 0.048 / -0.37 | 0.44 / -0.009 / -0.20 |
| S2_sinCapa_x1 | 0.29 / 0.011 / -0.16 | -0.46 / 0.032 / -0.18 | -0.45 / 0.062 / -0.21 | -0.45 / 0.032 / -0.11 | 0.51 / 0.041 / -0.39 | 0.34 / 0.005 / -0.20 |
| S1_conCapa_x1 | 0.06 / 0.003 / -0.12 | -0.71 / 0.042 / -0.16 | -0.67 / 0.075 / -0.16 | -0.20 / 0.040 / -0.06 | -0.30 / 0.048 / -0.16 | 0.10 / -0.009 / -0.14 |
| S2_conCapa_x1 | -0.08 / 0.011 / -0.15 | -0.54 / 0.032 / -0.15 | -0.59 / 0.062 / -0.17 | -0.33 / 0.032 / -0.06 | 0.00 / 0.041 / -0.14 | -0.12 / 0.005 / -0.14 |
| S1_sinCapa_x05 | 0.50 / 0.003 / -0.14 | -0.72 / 0.042 / -0.20 | -0.47 / 0.075 / -0.20 | 0.15 / 0.040 / -0.10 | 0.05 / 0.048 / -0.37 | 0.57 / -0.009 / -0.19 |
| S1_sinCapa_x2 | 0.19 / 0.003 / -0.16 | -1.01 / 0.042 / -0.23 | -0.75 / 0.075 / -0.21 | -0.20 / 0.040 / -0.10 | -0.32 / 0.048 / -0.37 | 0.27 / -0.009 / -0.20 |
| S2_sinCapa_x05 | 0.47 / 0.011 / -0.15 | -0.30 / 0.032 / -0.17 | -0.29 / 0.062 / -0.21 | -0.28 / 0.032 / -0.11 | 0.67 / 0.041 / -0.38 | 0.48 / 0.005 / -0.19 |
| S2_sinCapa_x2 | 0.09 / 0.011 / -0.17 | -0.65 / 0.032 / -0.20 | -0.65 / 0.062 / -0.22 | -0.64 / 0.032 / -0.11 | 0.31 / 0.041 / -0.39 | 0.15 / 0.005 / -0.20 |

## Replica de entropia (h=21, pe100, T3 transversal, S1) - NO independiente (mismos datos)

- IC 0.0503 (t NW 4.10), folds IC+ 6/6, IC 2024+ 0.0598 (universo completo 0.0334).
- IC por fold: {'1': 0.06130161170375648, '2': 0.06302395119875841, '3': 0.09388246714724865, '4': 0.0424386239296265, '5': 0.040508420457006575, '6': 0.0012817516673557787}
- Alfa neto -0.090 % (t -0.29), bruto 0.229, costo 0.319, folds+ 3/6, 2024+ 0.077, boot95 [-0.6441694899306339, 0.5022815254392604], DSR 0.001.
- replica NO independiente: mismos datos y misma historia que la celda hallada en N3 (48 celdas)
