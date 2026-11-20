# N1-B cambio de target (cross-sectional) y features simples

- Convencion: entrada Open t+1, salida Close t+h+1 (h=5 == ret_fwd del repo); mismos 6 folds de test; purga h+1 dias + embargo 10; NW lag max(5,h).
- Auditoria etiqueta (r5 vs ret_fwd del panel): {'r5_vs_ret_fwd_max_abs_dif': 3.099921471516609e-06, 'r5_vs_ret_fwd_frac_gt_1e-3': 0.0, 'n_nan_r1': 0, 'n_nan_r5': 0, 'n_nan_r21': 486}
- Ridge alpha=1000.0 sobre 40 features convertidas a percentil por fecha; objetivo winsorizado 1/99 en train.
- Targets: rank = percentil por fecha del ret_h; resid = ret_h - beta(t-1, 252d)*ret_h EW; dm = ret_h - media expanding del ticker (rezagada h+1); raw = ret_h.
- IC y alfa se miden SIEMPRE contra ret_h crudo. Alfa = largo tercil superior vs media del universo, rebalanceo cada h dias; `neto full` = 2*0.30% por rebalanceo (como la linea base), `neto turn` = 0.30%*2*turnover real.
- DSR (sobre el exceso neto turn por rebalanceo): N global de ensayos unicos en trial_log = 121; N familia n1b = 33.

## Horizonte 1 dias

| config | AUC | IC | t NW | alfa full % | alfa turn % | turnover | folds+ full/turn | IC 2024+ | IC T3 | dIC vs woe_dir | DSR glob |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ridge_rank_h1 | 0.5065 | 0.0246 | 4.78 | -0.568 | -0.191 | 0.37 | 0/0 | 0.0163 | 0.0400 | 0.0075 | 0.000 |
| ridge_resid_h1 | 0.5059 | 0.0244 | 4.84 | -0.553 | -0.166 | 0.36 | 0/0 | 0.0149 | 0.0422 | 0.0073 | 0.000 |
| ridge_dm_h1 | 0.5024 | 0.0149 | 2.92 | -0.583 | -0.179 | 0.33 | 0/0 | 0.0051 | 0.0373 | -0.0023 | 0.000 |
| ridge_raw_h1 | 0.5047 | 0.0224 | 4.45 | -0.563 | -0.189 | 0.38 | 0/0 | 0.0110 | 0.0398 | 0.0052 | 0.000 |
| ridge_rank_aug_h1 | 0.5066 | 0.0247 | 4.79 | -0.570 | -0.194 | 0.37 | 0/0 | 0.0167 | 0.0395 | 0.0075 | 0.000 |
| woe_dir_h1 | 0.5141 | 0.0171 | 3.51 | -0.568 | -0.215 | 0.41 | 0/0 | 0.0132 | 0.0329 | 0.0000 | 0.000 |
| woe_rank_h1 | 0.5080 | 0.0184 | 3.48 | -0.554 | -0.185 | 0.39 | 0/0 | 0.0037 | 0.0350 | 0.0013 | 0.000 |
| woe_resid_h1 | 0.5046 | 0.0141 | 2.73 | -0.578 | -0.192 | 0.36 | 0/0 | 0.0018 | 0.0320 | -0.0031 | 0.000 |
| woe_dm_h1 | 0.5119 | 0.0097 | 1.78 | -0.573 | -0.213 | 0.40 | 0/0 | -0.0009 | 0.0285 | -0.0074 | 0.000 |
| rev1d_h1 | 0.5051 | 0.0185 | 4.23 | -0.585 | -0.400 | 0.69 | 0/0 | 0.0171 | 0.0324 | 0.0013 | 0.000 |

## Horizonte 5 dias

| config | AUC | IC | t NW | alfa full % | alfa turn % | turnover | folds+ full/turn | IC 2024+ | IC T3 | dIC vs woe_dir | DSR glob |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ridge_rank_h5 | 0.5130 | 0.0298 | 4.40 | -0.469 | -0.215 | 0.58 | 0/0 | 0.0237 | 0.0576 | 0.0131 | 0.000 |
| ridge_resid_h5 | 0.5124 | 0.0313 | 4.57 | -0.472 | -0.191 | 0.53 | 0/1 | 0.0292 | 0.0628 | 0.0146 | 0.000 |
| ridge_dm_h5 | 0.5044 | 0.0197 | 2.72 | -0.481 | -0.165 | 0.47 | 0/1 | 0.0180 | 0.0513 | 0.0031 | 0.000 |
| ridge_raw_h5 | 0.5090 | 0.0273 | 3.99 | -0.517 | -0.247 | 0.55 | 0/0 | 0.0200 | 0.0579 | 0.0107 | 0.000 |
| ridge_rank_aug_h5 | 0.5135 | 0.0316 | 4.73 | -0.472 | -0.220 | 0.58 | 0/0 | 0.0259 | 0.0602 | 0.0149 | 0.000 |
| woe_dir_h5 | 0.5165 | 0.0167 | 2.60 | -0.561 | -0.373 | 0.69 | 0/0 | 0.0034 | 0.0350 | 0.0000 | 0.000 |
| woe_rank_h5 | 0.5095 | 0.0210 | 3.05 | -0.523 | -0.289 | 0.61 | 0/0 | 0.0115 | 0.0447 | 0.0043 | 0.000 |
| woe_resid_h5 | 0.5101 | 0.0264 | 3.79 | -0.534 | -0.271 | 0.56 | 0/1 | 0.0113 | 0.0505 | 0.0098 | 0.000 |
| woe_dm_h5 | 0.5159 | 0.0155 | 2.11 | -0.516 | -0.260 | 0.57 | 0/0 | 0.0080 | 0.0416 | -0.0012 | 0.000 |
| rev1d_h5 | 0.5053 | 0.0192 | 4.71 | -0.500 | -0.300 | 0.67 | 0/0 | 0.0176 | 0.0292 | 0.0026 | 0.000 |

## Horizonte 21 dias

| config | AUC | IC | t NW | alfa full % | alfa turn % | turnover | folds+ full/turn | IC 2024+ | IC T3 | dIC vs woe_dir | DSR glob |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ridge_rank_h21 | 0.5115 | 0.0321 | 3.50 | -0.466 | -0.272 | 0.68 | 1/1 | 0.0143 | 0.0458 | 0.0202 | 0.000 |
| ridge_resid_h21 | 0.5119 | 0.0320 | 3.18 | -0.366 | -0.151 | 0.64 | 0/2 | 0.0244 | 0.0433 | 0.0201 | 0.000 |
| ridge_dm_h21 | 0.4785 | -0.0096 | -0.68 | -0.674 | -0.376 | 0.50 | 0/1 | 0.0037 | 0.0326 | -0.0215 | 0.000 |
| ridge_raw_h21 | 0.4958 | 0.0292 | 3.16 | -0.449 | -0.249 | 0.67 | 1/2 | 0.0103 | 0.0508 | 0.0173 | 0.000 |
| ridge_rank_aug_h21 | 0.5118 | 0.0334 | 3.64 | -0.381 | -0.182 | 0.67 | 1/2 | 0.0166 | 0.0481 | 0.0214 | 0.000 |
| woe_dir_h21 | 0.5168 | 0.0119 | 1.33 | -0.398 | -0.177 | 0.63 | 1/3 | 0.0032 | 0.0222 | 0.0000 | 0.000 |
| woe_rank_h21 | 0.5018 | 0.0204 | 2.16 | -0.132 | 0.059 | 0.68 | 2/4 | -0.0066 | 0.0360 | 0.0084 | 0.011 |
| woe_resid_h21 | 0.5030 | 0.0245 | 2.23 | -0.448 | -0.222 | 0.62 | 1/1 | 0.0008 | 0.0197 | 0.0126 | 0.000 |
| woe_dm_h21 | 0.5257 | 0.0025 | 0.19 | -0.379 | -0.054 | 0.46 | 1/3 | 0.0003 | 0.0060 | -0.0095 | 0.002 |
| rev1d_h21 | 0.5094 | 0.0202 | 5.20 | -0.075 | 0.100 | 0.71 | 1/4 | 0.0133 | 0.0236 | 0.0083 | 0.018 |

## Por tercil de liquidez (IC / t / alfa neto turn %)

| config | T1 | T2 | T3 |
|---|---|---|---|
| ridge_rank_h1 | 0.0080 / 0.98 / -0.22 | 0.0168 / 2.04 / -0.21 | 0.0400 / 4.99 / -0.16 |
| ridge_resid_h1 | 0.0089 / 1.08 / -0.22 | 0.0178 / 2.17 / -0.19 | 0.0422 / 5.19 / -0.15 |
| ridge_dm_h1 | 0.0010 / 0.12 / -0.21 | 0.0007 / 0.09 / -0.23 | 0.0373 / 4.51 / -0.12 |
| ridge_raw_h1 | 0.0068 / 0.84 / -0.22 | 0.0095 / 1.15 / -0.22 | 0.0398 / 5.01 / -0.16 |
| ridge_rank_aug_h1 | 0.0083 / 1.02 / -0.22 | 0.0169 / 2.06 / -0.20 | 0.0395 / 4.86 / -0.17 |
| woe_dir_h1 | 0.0076 / 0.97 / -0.22 | 0.0104 / 1.32 / -0.25 | 0.0329 / 4.40 / -0.21 |
| woe_rank_h1 | 0.0021 / 0.25 / -0.23 | 0.0136 / 1.64 / -0.23 | 0.0350 / 4.25 / -0.16 |
| woe_resid_h1 | -0.0050 / -0.60 / -0.22 | 0.0060 / 0.69 / -0.24 | 0.0320 / 3.85 / -0.16 |
| woe_dm_h1 | -0.0047 / -0.54 / -0.24 | 0.0020 / 0.23 / -0.25 | 0.0285 / 3.44 / -0.20 |
| rev1d_h1 | 0.0058 / 0.80 / -0.42 | 0.0189 / 2.59 / -0.41 | 0.0324 / 4.75 / -0.34 |
| ridge_rank_h5 | 0.0069 / 0.66 / -0.27 | 0.0143 / 1.31 / -0.34 | 0.0576 / 5.22 / -0.10 |
| ridge_resid_h5 | 0.0048 / 0.45 / -0.33 | 0.0200 / 1.80 / -0.23 | 0.0628 / 5.52 / -0.10 |
| ridge_dm_h5 | -0.0055 / -0.48 / -0.23 | 0.0046 / 0.39 / -0.30 | 0.0513 / 4.38 / -0.06 |
| ridge_raw_h5 | 0.0044 / 0.41 / -0.34 | 0.0085 / 0.76 / -0.35 | 0.0579 / 5.16 / -0.10 |
| ridge_rank_aug_h5 | 0.0071 / 0.68 / -0.28 | 0.0164 / 1.52 / -0.40 | 0.0602 / 5.53 / -0.10 |
| woe_dir_h5 | 0.0035 / 0.34 / -0.35 | 0.0011 / 0.11 / -0.51 | 0.0350 / 3.27 / -0.24 |
| woe_rank_h5 | -0.0008 / -0.08 / -0.22 | 0.0034 / 0.34 / -0.45 | 0.0447 / 3.91 / -0.21 |
| woe_resid_h5 | 0.0019 / 0.18 / -0.26 | 0.0104 / 0.97 / -0.36 | 0.0505 / 4.39 / -0.19 |
| woe_dm_h5 | -0.0115 / -1.04 / -0.27 | -0.0038 / -0.34 / -0.41 | 0.0416 / 3.59 / -0.12 |
| rev1d_h5 | 0.0088 / 1.30 / -0.24 | 0.0093 / 1.37 / -0.48 | 0.0292 / 4.32 / -0.28 |
| ridge_rank_h21 | 0.0134 / 0.96 / -0.12 | 0.0366 / 2.68 / -0.42 | 0.0458 / 3.10 / -0.08 |
| ridge_resid_h21 | 0.0045 / 0.28 / -0.12 | 0.0450 / 2.96 / -0.11 | 0.0433 / 2.66 / -0.21 |
| ridge_dm_h21 | -0.0465 / -2.16 / -0.65 | -0.0321 / -1.46 / -0.41 | 0.0326 / 1.53 / 0.01 |
| ridge_raw_h21 | 0.0120 / 0.88 / -0.33 | 0.0238 / 1.73 / -0.30 | 0.0508 / 3.32 / -0.03 |
| ridge_rank_aug_h21 | 0.0150 / 1.07 / -0.05 | 0.0365 / 2.65 / -0.30 | 0.0481 / 3.27 / -0.03 |
| woe_dir_h21 | 0.0057 / 0.37 / -0.23 | -0.0064 / -0.45 / -0.51 | 0.0222 / 1.51 / -0.28 |
| woe_rank_h21 | 0.0064 / 0.42 / -0.06 | 0.0154 / 1.04 / 0.15 | 0.0360 / 2.39 / 0.02 |
| woe_resid_h21 | 0.0062 / 0.37 / -0.35 | 0.0181 / 1.08 / -0.25 | 0.0197 / 1.13 / -0.27 |
| woe_dm_h21 | -0.0117 / -0.59 / -0.61 | -0.0316 / -1.56 / -0.13 | 0.0060 / 0.30 / -0.16 |
| rev1d_h21 | 0.0129 / 2.07 / 0.20 | 0.0163 / 2.66 / -0.37 | 0.0236 / 3.69 / 0.03 |

## IC por anio (mejor IC de cada horizonte; entre parentesis alfa neto turn %)

| anio | ridge_rank_aug_h1 | ridge_rank_aug_h5 | ridge_rank_aug_h21 |
|---|---|---|---|
| 2017 | 0.0368 (-0.17) | -0.0138 (-0.24) | -0.0006 (-0.13) |
| 2018 | 0.0326 (-0.15) | 0.0143 (-0.21) | -0.0105 (0.06) |
| 2019 | 0.0200 (-0.21) | 0.0396 (-0.34) | 0.0403 (0.12) |
| 2020 | 0.0514 (-0.15) | 0.0498 (-0.20) | 0.0469 (-0.28) |
| 2021 | 0.0196 (-0.23) | 0.0401 (-0.17) | 0.0340 (-0.04) |
| 2022 | 0.0288 (-0.20) | 0.0395 (-0.12) | 0.0620 (-0.22) |
| 2023 | 0.0125 (-0.22) | 0.0420 (-0.23) | 0.0891 (-0.81) |
| 2024 | 0.0235 (-0.18) | 0.0465 (-0.02) | 0.0521 (0.08) |
| 2025 | 0.0261 (-0.18) | 0.0239 (-0.23) | -0.0402 (-0.66) |
| 2026 | -0.0025 (-0.26) | 0.0037 (-0.49) | 0.0442 (0.27) |
| 2024+ | 0.0167 (-0.20) | 0.0259 (-0.23) | 0.0166 (-0.14) |

## Calendario (serie de mercado EW; last5 es igual para todas las emisoras: no hay IC cross-sectional)

| h | ret last5 % | ret resto % | dif % | t HAC | entrada dtme=5 bruto % | t | neto % | meses |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.140 | 0.019 | 0.121 | 2.20 | 0.160 | 1.98 | -0.440 | 224 |
| 5 | 0.492 | 0.151 | 0.341 | 2.52 | 0.550 | 3.41 | -0.050 | 224 |
| 21 | 1.003 | 0.940 | 0.063 | 0.36 | 0.988 | 2.95 | 0.388 | 223 |

## Veredicto (t>2 y alfa neto turn>0 en >=4/6 folds, o DSR>0.95; hallazgo si dIC vs woe_dir > 0.03)

- ridge_rank_h1: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0075, DSR>0.95=False, pasa=False
- ridge_resid_h1: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0073, DSR>0.95=False, pasa=False
- ridge_dm_h1: t>2=True, alfa>0=False, folds mayoria=False, dIC=-0.0023, DSR>0.95=False, pasa=False
- ridge_raw_h1: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0052, DSR>0.95=False, pasa=False
- ridge_rank_aug_h1: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0075, DSR>0.95=False, pasa=False
- woe_dir_h1: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0000, DSR>0.95=False, pasa=False
- woe_rank_h1: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0013, DSR>0.95=False, pasa=False
- woe_resid_h1: t>2=True, alfa>0=False, folds mayoria=False, dIC=-0.0031, DSR>0.95=False, pasa=False
- woe_dm_h1: t>2=False, alfa>0=False, folds mayoria=False, dIC=-0.0074, DSR>0.95=False, pasa=False
- rev1d_h1: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0013, DSR>0.95=False, pasa=False
- ridge_rank_h5: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0131, DSR>0.95=False, pasa=False
- ridge_resid_h5: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0146, DSR>0.95=False, pasa=False
- ridge_dm_h5: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0031, DSR>0.95=False, pasa=False
- ridge_raw_h5: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0107, DSR>0.95=False, pasa=False
- ridge_rank_aug_h5: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0149, DSR>0.95=False, pasa=False
- woe_dir_h5: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0000, DSR>0.95=False, pasa=False
- woe_rank_h5: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0043, DSR>0.95=False, pasa=False
- woe_resid_h5: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0098, DSR>0.95=False, pasa=False
- woe_dm_h5: t>2=True, alfa>0=False, folds mayoria=False, dIC=-0.0012, DSR>0.95=False, pasa=False
- rev1d_h5: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0026, DSR>0.95=False, pasa=False
- ridge_rank_h21: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0202, DSR>0.95=False, pasa=False
- ridge_resid_h21: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0201, DSR>0.95=False, pasa=False
- ridge_dm_h21: t>2=False, alfa>0=False, folds mayoria=False, dIC=-0.0215, DSR>0.95=False, pasa=False
- ridge_raw_h21: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0173, DSR>0.95=False, pasa=False
- ridge_rank_aug_h21: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0214, DSR>0.95=False, pasa=False
- woe_dir_h21: t>2=False, alfa>0=False, folds mayoria=False, dIC=0.0000, DSR>0.95=False, pasa=False
- woe_rank_h21: t>2=True, alfa>0=True, folds mayoria=True, dIC=0.0084, DSR>0.95=False, pasa=True
- woe_resid_h21: t>2=True, alfa>0=False, folds mayoria=False, dIC=0.0126, DSR>0.95=False, pasa=False
- woe_dm_h21: t>2=False, alfa>0=False, folds mayoria=False, dIC=-0.0095, DSR>0.95=False, pasa=False
- rev1d_h21: t>2=True, alfa>0=True, folds mayoria=True, dIC=0.0083, DSR>0.95=False, pasa=True
