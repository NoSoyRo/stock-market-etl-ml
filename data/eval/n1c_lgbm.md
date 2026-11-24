# N1-C2 LightGBM pequeno (sustituto de TabPFN)

- NO VIABLE: venv scratchpad instalado (torch 2.5.1+cpu, tabpfn 2.0.9, pesos 44MB descargados) pero fit 1000x42 + predict 500 no termino en 300 s (swap 2035/2047 MB, carga 11-20, ~1 GB libre). Contexto de 3000 filas x walk-forward trimestral inviable. Sustituido por LightGBM.
- LightGBM 150 arboles, params {'objective': 'regression', 'learning_rate': 0.03, 'num_leaves': 7, 'min_data_in_leaf': 500, 'feature_fraction': 0.7, 'bagging_fraction': 0.5, 'bagging_freq': 1, 'lambda_l2': 50.0, 'num_threads': 2, 'verbose': -1, 'seed': 42}; target rank por fecha, features ranks40+rev1d+rev1d*last5; mismos 6 folds; N global DSR = 179.

| config | IC | t NW | alfa neto turn % | turnover | folds+ | IC 2024+ | IC T3 | DSR |
|---|---|---|---|---|---|---|---|---|
| lgbm_rank_aug_h5 | 0.0279 | 4.25 | -0.221 | 0.58 | 0/6 | 0.0168 | 0.0490 | 0.000 |
| lgbm_rank_aug_h5_top10_b10 | - | - | -0.103 | 0.35 | 1/6 | alfa 2024+ -0.104 | T3 alfa -0.043 | 0.000 |
| lgbm_rank_aug_h5_gate_disp_q50 | - | - | -0.312 | 0.58 | 0/6 | alfa 2024+ -0.363 | T3 alfa -0.170 | 0.000 |
| lgbm_rank_aug_h21 | 0.0165 | 1.87 | -0.476 | 0.66 | 1/6 | 0.0068 | 0.0342 | 0.000 |
| lgbm_rank_aug_h21_top10_b10 | - | - | -0.245 | 0.42 | 0/6 | alfa 2024+ -0.530 | T3 alfa 0.201 | 0.000 |
| lgbm_rank_aug_h21_gate_disp_q50 | - | - | -0.310 | 0.55 | 1/6 | alfa 2024+ -0.184 | T3 alfa -0.386 | 0.000 |
