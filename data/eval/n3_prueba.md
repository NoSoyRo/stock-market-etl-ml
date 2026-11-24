# N3 Prueba B: features OHLCV de microestructura/memoria

Cobertura de eventos (Paso 1): insuficiente para Prueba A (ver docstring de scripts/n3_prueba.py): noticias por emisora solo WALMEX 2018-2020 (~900 titulares); archivo PIT con 304 de 556 titulares en 2026; flujo (hechos) desde 2025-06-30 y no es de hechos relevantes; EPS solo emisoras US; events.csv solo macro (banxico/fomc/inpc).

- Umbrales pre-registrados: {'dIC_min': 0.005, 'dIC_t_min': 2.0, 'gate_ic_min': 0.05, 'gate_t_min': 2.9}. N global de ensayos: inicio 277, final 341.
- Cobertura de features (fraccion no-NaN del panel): {'cs60': 1.0, 'amihud60': 0.999, 'gk60': 1.0, 'pk60': 1.0, 'gaprng60': 0.978, 'hurst100': 0.996, 'pe100': 0.996}

## (i) Modelos: IC y alfa neto (tercil superior, rebalanceo cada h)

| h | modelo | IC (t NW) | folds IC+ | dIC (t) | dIC folds+ | material | aN var % | f+ | 2024+ | T3 var | aN fix030 % | f+ | T3 fix | DSR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | base | 0.0313 (4.68) | 6/6 | - (-) | - | - | -0.514 | 0/6 | -0.536 | -0.315 | -0.216 | 0/6 | -0.102 | 0.000 |
| 5 | micro | 0.0297 (4.36) | 5/6 | -0.0016 (-0.92) | 2 | False | -0.542 | 0/6 | -0.564 | -0.346 | -0.240 | 0/6 | -0.119 | 0.000 |
| 5 | mem | 0.0299 (4.37) | 6/6 | -0.0014 (-1.05) | 3 | False | -0.533 | 0/6 | -0.571 | -0.291 | -0.242 | 0/6 | -0.078 | 0.000 |
| 5 | all | 0.0273 (3.94) | 5/6 | -0.0040 (-1.88) | 3 | False | -0.556 | 0/6 | -0.628 | -0.345 | -0.262 | 0/6 | -0.126 | 0.000 |
| 21 | base | 0.0334 (3.64) | 5/6 | - (-) | - | - | -0.483 | 0/6 | -0.451 | -0.388 | -0.166 | 2/6 | -0.172 | 0.000 |
| 21 | micro | 0.0362 (3.89) | 6/6 | 0.0028 (0.45) | 4 | False | -0.116 | 2/6 | -0.094 | -0.239 | 0.192 | 4/6 | -0.031 | 0.000 |
| 21 | mem | 0.0295 (2.53) | 4/6 | -0.0038 (-0.50) | 3 | False | -0.264 | 2/6 | -0.812 | -0.050 | 0.016 | 2/6 | 0.147 | 0.000 |
| 21 | all | 0.0346 (3.17) | 5/6 | 0.0012 (0.14) | 4 | False | 0.100 | 5/6 | -0.543 | 0.130 | 0.394 | 5/6 | 0.336 | 0.007 |

## (ii) Gate por terciles de Hurst / entropia de permutacion

| h | score | variable | tipo | tercil | n | IC (t) | folds IC+ | IC 2024+ | aN % (costo emisora) | f+ | util |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | all | hurst100 | transversal | T1 | 24334 | 0.0301 (2.72) | 5 | 0.0243 | -0.638 | 0 | False |
| 5 | all | hurst100 | transversal | T2 | 24999 | 0.0484 (4.59) | 6 | 0.0434 | -0.491 | 0 | False |
| 5 | all | hurst100 | transversal | T3 | 25847 | 0.0105 (1.06) | 3 | -0.0017 | -0.525 | 0 | False |
| 5 | all | hurst100 | mercado | T1 | 20681 | 0.0022 (0.18) | 3 | 0.0232 | -0.693 | 0 | False |
| 5 | all | hurst100 | mercado | T2 | 28853 | 0.0285 (2.62) | 6 | 0.0205 | -0.675 | 0 | False |
| 5 | all | hurst100 | mercado | T3 | 25724 | 0.0461 (4.11) | 4 | -0.0194 | -0.498 | 1 | False |
| 5 | all | pe100 | transversal | T1 | 24329 | 0.0181 (1.63) | 5 | 0.0059 | -0.486 | 0 | False |
| 5 | all | pe100 | transversal | T2 | 24995 | 0.0214 (1.96) | 4 | -0.0094 | -0.634 | 0 | False |
| 5 | all | pe100 | transversal | T3 | 25848 | 0.0373 (3.84) | 6 | 0.0355 | -0.578 | 0 | False |
| 5 | all | pe100 | mercado | T1 | 8187 | 0.0150 (0.70) | 3 | -0.0542 | -0.700 | 0 | False |
| 5 | all | pe100 | mercado | T2 | 22533 | 0.0383 (3.42) | 5 | 0.0087 | -0.460 | 0 | False |
| 5 | all | pe100 | mercado | T3 | 44538 | 0.0241 (2.85) | 5 | 0.0256 | -0.552 | 0 | False |
| 5 | base | hurst100 | transversal | T1 | 24334 | 0.0321 (2.89) | 5 | 0.0221 | -0.600 | 0 | False |
| 5 | base | hurst100 | transversal | T2 | 24999 | 0.0561 (5.43) | 6 | 0.0465 | -0.434 | 0 | True |
| 5 | base | hurst100 | transversal | T3 | 25847 | 0.0158 (1.58) | 4 | 0.0083 | -0.467 | 0 | False |
| 5 | base | hurst100 | mercado | T1 | 20681 | 0.0156 (1.29) | 3 | 0.0339 | -0.680 | 0 | False |
| 5 | base | hurst100 | mercado | T2 | 28853 | 0.0287 (2.68) | 6 | 0.0261 | -0.660 | 0 | False |
| 5 | base | hurst100 | mercado | T3 | 25724 | 0.0467 (4.34) | 5 | 0.0012 | -0.476 | 0 | False |
| 5 | base | pe100 | transversal | T1 | 24329 | 0.0212 (1.95) | 5 | 0.0190 | -0.439 | 0 | False |
| 5 | base | pe100 | transversal | T2 | 24995 | 0.0261 (2.44) | 5 | 0.0051 | -0.626 | 0 | False |
| 5 | base | pe100 | transversal | T3 | 25848 | 0.0436 (4.51) | 6 | 0.0431 | -0.517 | 0 | False |
| 5 | base | pe100 | mercado | T1 | 8187 | 0.0191 (0.97) | 3 | -0.0385 | -0.670 | 1 | False |
| 5 | base | pe100 | mercado | T2 | 22533 | 0.0442 (4.04) | 5 | 0.0144 | -0.425 | 0 | False |
| 5 | base | pe100 | mercado | T3 | 44538 | 0.0271 (3.25) | 5 | 0.0368 | -0.497 | 0 | False |
| 21 | all | hurst100 | transversal | T1 | 24176 | 0.0211 (1.27) | 5 | 0.0347 | -0.656 | 1 | False |
| 21 | all | hurst100 | transversal | T2 | 24839 | 0.0477 (3.30) | 6 | 0.0385 | -0.086 | 2 | False |
| 21 | all | hurst100 | transversal | T3 | 25679 | 0.0298 (2.12) | 4 | -0.0214 | -0.380 | 2 | False |
| 21 | all | hurst100 | mercado | T1 | 20591 | -0.0153 (-0.96) | 1 | -0.0028 | -0.776 | 0 | False |
| 21 | all | hurst100 | mercado | T2 | 28755 | 0.0487 (3.00) | 5 | -0.0010 | 0.282 | 4 | False |
| 21 | all | hurst100 | mercado | T3 | 25426 | 0.0588 (3.50) | 5 | -0.0344 | -0.121 | 2 | False |
| 21 | all | pe100 | transversal | T1 | 24171 | 0.0251 (1.44) | 3 | -0.0626 | 0.028 | 3 | False |
| 21 | all | pe100 | transversal | T2 | 24835 | 0.0177 (1.11) | 3 | -0.0360 | -0.486 | 2 | False |
| 21 | all | pe100 | transversal | T3 | 25680 | 0.0539 (3.56) | 6 | 0.0641 | -0.177 | 1 | True |
| 21 | all | pe100 | mercado | T1 | 8181 | 0.0562 (2.02) | 3 | -0.0417 | 0.067 | 4 | False |
| 21 | all | pe100 | mercado | T2 | 22533 | 0.0472 (2.88) | 4 | -0.0133 | 0.044 | 3 | False |
| 21 | all | pe100 | mercado | T3 | 44058 | 0.0244 (1.99) | 5 | -0.0002 | -0.312 | 2 | False |
| 21 | base | hurst100 | transversal | T1 | 24176 | 0.0159 (1.16) | 4 | 0.0192 | -0.488 | 1 | False |
| 21 | base | hurst100 | transversal | T2 | 24839 | 0.0477 (3.61) | 6 | 0.0409 | -0.310 | 1 | False |
| 21 | base | hurst100 | transversal | T3 | 25679 | 0.0312 (2.19) | 4 | -0.0210 | -0.895 | 1 | False |
| 21 | base | hurst100 | mercado | T1 | 20591 | 0.0104 (0.70) | 3 | 0.0156 | -0.563 | 1 | False |
| 21 | base | hurst100 | mercado | T2 | 28755 | 0.0411 (3.04) | 4 | 0.0218 | -0.055 | 2 | False |
| 21 | base | hurst100 | mercado | T3 | 25426 | 0.0433 (3.15) | 5 | 0.0042 | -0.317 | 2 | False |
| 21 | base | pe100 | transversal | T1 | 24171 | 0.0284 (1.92) | 4 | -0.0153 | -0.325 | 1 | False |
| 21 | base | pe100 | transversal | T2 | 24835 | 0.0154 (1.10) | 4 | -0.0027 | -0.877 | 0 | False |
| 21 | base | pe100 | transversal | T3 | 25680 | 0.0503 (4.10) | 6 | 0.0598 | -0.326 | 1 | True |
| 21 | base | pe100 | mercado | T1 | 8181 | 0.0228 (1.15) | 4 | -0.0463 | -0.193 | 2 | False |
| 21 | base | pe100 | mercado | T2 | 22533 | 0.0360 (2.78) | 5 | 0.0062 | -0.081 | 3 | False |
| 21 | base | pe100 | mercado | T3 | 44058 | 0.0340 (3.17) | 5 | 0.0282 | -0.326 | 2 | False |
