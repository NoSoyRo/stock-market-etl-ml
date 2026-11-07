# M4 · Estudio de reversión de 5 días (2026-11-05)

Universo: 32 tickers .MX, 2008-01-02 → 2026-06-26; máscara M3a: True. Score = −señal; IC Spearman semanal (paso 5). Ensayos: 288.

## Todo el universo

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_1d | +0.0433 (t=5.9) | +0.0152 (t=2.1) | +0.0193 (t=2.7) | +0.0211 (t=3.0) |
| ret_2d | +0.0542 (t=7.2) | +0.0333 (t=4.6) | +0.0408 (t=5.7) | +0.0371 (t=5.3) |
| ret_5d | +0.0615 (t=8.3) | +0.0424 (t=5.7) | +0.0412 (t=5.5) | +0.0427 (t=5.7) |
| ret_21d | +0.0354 (t=4.6) | +0.0183 (t=2.4) | +0.0238 (t=3.1) | +0.0247 (t=3.2) |
| overnight_1d | -0.0020 (t=-0.3) | +0.0099 (t=1.4) | +0.0028 (t=0.4) | -0.0032 (t=-0.5) |
| overnight_5d | +0.0111 (t=1.6) | +0.0165 (t=2.3) | +0.0085 (t=1.2) | +0.0085 (t=1.2) |
| intraday_1d | +0.0415 (t=5.7) | +0.0157 (t=2.2) | +0.0188 (t=2.6) | +0.0299 (t=4.2) |
| intraday_5d | +0.0482 (t=6.5) | +0.0297 (t=3.9) | +0.0317 (t=4.3) | +0.0360 (t=4.9) |
| resid_5d | +0.0604 (t=8.1) | +0.0392 (t=5.3) | +0.0442 (t=6.0) | +0.0385 (t=5.3) |

## Subconjunto: liq_bajo (solo ret_5d, ret_1d, resid_5d)

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_5d | +0.0329 (t=2.5) | +0.0278 (t=2.1) | +0.0258 (t=2.0) | +0.0378 (t=2.8) |
| ret_1d | +0.0255 (t=2.0) | +0.0170 (t=1.3) | +0.0101 (t=0.8) | +0.0294 (t=2.3) |
| resid_5d | +0.0519 (t=4.1) | +0.0311 (t=2.5) | +0.0363 (t=2.9) | +0.0393 (t=3.0) |

## Subconjunto: liq_medio (solo ret_5d, ret_1d, resid_5d)

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_5d | +0.0417 (t=3.2) | +0.0147 (t=1.2) | +0.0182 (t=1.4) | +0.0199 (t=1.5) |
| ret_1d | +0.0569 (t=4.4) | +0.0185 (t=1.4) | +0.0164 (t=1.3) | +0.0343 (t=2.6) |
| resid_5d | +0.0431 (t=3.3) | +0.0089 (t=0.7) | +0.0121 (t=0.9) | +0.0174 (t=1.3) |

## Subconjunto: liq_alto (solo ret_5d, ret_1d, resid_5d)

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_5d | +0.0703 (t=5.5) | +0.0737 (t=5.6) | +0.0757 (t=5.6) | +0.0619 (t=4.6) |
| ret_1d | +0.0576 (t=4.4) | +0.0381 (t=2.8) | +0.0360 (t=2.7) | +0.0431 (t=3.3) |
| resid_5d | +0.0822 (t=6.0) | +0.0498 (t=3.7) | +0.0581 (t=4.5) | +0.0509 (t=3.8) |

## Subconjunto: volanor_bajo (solo ret_5d, ret_1d, resid_5d)

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_5d | +0.0370 (t=2.8) | +0.0265 (t=2.1) | +0.0323 (t=2.6) | +0.0200 (t=1.5) |
| ret_1d | +0.0479 (t=3.8) | +0.0232 (t=1.9) | +0.0340 (t=2.7) | +0.0212 (t=1.7) |
| resid_5d | +0.0356 (t=2.7) | +0.0206 (t=1.6) | +0.0240 (t=1.9) | +0.0198 (t=1.5) |

## Subconjunto: volanor_medio (solo ret_5d, ret_1d, resid_5d)

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_5d | +0.0522 (t=3.9) | +0.0335 (t=2.6) | +0.0280 (t=2.2) | +0.0339 (t=2.6) |
| ret_1d | +0.0434 (t=3.4) | +0.0269 (t=2.1) | +0.0375 (t=2.9) | +0.0246 (t=1.9) |
| resid_5d | +0.0576 (t=4.6) | +0.0316 (t=2.4) | +0.0344 (t=2.6) | +0.0219 (t=1.7) |

## Subconjunto: volanor_alto (solo ret_5d, ret_1d, resid_5d)

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_5d | +0.0753 (t=5.5) | +0.0515 (t=3.9) | +0.0566 (t=4.2) | +0.0475 (t=3.6) |
| ret_1d | +0.0445 (t=3.4) | +0.0094 (t=0.7) | +0.0174 (t=1.3) | +0.0277 (t=2.1) |
| resid_5d | +0.0949 (t=7.2) | +0.0396 (t=3.0) | +0.0492 (t=3.6) | +0.0356 (t=2.7) |

## Subconjunto: sin_exdiv (solo ret_5d, ret_1d, resid_5d)

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_5d | +0.0538 (t=7.0) | +0.0410 (t=5.3) | +0.0462 (t=6.0) | +0.0434 (t=5.5) |
| ret_1d | +0.0398 (t=5.4) | +0.0236 (t=3.2) | +0.0282 (t=3.8) | +0.0308 (t=4.2) |
| resid_5d | +0.0571 (t=7.3) | +0.0385 (t=5.1) | +0.0375 (t=4.9) | +0.0384 (t=5.0) |

## Bloques (ret_5d, todo)

| entrada | B1 | B2 | B3 | B4 | B5 |
|---|---|---|---|---|---|
| close_close | +0.058 | +0.081 | +0.047 | +0.057 | +0.064 |
| open_close | +0.064 | +0.065 | +0.021 | +0.032 | +0.030 |
| open_open | +0.061 | +0.083 | +0.017 | +0.021 | +0.023 |
| skip_close | +0.057 | +0.075 | +0.015 | +0.037 | +0.028 |

Bloques: B1 2008-01-09→2011-11-10; B2 2011-11-17→2015-07-14; B3 2015-07-21→2019-03-04; B4 2019-03-11→2022-10-21; B5 2022-10-28→2026-06-19

## Puerta

- Criterio 1 (bloques): PASA — 4/5 con IC≥0.03; medias [0.0637, 0.0651, 0.0207, 0.0325, 0.0302]
- Criterio 2 (tercio más líquido): PASA — IC +0.0737 (t=5.6)
- **Puerta: PASA**

## Portafolio top-5 de las peores de ret_5d (Open t+1 → Close t+6)

- periodos: 854
- bruto_medio_pct: 0.5507
- neto_medio_pct_a_0.30: 0.0588
- ew_medio_pct: 0.2747
- aleatorio_medio_pct: 0.2745
- percentil_bruto_vs_aleatorio: 100.0000
- exceso_bruto_vs_ew_pct: 0.2736
- exceso_t: 3.8595
- exceso_neto_vs_ew_pct: -0.2181
- rotacion_media: 0.8198
- costo_equilibrio_por_lado_pct: 0.3359
- bruto_cubre_costo_0.30: True
- max_dif_vs_topk_returns: 0.0000

| costo/lado | neto medio | Sharpe | retorno total |
|---|---|---|---|
| 0.0010 | 0.0039 | 0.7753 | 14.7649 |
| 0.0015 | 0.0030 | 0.6109 | 6.8396 |
| 0.0030 | 0.0006 | 0.1178 | -0.0394 |
| 0.0050 | -0.0027 | -0.5388 | -0.9420 |
