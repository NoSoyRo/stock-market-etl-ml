# M4 · Estudio de reversión de 5 días (2026-11-05)

Universo: 32 tickers .MX, 2008-01-02 → 2026-06-26; máscara M3a: False. Score = −señal; IC Spearman semanal (paso 5). Ensayos: 288.

## Todo el universo

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_1d | +0.0476 (t=6.5) | +0.0062 (t=0.9) | +0.0036 (t=0.5) | +0.0212 (t=3.0) |
| ret_2d | +0.0544 (t=7.3) | +0.0279 (t=3.8) | +0.0331 (t=4.5) | +0.0274 (t=3.8) |
| ret_5d | +0.0560 (t=7.3) | +0.0442 (t=5.9) | +0.0444 (t=6.0) | +0.0456 (t=6.2) |
| ret_21d | +0.0292 (t=3.9) | +0.0173 (t=2.2) | +0.0233 (t=3.1) | +0.0245 (t=3.2) |
| overnight_1d | -0.0026 (t=-0.4) | -0.0103 (t=-1.5) | -0.0134 (t=-2.0) | -0.0125 (t=-1.9) |
| overnight_5d | +0.0039 (t=0.5) | +0.0141 (t=2.1) | +0.0045 (t=0.7) | +0.0058 (t=0.8) |
| intraday_1d | +0.0407 (t=5.6) | +0.0152 (t=2.1) | +0.0188 (t=2.6) | +0.0298 (t=4.2) |
| intraday_5d | +0.0516 (t=7.0) | +0.0232 (t=3.1) | +0.0277 (t=3.7) | +0.0327 (t=4.5) |
| resid_5d | +0.0528 (t=6.9) | +0.0368 (t=5.1) | +0.0367 (t=5.0) | +0.0377 (t=5.2) |

## Subconjunto: liq_bajo (solo ret_5d, ret_1d, resid_5d)

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_5d | +0.0421 (t=3.3) | +0.0318 (t=2.5) | +0.0335 (t=2.7) | +0.0385 (t=3.0) |
| ret_1d | +0.0309 (t=2.5) | +0.0111 (t=0.9) | +0.0084 (t=0.7) | +0.0166 (t=1.3) |
| resid_5d | +0.0467 (t=3.7) | +0.0209 (t=1.7) | +0.0211 (t=1.7) | +0.0219 (t=1.7) |

## Subconjunto: liq_medio (solo ret_5d, ret_1d, resid_5d)

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_5d | +0.0418 (t=3.3) | +0.0156 (t=1.2) | +0.0201 (t=1.5) | +0.0166 (t=1.3) |
| ret_1d | +0.0503 (t=3.9) | +0.0043 (t=0.3) | +0.0063 (t=0.5) | +0.0208 (t=1.7) |
| resid_5d | +0.0462 (t=3.6) | +0.0284 (t=2.2) | +0.0257 (t=2.0) | +0.0309 (t=2.4) |

## Subconjunto: liq_alto (solo ret_5d, ret_1d, resid_5d)

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_5d | +0.0687 (t=5.0) | +0.0572 (t=4.2) | +0.0608 (t=4.5) | +0.0608 (t=4.5) |
| ret_1d | +0.0438 (t=3.3) | +0.0060 (t=0.5) | +0.0153 (t=1.2) | +0.0347 (t=2.6) |
| resid_5d | +0.0655 (t=4.9) | +0.0594 (t=4.4) | +0.0584 (t=4.2) | +0.0649 (t=4.7) |

## Subconjunto: volanor_bajo (solo ret_5d, ret_1d, resid_5d)

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_5d | +0.0374 (t=3.0) | +0.0296 (t=2.3) | +0.0322 (t=2.5) | +0.0338 (t=2.6) |
| ret_1d | +0.0394 (t=3.2) | +0.0203 (t=1.6) | +0.0167 (t=1.3) | +0.0331 (t=2.5) |
| resid_5d | +0.0357 (t=2.9) | +0.0406 (t=3.1) | +0.0451 (t=3.5) | +0.0328 (t=2.5) |

## Subconjunto: volanor_medio (solo ret_5d, ret_1d, resid_5d)

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_5d | +0.0555 (t=4.3) | +0.0333 (t=2.6) | +0.0216 (t=1.7) | +0.0367 (t=2.9) |
| ret_1d | +0.0661 (t=5.2) | +0.0116 (t=0.9) | +0.0274 (t=2.2) | +0.0114 (t=0.9) |
| resid_5d | +0.0492 (t=3.8) | +0.0296 (t=2.2) | +0.0198 (t=1.5) | +0.0316 (t=2.5) |

## Subconjunto: volanor_alto (solo ret_5d, ret_1d, resid_5d)

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_5d | +0.0698 (t=5.3) | +0.0523 (t=3.9) | +0.0580 (t=4.3) | +0.0524 (t=4.1) |
| ret_1d | +0.0473 (t=3.5) | +0.0404 (t=3.1) | +0.0387 (t=2.9) | +0.0554 (t=4.2) |
| resid_5d | +0.0666 (t=5.0) | +0.0374 (t=2.9) | +0.0473 (t=3.5) | +0.0377 (t=2.9) |

## Subconjunto: sin_exdiv (solo ret_5d, ret_1d, resid_5d)

| señal | close_close | open_close | open_open | skip_close |
|---|---|---|---|---|
| ret_5d | +0.0556 (t=7.4) | +0.0389 (t=5.1) | +0.0397 (t=5.2) | +0.0418 (t=5.5) |
| ret_1d | +0.0386 (t=5.2) | +0.0232 (t=3.2) | +0.0276 (t=3.7) | +0.0307 (t=4.2) |
| resid_5d | +0.0564 (t=7.4) | +0.0409 (t=5.3) | +0.0399 (t=5.2) | +0.0404 (t=5.4) |

## Bloques (ret_5d, todo)

| entrada | B1 | B2 | B3 | B4 | B5 |
|---|---|---|---|---|---|
| close_close | +0.052 | +0.081 | +0.048 | +0.055 | +0.044 |
| open_close | +0.066 | +0.079 | +0.019 | +0.032 | +0.025 |
| open_open | +0.063 | +0.070 | +0.028 | +0.030 | +0.030 |
| skip_close | +0.066 | +0.079 | +0.022 | +0.030 | +0.030 |

Bloques: B1 2008-01-09→2011-11-09; B2 2011-11-16→2015-07-13; B3 2015-07-20→2019-03-01; B4 2019-03-08→2022-10-20; B5 2022-10-27→2026-06-18

## Puerta

- Criterio 1 (bloques): NO PASA — 3/5 con IC≥0.03; medias [0.0657, 0.0787, 0.0193, 0.0322, 0.0251]
- Criterio 2 (tercio más líquido): PASA — IC +0.0572 (t=4.2)
- **Puerta: NO PASA**

## Portafolio top-5 de las peores de ret_5d (Open t+1 → Close t+6)

- periodos: 851
- bruto_medio_pct: 0.5505
- neto_medio_pct_a_0.30: 0.0578
- ew_medio_pct: 0.2847
- aleatorio_medio_pct: 0.2817
- percentil_bruto_vs_aleatorio: 100.0000
- exceso_bruto_vs_ew_pct: 0.2634
- exceso_t: 3.7059
- exceso_neto_vs_ew_pct: -0.2292
- rotacion_media: 0.8213
- costo_equilibrio_por_lado_pct: 0.3352
- bruto_cubre_costo_0.30: True
- max_dif_vs_topk_returns: 0.0000

| costo/lado | neto medio | Sharpe | retorno total |
|---|---|---|---|
| 0.0010 | 0.0039 | 0.7742 | 14.5487 |
| 0.0015 | 0.0030 | 0.6095 | 6.7408 |
| 0.0030 | 0.0006 | 0.1157 | -0.0483 |
| 0.0050 | -0.0027 | -0.5416 | -0.9423 |
