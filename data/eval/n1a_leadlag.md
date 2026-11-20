# N1-A lead-lag US -> Mexico

## Datos y alineacion
- Descargas yfinance: {'EWW': True, 'HG=F': True, '^TNX': True, '^VIX': True}
- Factores usados: ['spy', 'ewz', 'eem', 'xle', 'hg', 'tnx', 'mxn', 'vix', 'eww']; faltantes: []
- ADR disponibles: ['AMXL', 'FEMSAUBD'] (solo AMXL->AMX, FEMSAUBD->FMX; el resto adr_gap=0)
- Auditoria beta sin futuro (corromper retornos MX recientes no cambia betas previos): {'WALMEX': True, 'GMEXICOB': True, 'GCARSOA1': True}
- Retorno US de la fila t = acumulado US en (t_prev_BMV, t]; BMV cierra 15:00 CT = 16:00 ET, asi que el cierre NY de t se conoce antes del Open t+1 (entrada).
- Betas rodantes 252d con datos hasta t-1; residual y gap ADR usan solo cierres del dia t.
- Las features `us_*` son iguales para todas las emisoras de una fecha: no pueden mover el IC transversal por si solas (solo AUC pooled).

## (a) Correlacion cruzada por emisora (FDR Benjamini-Hochberg q=0.05)

corr(factor US del dia t, objetivo); t Newey-West (lag 5 en ret_fwd, 1 en los demas). In-sample, diagnostico.

| objetivo/periodo | tests | rechazos BH | p<.05 sin corregir | esperados por azar | mediana corr |
|---|---|---|---|---|---|
| cc0|2024+ | 306 | 170 | 179 | 15.3 | 0.0864 |
| cc0|full | 315 | 299 | 300 | 15.8 | 0.1751 |
| gap|2024+ | 306 | 0 | 16 | 15.3 | -0.0043 |
| gap|full | 315 | 8 | 52 | 15.8 | 0.0055 |
| oc1|2024+ | 306 | 1 | 13 | 15.3 | 0.0028 |
| oc1|full | 315 | 34 | 69 | 15.8 | 0.0075 |
| ret_fwd|2024+ | 306 | 0 | 15 | 15.3 | -0.0050 |
| ret_fwd|full | 315 | 0 | 29 | 15.8 | -0.0017 |

Top 8 por |t| en ret_fwd (completo):

| emisora | factor | corr | t | p | BH |
|---|---|---|---|---|---|
| VESTA | tnx | -0.0553 | -2.99 | 0.0028 | False |
| BIMBOA | spy | -0.0631 | -2.90 | 0.0038 | False |
| BIMBOA | ewz | -0.0636 | -2.83 | 0.0046 | False |
| CHDRAUIB | ewz | -0.0532 | -2.79 | 0.0053 | False |
| WALMEX | tnx | -0.0505 | -2.77 | 0.0056 | False |
| WALMEX | eww | -0.0651 | -2.71 | 0.0066 | False |
| KIMBERA | tnx | -0.0439 | -2.60 | 0.0093 | False |
| BIMBOA | eem | -0.0531 | -2.44 | 0.0147 | False |

## (b) Modelo base + features nuevas (mismos 6 folds, WoE+Logistica)

| variante | AUC | IC | t NW5 | alfa largo neto % | folds alfa+ | IC 2024+ | t 2024+ | dIC vs control | t dIC |
|---|---|---|---|---|---|---|---|---|---|
| base_top15 | 0.5165 | 0.0167 | 2.60 | -0.526 | 0/6 | 0.0034 | 0.29 | nan | nan |
| base_top25 | 0.5136 | 0.0150 | 2.47 | -0.572 | 0/6 | 0.0135 | 1.24 | nan | nan |
| mkt_top15 | 0.5133 | 0.0179 | 2.75 | -0.559 | 0/6 | 0.0042 | 0.36 | 0.0013 | 0.93 |
| beta_top15 | 0.5159 | 0.0175 | 2.55 | -0.515 | 0/6 | 0.0052 | 0.44 | 0.0008 | 0.22 |
| adr_top15 | 0.5165 | 0.0167 | 2.60 | -0.526 | 0/6 | 0.0034 | 0.29 | 0.0000 | 0.00 |
| all_top15 | 0.5121 | 0.0191 | 2.87 | -0.509 | 0/6 | 0.0069 | 0.60 | 0.0025 | 0.79 |
| all_top25 | 0.5161 | 0.0180 | 2.71 | -0.514 | 0/6 | 0.0068 | 0.59 | 0.0030 | 0.62 |
| solo_nuevas_top15 | 0.5074 | 0.0125 | 2.15 | -0.520 | 0/6 | 0.0134 | 1.33 | -0.0042 | -0.57 |

Control: top15 -> base_top15; top25 -> base_top25.

### Por tercil de liquidez (ADV 60d): IC / t / alfa neto %

| variante | T1 iliquido | T2 | T3 liquido |
|---|---|---|---|
| base_top15 | 0.0035 / 0.34 / -0.57 | 0.0011 / 0.11 / -0.65 | 0.0350 / 3.27 / -0.48 |
| base_top25 | 0.0028 / 0.27 / -0.67 | 0.0008 / 0.08 / -0.60 | 0.0328 / 3.27 / -0.50 |
| mkt_top15 | 0.0008 / 0.08 / -0.52 | 0.0005 / 0.04 / -0.64 | 0.0390 / 3.55 / -0.44 |
| beta_top15 | 0.0053 / 0.52 / -0.55 | -0.0039 / -0.38 / -0.57 | 0.0383 / 3.41 / -0.47 |
| adr_top15 | 0.0035 / 0.34 / -0.57 | 0.0011 / 0.11 / -0.65 | 0.0350 / 3.27 / -0.48 |
| all_top15 | 0.0075 / 0.73 / -0.53 | -0.0026 / -0.26 / -0.59 | 0.0418 / 3.76 / -0.44 |
| all_top25 | 0.0033 / 0.33 / -0.57 | -0.0021 / -0.21 / -0.63 | 0.0389 / 3.53 / -0.50 |
| solo_nuevas_top15 | 0.0081 / 0.92 / -0.54 | -0.0025 / -0.30 / -0.66 | 0.0247 / 2.56 / -0.51 |

### IC por anio

| anio | base_top15 | beta_top15 | all_top15 |
|---|---|---|---|
| 2017 | -0.0302 | -0.0381 | -0.0328 |
| 2018 | 0.0206 | 0.0280 | 0.0303 |
| 2019 | 0.0096 | 0.0219 | 0.0233 |
| 2020 | 0.0337 | 0.0413 | 0.0387 |
| 2021 | 0.0217 | 0.0201 | 0.0217 |
| 2022 | 0.0322 | 0.0065 | 0.0183 |
| 2023 | 0.0406 | 0.0464 | 0.0400 |
| 2024 | 0.0090 | 0.0034 | 0.0070 |
| 2025 | 0.0051 | 0.0156 | 0.0173 |
| 2026 | -0.0054 | -0.0052 | -0.0055 |
| 2024+ | 0.0034 | 0.0052 | 0.0069 |

## Veredicto (criterio: t NW5>2, alfa neto>0 en >=4/6 folds, dIC>0.03)

- base_top15: t>2=True, alfa>0=False, folds alfa+ mayoria=False, dIC=0.0000 (>0.03: False), pasa=False
- base_top25: t>2=True, alfa>0=False, folds alfa+ mayoria=False, dIC=0.0000 (>0.03: False), pasa=False
- mkt_top15: t>2=True, alfa>0=False, folds alfa+ mayoria=False, dIC=0.0013 (>0.03: False), pasa=False
- beta_top15: t>2=True, alfa>0=False, folds alfa+ mayoria=False, dIC=0.0008 (>0.03: False), pasa=False
- adr_top15: t>2=True, alfa>0=False, folds alfa+ mayoria=False, dIC=0.0000 (>0.03: False), pasa=False
- all_top15: t>2=True, alfa>0=False, folds alfa+ mayoria=False, dIC=0.0025 (>0.03: False), pasa=False
- all_top25: t>2=True, alfa>0=False, folds alfa+ mayoria=False, dIC=0.0030 (>0.03: False), pasa=False
- solo_nuevas_top15: t>2=True, alfa>0=False, folds alfa+ mayoria=False, dIC=-0.0042 (>0.03: False), pasa=False

DSR no calculado en esta tanda.
