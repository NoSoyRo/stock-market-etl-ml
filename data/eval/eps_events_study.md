# N5 · Eventos de EPS (2026-11-12)

Criterios en `avances/2026-11-12_N5_preregistro.md`. Ensayos: 9.

## (a) Reversión con / sin anuncio (35 tickers MX con EPS)

| periodo | simple | sin evento | con evento | dif (sin−con) [IC95%] |
|---|---|---|---|---|
| 2008-2017 | +0.0455 (t=4.7, n=492) | +0.0516 (t=4.9, n=491) | -0.0435 (t=-1.0, n=56) | +0.1163 [-0.0144, +0.2361] (n=54) |
| 2018-2026 | +0.0215 (t=2.2, n=443) | +0.0242 (t=2.4, n=443) | -0.0576 (t=-1.4, n=62) | +0.0725 [-0.0294, +0.1868] (n=62) |
| 2023-2026 | +0.0086 (t=0.6, n=191) | +0.0142 (t=1.0, n=191) | -0.0829 (t=-1.2, n=27) | +0.1199 [-0.0709, +0.3162] (n=27) |
| 2008-2026 | +0.0341 (t=5.0, n=935) | +0.0386 (t=5.3, n=934) | -0.0509 (t=-1.7, n=118) | +0.0929 [+0.0049, +0.1751] (n=116) |

Tamaño medio del grupo: {'sin_evento': 27.306468716861083, 'con_evento': 6.560137457044673}

Top-5 (2018-2026, bruto, % por semana): {"sin_evento": {"semanas": 443, "top5_bruto_pct": 0.35058264826133695, "ew_pct": 0.1521652815136873, "aleatorio_pct": 0.14957117180377041, "exceso_vs_ew_pct": 0.1984173667476497}, "con_evento": {"semanas": 62, "top5_bruto_pct": -0.01677680548200609, "ew_pct": -0.03780489884696972, "aleatorio_pct": -0.05045008175087177, "exceso_vs_ew_pct": 0.021028093364963557}, "simple": {"semanas": 443, "top5_bruto_pct": 0.32959421146419143, "ew_pct": 0.16216790543486795, "aleatorio_pct": 0.15486182849881763, "exceso_vs_ew_pct": 0.16742630602932349}}

**Veredicto (a): NO PASA** — {'sin>+0.03_t>2': False, 'con<=0': True, 'dif_ci_lo>0': False, 'descarte_sin<0_2023_26': False}

## (b) Calendario esperado

Acierto de la fecha esperada (n=2056): |err|≤1 48.9%, ≤2 58.6%, ≤5 78.6% (de días hábiles).

### (b1) Volatilidad vs HAR

| h | n_obs | skill HAR | Δskill agrupado [IC95%] | DM p | pasa |
|---|---|---|---|---|---|
| 5 | 128,541 | +0.3023 | +0.0040 [+0.0014, +0.0066] | 0.003 | False |
| 21 | 128,088 | +0.2157 | +0.0060 [+0.0005, +0.0119] | 0.040 | False |

**Veredicto (b1): NO PASA**

### (b2) Filtro de riesgo (EW semanal, neto 0.30%/lado)

| periodo | sin filtro (media%, sd%) | filtrado | oráculo | ΔSharpe filtrado [IC95%] | ΔMedia% |
|---|---|---|---|---|---|
| 2008-2017 | +0.274, 2.54 | +0.206, 2.65 | +0.151, 2.57 | -0.25 [-0.39, -0.11] | -0.079 |
| 2018-2026 | +0.158, 2.37 | +0.014, 2.35 | +0.053, 2.39 | -0.27 [-0.45, -0.12] | -0.090 |

Fracción de nombre-semanas marcadas: 13.9%.
**Veredicto (b2): NO PASA** — {'dSharpe_ci_lo>0_2018_26': False, 'dMedia>=-0.05pct': False, 'mismo_signo_2008_17': False}

## (c) PEAD diagnóstico (entrada Close d+1, 5 días, quintil de sorpresa por trimestre)

Eventos: 1874 (Open limpio: 1536).

| muestra | periodo | eventos | temporadas | ret% Q1..Q5 | spread Q5−Q1 % (t) | Q5 − 0.60% |
|---|---|---|---|---|---|---|
| todos | 2008-2017 | 787 | 39 | -0.18 / +0.17 / +0.43 / +0.90 / +0.67 | +0.85 (1.7) | +0.07 |
| todos | 2018-2026 | 1078 | 35 | -0.03 / -0.19 / +0.14 / +0.27 / +0.72 | +0.82 (1.9) | +0.12 |
| open_limpio | 2008-2017 | 502 | 24 | -0.07 / -0.21 / +0.06 / +0.93 / +0.32 | +0.40 (0.7) | -0.28 |
| open_limpio | 2018-2026 | 995 | 35 | -0.01 / -0.18 / -0.00 / +0.45 / +0.73 | +0.70 (1.4) | +0.13 |
