# N3 · captura del ex-dividendo — veredicto: **no operable**

_con la fecha real el efecto pasa, pero con la regla causal de fecha no_

Compra Close[p−5], venta Close[p] (+dividendo); costo 0.30%/lado; retención 10% (supuesto sin verificar).
Umbral Q4 de yield (p75 2008-17): oráculo 1.569%, causal 1.456%.

## Sanidad (bruto, todos los eventos 2004-2026)

| serie | n | trim. | media/evento | media/trim. | t |
|---|---|---|---|---|---|
| B bruto (vs EW) | 1138 | 92 | +1.21% | +1.39% | 5.10 |
| A bruto (largo-solo) | 1138 | 92 | +1.37% | +1.60% | 5.10 |

## Criterio (A, cuartil alto, neto con retención, 2018-2026)

| fecha | media test | t test | n trim. | media 2008-17 | cumple |
|---|---|---|---|---|---|
| causal | -0.50% | 0.31 | 36 | -0.65% | no |
| oráculo | +0.70% | 2.30 | 34 | +0.85% | sí |

## Detalle por variante (2018-2026)

| variante | n | trim. | media/evento | media/trim. | t |
|---|---|---|---|---|---|
| oráculo todos A gross | 545 | 36 | +1.10% | +1.16% | 4.71 |
| oráculo todos A net_tax | 545 | 36 | +0.32% | +0.40% | 1.63 |
| oráculo todos A net_notax | 545 | 36 | +0.50% | +0.56% | 2.28 |
| oráculo todos B gross | 545 | 36 | +0.98% | +0.97% | 5.54 |
| oráculo todos B net_tax | 545 | 36 | +0.20% | +0.21% | 1.19 |
| oráculo todos B net_notax | 545 | 36 | +0.38% | +0.37% | 2.10 |
| oráculo Q4 A gross | 226 | 34 | +1.58% | +1.72% | 4.53 |
| oráculo Q4 A net_tax | 226 | 34 | +0.70% | +0.87% | 2.30 |
| oráculo Q4 A net_notax | 226 | 34 | +0.98% | +1.12% | 2.95 |
| oráculo Q4 B gross | 226 | 34 | +1.59% | +1.65% | 5.09 |
| oráculo Q4 B net_tax | 226 | 34 | +0.71% | +0.79% | 2.47 |
| oráculo Q4 B net_notax | 226 | 34 | +0.99% | +1.05% | 3.24 |
| causal todos A gross | 525 | 36 | +0.45% | +0.62% | 2.37 |
| causal todos A net_tax | 525 | 36 | -0.32% | -0.14% | -0.54 |
| causal todos A net_notax | 525 | 36 | -0.15% | +0.02% | 0.06 |
| causal todos B gross | 525 | 36 | +0.32% | +0.43% | 2.15 |
| causal todos B net_tax | 525 | 36 | -0.45% | -0.32% | -1.59 |
| causal todos B net_notax | 525 | 36 | -0.28% | -0.17% | -0.83 |
| causal Q4 A gross | 241 | 36 | +0.37% | +0.97% | 2.50 |
| causal Q4 A net_tax | 241 | 36 | -0.50% | +0.12% | 0.31 |
| causal Q4 A net_notax | 241 | 36 | -0.23% | +0.37% | 0.95 |
| causal Q4 B gross | 241 | 36 | +0.28% | +0.75% | 2.28 |
| causal Q4 B net_tax | 241 | 36 | -0.59% | -0.10% | -0.31 |
| causal Q4 B net_notax | 241 | 36 | -0.32% | +0.15% | 0.45 |

## Acierto de la fecha causal

- Operaciones previstas con evento real a ±2 días hábiles: 50.8%
- Eventos reales cubiertos por una previsión a ±2: 47.3%
- Con acierto: n=571, bruto A +1.66%; con fallo: n=553, bruto A +0.19%
- Capturaron algún dividendo real: 33.1%

## Eventos por año (oráculo)

Todos: {2000: 10, 2001: 14, 2002: 15, 2003: 25, 2004: 28, 2005: 29, 2006: 34, 2007: 34, 2008: 32, 2009: 30, 2010: 39, 2011: 39, 2012: 44, 2013: 65, 2014: 40, 2015: 58, 2016: 60, 2017: 61, 2018: 61, 2019: 62, 2020: 51, 2021: 54, 2022: 60, 2023: 61, 2024: 59, 2025: 78, 2026: 59}

Q4: {2000: 1, 2001: 2, 2002: 8, 2003: 15, 2004: 12, 2005: 10, 2006: 13, 2007: 13, 2008: 11, 2009: 9, 2010: 8, 2011: 7, 2012: 11, 2013: 15, 2014: 6, 2015: 13, 2016: 18, 2017: 19, 2018: 26, 2019: 23, 2020: 18, 2021: 20, 2022: 30, 2023: 25, 2024: 29, 2025: 33, 2026: 22}

## Extraordinarios descartados (yield > 15%)

- AC 2011-07-25: yield 16.5%, bruto -8.21%
- GCARSOA1 2006-12-19: yield 16.3%, bruto +70.88%
- GCARSOA1 2008-03-04: yield 29.1%, bruto +16.39%
- HERDEZ 2026-04-13: yield 17.4%, bruto +3.49%
- HOTEL 2023-05-29: yield 23.3%, bruto +8.86%

Ensayos: 8 escritos ahora; N familia n3_exdiv = 8.
