# M1 · Tabla única (modelo top-5 vs baselines)
Test 2023-03-09 → 2026-06-18 · 172 rebalanceos de 5 días · 31 tickers · semilla 42 · AUC p_up 0.5133

| Estrategia (por periodo de 5d) | Media | Sharpe anual | Total |
|---|---|---|---|
| Top-5 del modelo, bruto | +0.308% | +0.76 | +58.23% |
| Top-5 neto, costo fijo (100% rotación) | -0.292% | -0.73 | -43.67% |
| Top-5 neto, rotación real (83%) | -0.191% | -0.47 | -32.98% |
| Equiponderado del universo | +0.128% | +0.43 | +19.88% |
| Top-5 aleatorio (media de 1000) | +0.129% | +0.33 | +20.19% |
| 5 peores del ret_5d (sin modelo) | +0.060% | +0.14 | +2.82% |
| NAFTRAC | +0.107% | +0.36 | +15.81% |

IC por fecha (Spearman, step=5): p_up +0.0420 (t=2.61, IC95 [+0.0156, +0.0736], n=164) · p_rel +0.0346 (t=2.21, n=164) · −ret_5d +0.0228 (t=1.44).
AUC medio por fecha: p_up 0.5133 · p_rel 0.5218.

## Conclusiones
1. **¿Es mercado?** El top-5 aleatorio rinde Sharpe +0.33 y el equiponderado +0.43, contra +0.76 del modelo (percentil 90 entre sorteos). Queda arriba del azar, pero sin llegar al percentil 95: evidencia débil.
2. **¿Hay señal?** IC medio +0.0420 con t=2.61: distinto de cero, pero chico. Alfa del top-5 sobre el equiponderado: +8.9% anual (t=1.24).
3. **Atribución:** el top-5 elige acciones más nerviosas (ATR 1.03× el universo; vol anual 20% vs 15%), beta 1.02 sobre EW. Un top-5 aleatorio con la misma vol da Sharpe +0.64 (modelo en percentil 64).
4. **Costos:** la rotación real es 83%, no 100%: neto -0.47 (rotación real) vs -0.73 (fijo). Costo de equilibrio 0.185% por lado; el base es 0.30%, más que el equilibrio.
5. **Supervivencia:** EW de la lista actual − NAFTRAC = +0.1 pp en el año promedio. Colchón ≈ 0 o negativo: con esta estimación gruesa, el sesgo no explica el bruto.

Detalle por año, sensibilidad de costos, tickers y fechas: `data/eval/m1_report.json`.
Ensayos: 2 modelos, 2 backtests, 1 configuración (sin búsqueda).
