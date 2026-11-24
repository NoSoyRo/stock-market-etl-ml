# Pre-registro · prueba fuera de muestra (OOS v2)

Escrito el 2026-11-09 **antes de correr nada** sobre la ventana nueva. Rama `eval/oos-v2-recon`. Si algo de aquí cambia después de ver resultados, se anota como desviación en el reporte, no se edita este texto.

## Qué se prueba
Reglas ya construidas en la v1, **sin reajustar hiperparámetros, reglas ni features**.

| id | Señal | Entrenamiento |
|---|---|---|
| E-frozen | `p_up` del ensemble (`EnsembleModel`, 40 features) | Igual que la v1: train < 2023-03-09 |
| E-refit | ídem | Misma receta, train hasta 2026-06-09 (embargo ≈ horizonte + Open t+1) |
| R-frozen | Ridge sobre ranks (`RidgeArm`, alpha 100, 24 features `rn_*`, objetivo `pct_fwd`) | train < 2023-03-09 |
| R-refit | ídem | train hasta 2026-06-09 |
| REV | −`ret_5d` (reversión de M4, sin modelo) | no aplica |

- **Regla de cartera (E y R):** celda M5 10d / K=10 / salida 2K (rank 20), pesos iguales, rebalanceo cada 2 periodos de 5 días. Desempate aleatorio con semilla 42 (jitter 1e-9), como en M5.
- **Regla de cartera (REV):** las 5 peores por `ret_5d`, semanal, sin histéresis (la cartera de M4).
- **Panel:** `build_panel(..., use_mask=True)` para todo. `model_feature_cols` (40) para el ensemble; `AUX_FEATURE_COLS` fuera.
- **Ventana de evaluación:** fechas de señal 2026-06-19 → última con retorno futuro completo (≤ 2026-10-30). ~19 semanas.
- **Medición:** entrada al Open del día siguiente, costo 0.30%/lado con rotación real, IC por fecha, Sharpe con ppy = 50.4, comparación contra equiponderado (EW) a la misma frecuencia y contra top-K aleatorio.
- **Ensayos:** 5 filas nuevas en `data/eval/trial_log.jsonl`, familia `oos_v2`; se reporta N total.

## Criterio de éxito (fijado antes de ver resultados)
Para E-frozen, E-refit, R-frozen y R-refit:
1. **S1:** IC por fecha medio > 0.
2. **S2:** retorno neto @0.30% por periodo > el del EW.

- **Se sostiene** = S1 y S2.
- **Inconcluso** = solo una de las dos.
- **Falla grosera** = IC ≤ 0, **o** exceso neto vs EW < −½ de la ventaja neta que tuvo la v1.
  - Referencia v1 (celda M5 `p_up`): exceso neto +0.145%/semana. El umbral de falla es por tanto −0.0725%/semana.

Para REV (la v1 ya mostró que neta pierde contra el EW, −0.23%/sem; su evidencia era bruta):
1. **S1:** IC por fecha medio > 0 (Open t+1 → Close t+6).
2. **S2':** retorno **bruto** de la cartera > bruto del EW.
- Misma clasificación. Falla grosera = IC ≤ 0.

**Alarma de fuga:** un IC medio > 0.10 en la ventana se trata como sospecha de error, no como éxito.

## Qué se dirá al reportar
- Con ~19 semanas (≈ 9 periodos independientes a 10d) la prueba solo detecta fracasos groseros. "Se sostiene" **no** prueba ventaja.
- 5 ensayos más sobre los ~25 previos; la ventana nueva no sirve como justificación para reajustar nada.
- No se compara contra la v1 con otro criterio que éste.
