# Último intento ML — registro de experimentos

Rama: `exp/ml-ultimo-intento` · Inicio: 2026-11-18 · Origen: panel de 8 agentes (ideas frescas) → ranking por niveles.

## Criterio de parada global (fijado ANTES de correr)

Éxito = IC medio por fecha con t(Newey-West) > 2 **y** alfa neto de costos (0.30 %/lado) > 0 OOS en la mayoría de folds, **o** DSR > 0.95.
Para contar como hallazgo, el IC incremental debe quedar > ~0.03 (MDE ≈ 0.026 con este panel).
Si ningún nivel lo cumple, se cierra con resultado negativo documentado.

## Hechos del panel (medidos en N0-A)

- Panel real del CV: **35 emisoras**, 135,534 filas, 2008-03 a 2026-11 (no 94, no 39).
- Protocolo: 6 folds por fechas (`purged_walk_forward_date_splits`), entrada Open t+1. La alerta de fuga por split de filas aplicaba solo a comparadores LSTM/TCN viejos, no al CV principal.
- Hardware: ~1 GB RAM libre, GPU MX150 2 GB inservible.
- Trials en `data/eval/trial_log.jsonl`: 55 al inicio → 179 tras el Nivel 1.

## Línea base

WoE + logística, h=5: AUC 0.5165 · IC por fecha 0.0167 (t NW 2.60) · alfa largo neto −0.53 %.

## Nivel 0 — diagnóstico (cerrado, no pasa)

| Exp | Qué | Resultado |
|---|---|---|
| N0-A | IC por fecha, AUC demeaned, limpieza (máscara, winsor ±25 %, sin ceros) | AUC 0.5165–0.5196; IC 0.014–0.018; alfa neto −0.53 a −0.57 %. El 0.52 es sobre todo nivel por ticker (AUC demeaned por fecha 0.508). La limpieza casi no cambia nada. MDE 0.026 > IC → no concluyente |
| N0-B | Persistencia por emisora, cortes de liquidez, adversarial, decil | Persistencia t→t+1: rho 0.014 (p≈0.4), no persiste. Tercil líquido IC 0.035 (t 3.27) pero alfa −0.48 %, 0/6 folds. Adversarial AUC 0.869 (deriva fuerte). 2024+ IC ≈ 0.003–0.009 |

Scripts: `scripts/n0a_diagnostico.py`, `n0b_diagnostico.py` · Resultados: `data/eval/n0a_*`, `n0b_*`.

## Nivel 1 — ideas con tesis (cerrado, no pasa)

| Exp | Qué | Resultado |
|---|---|---|
| N1-A | Lead-lag US→MX (SPY, EWZ, EEM, XLE, HG, TNX, MXN, VIX, EWW, ADR) | dIC ≤ 0.003; alfa neto < 0 en 0/6 folds. Señal solo en día t+1, no en h=5. Descartado |
| N1-B | Target a rank / residual, h = 1, 5, 21, reversión 1d, fin de mes | Ridge sobre ranks: IC ≈ 0.030–0.033 (t ≈ 4.4–4.7), 2024+ ≈ 0.025. Alfa neto < 0 (turnover 0.4–0.7). Configs que "pasan" (rank h21, rev1d h21): t del alfa 0.3–0.5, DSR 0.01–0.02 → ruido. Fin de mes: bruto +0.34 % (t 2.52), neto ≈ 0 |
| N1-C | Abstención / conformal / top-k con buffer; TabPFN | Casi todo alfa neto < 0; mejor celda +0.031 % con t 0.16 y 2024+ negativo. IC condicional por dispersión no monótono. TabPFN no viable (RAM), sustituido por LightGBM: IC 0.028 (h5), 0.017 (h21), peor que Ridge. DSR máx 0.005 (N = 179) |

Scripts: `scripts/n1a_leadlag.py`, `n1b_target.py`, `n1c_abstencion.py`, `n1c_lgbm.py` · Resultados: `data/eval/n1a_*`, `n1b_*`, `n1c_*`.

### Lectura tras Nivel 1

El IC ≈ 0.03 del Ridge sobre ranks es una señal estadística real y sobrevive en 2024+, pero no cubre el costo de 0.30 %/lado: el alfa neto es negativo en prácticamente todas las variantes y ningún DSR se acerca a 0.95. Queda como posible componente de ranking donde el costo ya se paga por otra razón; no cumple el criterio de éxito.

## Nivel 2 — cambiar el objetivo (en curso)

Idea A (motor de riesgo: shrinkage/HRP, escala por vol/cola) y B (modelo de liquidez/costo por emisora). Métricas pre-registradas en los scripts (MDD, CVaR, Calmar, Sharpe neto vs 1/N, EWMA vol targeting, min-var).

| Exp | Estado |
|---|---|
| N2-A motor de riesgo | **cerrado, no mejora Sharpe neto ni MDD de forma robusta** (ver abajo) |
| N2-B liquidez/costo | **cerrado, hipótesis no se confirma** (ver abajo) |

### N2-A — resultado (OOS agrupado, neto de 0.30 %/lado, rebalanceo mensual, 35 emisoras)

| config | vol | MDD | CVaR5 | Calmar | Sharpe |
|---|---|---|---|---|---|
| 1/N | 15.2 % | −34.5 % | −2.17 % | 0.17 | 0.44 |
| min-var muestral | 12.4 % | −31.5 % | −1.71 % | 0.13 | 0.39 |
| HRP-LW | 13.2 % | −33.2 % | −1.87 % | 0.13 | 0.39 |
| risk parity LW | 13.9 % | −33.9 % | −1.98 % | 0.15 | 0.43 |
| 1/N + vol targeting EWMA | 10.4 % | −19.3 % | −1.45 % | 0.11 | 0.25 |
| 1/N + cuantil EWMA (FHS5) | 9.3 % | −19.3 % | −1.30 % | 0.08 | 0.20 |
| HRP-LW + LGBM cuantil | 10.0 % | −18.4 % | −1.35 % | 0.13 | 0.30 |

- Shrinkage (LW/OAS/const-corr) no se distingue de la covarianza muestral (Diebold-Mariano p 0.24–0.50). HRP y risk parity tienen más varianza realizada que min-var (p < 0.01).
- Vol targeting EWMA baja el MDD en 6/6 folds, pero el Sharpe cae 0.44 → 0.25 con costo 0.30 %/lado (el hallazgo previo de "mismo Sharpe" usaba 0.05 %). La diferencia de MDD viene casi toda de 2020 (F2: −33.9 % vs −15.1 %); en los demás folds es ruido (0–4 pts).
- Cuantil (FHS / LightGBM) reduce MDD/CVaR por menor exposición, no por mayor eficiencia. Kupiec/Christoffersen no rechazan cobertura del 5 % en ningún modelo.
- Tilt Ridge-rank sobre HRP: turnover 1.75 → 3.5 y 6.1 por año, sin mejora.
- Límites: el MDD agrupado depende casi solo de 2020; 2008 queda en train; no se igualó exposición entre FHS y EWMA.

Script: `scripts/n2a_riesgo.py` · Resultados: `data/eval/n2a_riesgo.{json,md}` · 38 trials `n2a_*`.

### N2-B — resultado (costo por emisora, señal Ridge-rank de 1B)

Costo/lado estimado con CS60 + Roll60 + Amihud60 (datos ≤ t−1): `max(piso 0.15 %, 0.10 % comisión + mult·(S/2 + 0.5·Amihud·Q))`, Q = 1M MXN, tope 1 %. Supuestos propios, sin datos de ejecución real.

- El costo estimado en OOS (percentiles 5/25/50/75/95): 0.26 / 0.35 / 0.54 / 0.73 / 1.0 %. **El 0.30 % fijo era optimista**: nadie cae en el piso.
- Filtrar a emisoras baratas conserva el IC (≈ 0.036–0.04) pero el alfa bruto se evapora (≈ 0).

| config | bruto | costo | neto | folds+ | 2024+ |
|---|---|---|---|---|---|
| h5 costo fijo 0.30, top10_b10 | 0.098 | 0.202 | −0.103 | 1/6 | −0.107 |
| h5 costo por emisora, sin filtro | 0.098 | 0.374 | −0.275 | 1/6 | −0.281 |
| h5 costo por emisora, filtro q50 | 0.005 | 0.121 | −0.116 | 2/6 | −0.072 |
| h5 con penalización de turnover (ca5) | 0.046 | 0.025 | +0.021 (t 0.23) | 4/6 | −0.263 |
| h21 costo fijo 0.30, top10_b10 | 0.274 | 0.225 | +0.049 (t 0.25) | 4/6 | −0.091 |
| h21 costo por emisora, ca5 | 0.192 | 0.075 | +0.117 (t 0.45) | 3/6 | +0.013 |

- Mejor DSR de todas las configs: 0.008 (N global = 277 trials). Ninguna se acerca a 0.95.
- Liquidez predecible (Ridge vs EWMA20): skill en MAE log de ADV 1.6 % (h5) / 5.0 % (h21); spread CS 7.2 % / 22.6 %. Ganancia útil modesta.
- Cortes por tercil líquido con neto positivo (3–4 nombres, sin folds ni DSR): no se toman como evidencia.
- Sensibilidad: la escala del costo domina (×0.5 → h21 sin filtro +0.007 %, 4/6 folds; ×2 todo se hunde). Roll es ruidoso (correl. Spearman con CS 0.23).

Script: `scripts/n2b_costos.py` · Resultados: `data/eval/n2b_costos.{json,md}` · ≈ 60 trials `n2b_*`.

## Nivel 3 — una prueba barata (cerrado, no pasa)

Elegida: **Prueba B** (features OHLCV de microestructura/memoria, como feature y como gate del Ridge-rank). La Prueba A (hechos relevantes con hora) se descartó por cobertura insuficiente en el repo: noticias por emisora solo WALMEX (~900 titulares, 2018-2020); `hechos` es flujo por casa de bolsa desde 2025-06-30, no hechos relevantes; `data/eps` solo trae emisoras de EE. UU.; `events.csv` es macro. Sigue pendiente conseguir/scrapear hechos relevantes oficiales si se quiere probar esa idea.

| h | modelo | IC (t NW) | dIC vs base (t) | alfa costo emisora (folds+) | alfa 0.30 % fijo (folds+) | DSR |
|---|---|---|---|---|---|---|
| 5 | base | 0.0313 (4.68) | – | −0.514 (0/6) | −0.216 (0/6) | 0.000 |
| 5 | all | 0.0273 (3.94) | −0.0040 (−1.88) | −0.556 (0/6) | −0.262 (0/6) | 0.000 |
| 21 | base | 0.0334 (3.64) | – | −0.483 (0/6) | −0.166 (2/6) | 0.000 |
| 21 | micro | 0.0362 (3.89) | +0.0028 (0.45) | −0.116 (2/6) | +0.192 (4/6) | 0.000 |
| 21 | all | 0.0346 (3.17) | +0.0012 (0.14) | +0.100 (5/6) | +0.394 (5/6) | 0.007 |

- Ninguna variante añade IC material. h=21 `all` tiene alfa neto positivo pero IC 2024+ = −0.007 y alfa 2024+ = −0.543: ruido.
- Gates por terciles de Hurst/entropía (48 celdas): 3 pasaron el umbral de régimen útil. El único interpretable: h=21, entropía de permutación, tercil transversal T3, IC 0.050 con el score base (t 4.10), positivo en 6/6 folds, 2024+ 0.060. Pero su alfa neto es −0.33 % (1/6 folds) y lo logra también el score base, así que no lo aportan los features nuevos. **Candidato a replicar con prueba pre-registrada nueva, no un resultado** (48 celdas → riesgo de selección).
- DSR máximo 0.007 con N = 341.

Script: `scripts/n3_prueba.py` · Resultados: `data/eval/n3_prueba.{json,md}` · 64 trials `n3_*`.

Sin probar (descartadas por costo/beneficio): Chronos (RAM/GPU), transfer entropy, reglas débiles con FDR, hechos relevantes oficiales.

## Nivel 4 — combinación pre-registrada (cerrado, no pasa)

Diseño fijado antes de correr (docstring de `scripts/n4_combinacion.py`): S1 = Ridge-rank h21; S2 = promedio simple de rank Ridge y rank rev1d; top-10 mensual con buffer (entra 10, sale 20), entrada Open t+1, costo por emisora de N2-B (×1; sensibilidad ×0.5 y ×2), con y sin vol targeting EWMA 10 %. Tope 8 configs. Criterio endurecido con costo ×1: (a) IC t(NW) > 2; (b) alfa neto > 0 en ≥ 4/6 folds; (c) alfa neto 2024+ > 0; (d) bootstrap 95 % del alfa excluye 0; (e) DSR reportado.

| config | alfa neto % | t | folds+ | 2024+ % | IC 95 % bootstrap | MDD | Sharpe |
|---|---|---|---|---|---|---|---|
| S1 sin capa | −0.136 | −0.70 | 2/6 | −0.295 | [−0.50, 0.24] | −38.2 % | 0.26 |
| S2 sin capa | −0.037 | −0.19 | 3/6 | −0.087 | [−0.33, 0.25] | −38.6 % | 0.29 |
| S1 con capa | −0.289 | −2.25 | 2/6 | −0.371 | [−0.52, −0.05] | −19.4 % | 0.00 |
| S2 con capa | −0.279 | −2.50 | 1/6 | −0.316 | [−0.47, −0.09] | −19.3 % | −0.03 |
| S1 sin capa ×0.5 | +0.007 | 0.03 | 4/6 | −0.152 | [−0.37, 0.36] | −36.9 % | 0.36 |
| S2 sin capa ×0.5 | +0.121 | 0.60 | 3/6 | +0.083 | [−0.17, 0.44] | −38.5 % | 0.40 |
| S1 sin capa ×2 | −0.308 | −1.59 | 2/6 | −0.460 | [−0.71, 0.07] | −40.6 % | 0.15 |
| S2 sin capa ×2 | −0.236 | −1.18 | 3/6 | −0.291 | [−0.53, 0.06] | −39.0 % | 0.16 |

Referencia 1/N con mismo rebalanceo y costo ×1: MDD −33.7 %, Sharpe 0.45 (con capa: −17.5 %, 0.19). S1 y S2 quedan **por debajo del 1/N** en Sharpe.

Veredicto (costo ×1): (a) se cumple (S1 IC 0.0334 t 3.64; S2 0.0308 t 4.64); (b) NO (2/6, 3/6, 2/6, 1/6); (c) NO en ninguna de las 4 principales; (d) NO (sin capa incluye 0; con capa excluye 0 del lado negativo); (e) DSR 0.000–0.001 con N = 350 (máx 0.011 en ×0.5).

- Capa de riesgo: baja el MDD de ~−38 % a ~−19 % (2020: −35 % → −13.6 %; 2022: −20.9 % → −16.7 %), pero el Sharpe cae a ≈ 0 y el alfa neto empeora (paga costo por cambiar de peso).
- Score aleatorio (200 permutaciones, costo ×1): alfa neto medio −0.43 %; S1 en el percentil 94.5 del neto, S2 en el 97.5. Hay algo sobre el azar, pero no alcanza a ser alfa neto positivo.
- Costo fijo 0.30 % (referencia, no registrada): S1 +0.049 % (t 0.25), S2 +0.155 % (t 0.77) — vuelve a mostrar que el resultado depende de cuánto se asuma de costo.
- Réplica de entropía (h=21, tercil T3, score S1; NO independiente): IC 0.0503 (t 4.10), 6/6 folds positivos, 2024+ 0.0598; alfa neto −0.090 % (t −0.29), 3/6 folds+, bootstrap [−0.64, 0.50]. Hay ranking dentro de T3, pero con ~3 nombres por fecha el costo (0.32 %) supera al alfa bruto (0.23 %).

Script: `scripts/n4_combinacion.py` · Resultados: `data/eval/n4_combinacion.{json,md}` · 9 trials `n4_*`.

## Cierre

**Ningún nivel cumple el criterio de parada, tampoco el endurecido del Nivel 4.** Trials: 55 → 350. DSR máximo 0.018 (N creciente). Resultado: negativo, documentado.

Lo que sí queda en pie:
- IC por fecha ≈ 0.03 (t ≈ 4.5) con Ridge sobre ranks de features, también en 2024+ (≈ 0.025), concentrado en emisoras líquidas. Real estadísticamente, no explotable tras costos (costo realista mediana ≈ 0.54 %/lado; el 0.30 % supuesto era optimista).
- El AUC 0.52 original era sobre todo nivel por ticker, no orden relativo.
- Vol targeting baja el MDD (6/6 folds) pero con costo realista pierde Sharpe; el beneficio de MDD viene casi todo de 2020.
- Pistas sin explotar: gate por entropía de permutación (h=21) y conseguir hechos relevantes oficiales con hora.

## Notas de proceso

- Cada configuración se registra con `src/ml/trials.py` (`log_trial`, familias `n0a_*` … `n2a_*`).
- Nada commiteado todavía; archivos nuevos en `scripts/n*_*.py` y `data/eval/n*_*.{json,md}`.
- Limpieza hecha: venv de TabPFN (1.4 GB, scratchpad) borrado.
