# N0-A diagnostico (WoE+Logistic, 6 folds por fechas)

## 1. Historia usable (panel del CV)
- Tickers en panel: 35 (.MX: 0); >50 filas: 35; >250: 35; >1000: 34
- Emisoras por fecha: mediana 30.0, max 35; fechas con >= 8: 4672/4678
- Cobertura quality_mask: 92.3%; excluidos por >5% ceros: 8

## 2-5. Resultados globales por configuracion

| config | filas | tick. | AUC | IC medio | t NW5 | AUC dem.fecha | AUC terc. | AUC dem.ticker | alfa largo neto % | L/S neto % |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 135534 | 35 | 0.5165 | 0.0167 | 2.60 | 0.5079 | 0.5095 | 0.5169 | -0.526 | -1.017 |
| A_valid | 134414 | 35 | 0.5173 | 0.0184 | 2.88 | 0.5091 | 0.5109 | 0.5174 | -0.541 | -1.046 |
| B_valid_winsor25 | 134414 | 35 | 0.5173 | 0.0184 | 2.89 | 0.5091 | 0.5109 | 0.5174 | -0.537 | -1.042 |
| C_valid_winsor25_sin_ceros | 109816 | 27 | 0.5196 | 0.0144 | 2.02 | 0.5078 | 0.5090 | 0.5199 | -0.567 | -1.086 |

## Por fold (config base)

| fold | AUC | IC | t NW5 | AUC dem.fecha | AUC dem.ticker | fechas |
|---|---|---|---|---|---|---|
| 1 | 0.5297 | -0.0032 | -0.20 | 0.4973 | 0.5358 | 388 |
| 2 | 0.5115 | 0.0115 | 0.78 | 0.5055 | 0.5146 | 389 |
| 3 | 0.4908 | 0.0500 | 3.35 | 0.5289 | 0.4965 | 389 |
| 4 | 0.5216 | 0.0145 | 0.95 | 0.5095 | 0.5270 | 389 |
| 5 | 0.5326 | 0.0209 | 1.19 | 0.5128 | 0.5322 | 389 |
| 6 | 0.5225 | 0.0063 | 0.43 | 0.4942 | 0.5258 | 389 |

## 6. Poder / MDE

| config | sigma IC | periodos 5d no solapados | MDE IC (t>2, pot 80%) | MDE (a=.05, pot 80%) |
|---|---|---|---|---|
| base | 0.200 | 466 | 0.0263 | 0.0230 |
| A_valid | 0.199 | 466 | 0.0262 | 0.0229 |
| B_valid_winsor25 | 0.199 | 466 | 0.0262 | 0.0229 |
| C_valid_winsor25_sin_ceros | 0.224 | 466 | 0.0294 | 0.0258 |

## Veredicto

- base: t>2=True, alfa neto>0=False, pasa=False, clase=senal_detectada
- A_valid: t>2=True, alfa neto>0=False, pasa=False, clase=senal_detectada
- B_valid_winsor25: t>2=True, alfa neto>0=False, pasa=False, clase=senal_detectada
- C_valid_winsor25_sin_ceros: t>2=True, alfa neto>0=False, pasa=False, clase=senal_detectada

## Notas
- DSR no calculado aqui (criterio alterno DSR>0.95 se evalua en tandas posteriores).
- IC por fecha solo con >= 8 emisoras; t Newey-West lag 5 (y 10).
- Retorno cero medido en retornos diarios Close->Close del store por ticker.
- Cada config reentrena WoE+LR por fold con train y test filtrados igual.
- Alfa: rebalanceo no solapado cada 5 fechas; largo tercil superior vs promedio universo, costo 2*0.30%; L/S 4*0.30%. Sin impacto de mercado ni borrow.
