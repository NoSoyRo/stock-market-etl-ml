
# Arquitectura Medallion - Evaluación/Riesgo

## Objetivo

Implementar una arquitectura Medallion exclusiva del módulo de
Evaluación/Riesgo del proyecto Stock Market ETL/ML.

El pipeline organiza, valida y consolida resultados financieros
existentes sin modificar los procesos de ingesta, entrenamiento
o transformación de otros equipos.

## Arquitectura

### Bronze

Recibe copias de los resultados originales:

- `data/eval/backtest_topk.json`
- `data/eval/portfolio_metrics.json`
- `data/eval/experiments.json`

Genera copias idénticas y un manifiesto con hashes SHA-256
para verificar la integridad de cada archivo.

### Silver

Valida la calidad de los resultados Bronze:

- Existencia de archivos y campos requeridos.
- Integridad de los archivos mediante SHA-256.
- Tipos de datos numéricos y booleanos.
- Rangos válidos de AUC, win rate y drawdown.
- Consistencia de métricas entre backtesting y portafolio.

Los resultados negativos de una estrategia no se consideran
errores de calidad de datos.

Genera:

`data/eval/medallion/silver/quality_report.json`

### Gold

Consolida las métricas verificadas en un producto de datos
para Evaluación/Riesgo.

Incluye:

- Sharpe bruto y neto.
- Rendimiento neto.
- Drawdown máximo.
- Win rate y AUC.
- Comparación con NAFTRAC.
- Resumen de experimentos.
- Estado de calidad y fuentes utilizadas.

Si Silver detecta errores, Gold no se genera.

La clasificación `NO_BET_LIVE` se utiliza cuando el Sharpe
neto es negativo. Un Sharpe no negativo produce
`REQUIERE_REVISION`, no una autorización automática para invertir.

Genera:

`data/eval/medallion/gold/evaluacion_riesgo.json`

## Ejecución

Desde la raíz del repositorio:

```bash
python3 scripts/medallion_evaluacion.py
```

## Pruebas

```bash
python3 -m unittest tests.test_medallion_evaluacion -v
```

## Alcance

Esta implementación aplica Medallion a los resultados del
producto Evaluación/Riesgo.

No modifica los datos originales, los modelos de Machine
Learning, los scripts de ingesta, los esquemas SQL ni los
componentes de otros equipos.

Los archivos JSON generados dentro de Medallion se excluyen
de Git porque pueden regenerarse mediante el pipeline.
