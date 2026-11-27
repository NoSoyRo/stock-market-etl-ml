# Producto de datos: Evaluacion / Riesgo

## 1. Objetivo

El producto de Evaluacion/Riesgo permite organizar y validar
los resultados de backtesting de estrategias financieras.

Su objetivo es proporcionar informacion reproducible
sobre rendimiento, riesgo y experimentos para apoyar
la evaluacion de estrategias.

## 2. Fuentes de datos

El producto utiliza archivos existentes del proyecto:

- data/eval/backtest_topk.json
- data/eval/portfolio_metrics.json
- data/eval/experiments.json

En esta primera version, el exportador procesa
backtest_topk.json. Las otras fuentes quedan
documentadas para futuras integraciones.

## 3. Arquitectura

Flujo inicial:

backtest_topk.json
        |
        v
scripts/export_evaluacion.py
        |
        v
evaluacion_export.json

El resultado contiene dos secciones:

- backtest_runs: configuracion y periodo de evaluacion.
- risk_metrics: rendimiento, riesgo y costos.

El archivo SQL define las tablas donde estos
resultados podran almacenarse posteriormente.

## 4. Esquema de datos

Archivo: sql/04_eval_schema.sql

Tablas propuestas:

- eval.backtest_runs
- eval.risk_metrics
- eval.experiment_log

La base de datos objetivo es SQL Server.

El esquema aun no se ha desplegado ni probado
contra una instancia de SQL Server.

## 5. Contrato de datos

Archivo: contracts/evaluacion_riesgo.yaml

Define:

- Equipo responsable.
- Fuentes de datos.
- Estructura de las tablas.
- Reglas de calidad.
- Consumidores previstos.

## 6. Metricas

Las metricas disponibles incluyen:

- Sharpe bruto y neto.
- Maximum Drawdown.
- Calmar Ratio.
- Win Rate neto.
- Rendimiento bruto y neto.
- AUC.
- Costos de comision y slippage.

Deflated Sharpe Ratio permanece pendiente
de integracion y validacion estadistica.

## 7. Ejecucion

Desde la raiz del repositorio:

    python scripts/export_evaluacion.py

Salida predeterminada:

    data/eval/evaluacion_export.json

Para elegir otra salida:

    python scripts/export_evaluacion.py --output salida.json

## 8. Pruebas

Ejecutar:

    python -m unittest discover -s tests -p "test_evaluacion.py" -v

Se incluyen diez pruebas unitarias para verificar
la estructura de exportacion y algunas reglas
de validacion.

## 9. Resultados iniciales

La ejecucion local sobre los datos disponibles
produjo:

- Sharpe neto: -1.786249091327774
- Maximum Drawdown: -0.5311996654071603

Estos valores corresponden al backtest existente
y no representan resultados de una nueva estrategia.

## 10. Trabajo pendiente

- Integrar portfolio_metrics.json.
- Integrar experiments.json en la bitacora.
- Calcular y validar Deflated Sharpe Ratio.
- Conectar el exportador con SQL Server.
- Probar la creacion de tablas en la base de datos.
- Integrar resultados de otros equipos.

## 11. Alcance del primer avance

Este avance incluye:

1. Esquema SQL propuesto.
2. Contrato de datos inicial.
3. Exportador funcional de backtests.
4. Validaciones basicas.
5. Diez pruebas unitarias.
6. Documentacion tecnica.

No incluye aun la persistencia real en SQL Server.
