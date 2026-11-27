# Arquitectura Medallion - Evaluación/Riesgo

Este directorio contiene dos implementaciones del Medallion
de Evaluación y Riesgo.

| Implementación | Script | Producto Gold |
|---|---|---|
| Primer avance | `scripts/medallion_evaluacion.py` | `evaluacion_riesgo.json` |
| Múltiples estrategias | `scripts/medallion_estrategias.py` | `evaluacion_multiples_estrategias.json` |

Para reproducir el producto de múltiples estrategias,
consultar la sección correspondiente al final de este documento.

La documentación siguiente describe la implementación
original del primer avance.


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

---

## Implementación de múltiples estrategias

Esta implementación amplía el producto de Evaluación y Riesgo
mediante seis estrategias financieras homologadas:

- Top-K 2.
- Top-K 5.
- Top-K 10.
- Top-K 5 Ranking.
- SMA 50/200.
- RSI 14.

Utiliza un benchmark común NAFTRAC Buy and Hold y condiciones
comparables de calendario, capital, moneda y costos.

### Bronze de múltiples estrategias

Conserva diez archivos de entrada:

- Seis resultados originales de estrategias.
- Un archivo del universo tradicional.
- Un benchmark.
- Un catálogo de estrategias homologadas.
- Un reporte de comparación y validación.

Genera un manifiesto con hashes SHA-256.

### Silver de múltiples estrategias

Verifica la integridad y consistencia de las fuentes,
los calendarios, costos, curvas de capital, rendimientos,
benchmark y resultados de validación.

Genera tres archivos de resultados y un reporte de calidad.

En la verificación final se obtuvieron seis estrategias
validadas, cero errores y nueve advertencias.

### Gold de múltiples estrategias

Genera el producto:

`data/eval/medallion/gold/evaluacion_multiples_estrategias.json`

Versión del esquema: `1.0.0`.

Consolida métricas, rankings, benchmark, criterios de
aprobación, calidad, trazabilidad y decisión financiera.

La decisión depende del conjunto de criterios de validación,
no exclusivamente del ratio de Sharpe.

Resultados históricos:

| Indicador | Resultado |
|---|---|
| Calidad | VALIDO |
| Estrategias evaluadas | 6 |
| Estrategias aprobadas | 0 |
| Estrategias no aprobadas | 6 |
| Decisión | NO_BET_LIVE |

La calidad técnica del producto no implica aprobación
financiera de las estrategias.

### Ejecución de múltiples estrategias

Desde la raíz del repositorio y con el entorno virtual
previamente preparado:

```bash
python -m scripts.medallion_estrategias
```

El pipeline utiliza los resultados financieros previamente
generados. No requiere volver a entrenar los modelos.

Archivos JSON generados:

| Capa | Archivos |
|---|---:|
| Bronze | 11 |
| Silver | 4 |
| Gold | 1 |
| Total | 16 |

### Pruebas de múltiples estrategias

```bash
python -m pytest -v \
    tests/test_medallion_estrategias.py \
    tests/test_evaluacion_semantica.py \
    tests/test_mcp_estrategias.py
```

### Consultas semánticas MCP

El producto Gold de múltiples estrategias se consulta
mediante las siguientes herramientas:

- `resumen_estrategias`
- `consultar_estrategia`
- `comparar_estrategias`
- `consultar_benchmark`
- `consultar_validacion`
- `consultar_calidad`
- `consultar_decision`

Las siete herramientas fueron verificadas mediante llamadas
directas y solicitudes HTTP al servidor MCP local.

### Consideraciones sobre las dos implementaciones

Los dos pipelines utilizan los mismos directorios de capas
Medallion. No debe asumirse que sus manifiestos y reportes
de calidad pueden coexistir sin interferencias.

Para utilizar las herramientas semánticas de múltiples
estrategias, reconstruir previamente su producto mediante:

```bash
python -m scripts.medallion_estrategias
```

Los resultados del primer avance no deben confundirse con
las métricas homologadas de múltiples estrategias.

Los archivos JSON generados permanecen excluidos de Git
y pueden reconstruirse a partir de las fuentes correspondientes.

### Documentación complementaria

- [Evaluación y Riesgo](../../../docs/evaluacion_riesgo.md)
- [Reproducibilidad](../../../docs/REPRODUCIBILIDAD_EVALUACION.md)
- [Primer avance](../../../docs/evaluacion_riesgo_primer_avance.md)
