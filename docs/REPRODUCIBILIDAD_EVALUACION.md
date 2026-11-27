# Guía de reproducibilidad: evaluación y riesgo

## 1. Objetivo

Reproducir los resultados de evaluación de múltiples estrategias,
su integración Medallion y las consultas de la capa semántica MCP.

Todos los comandos deben ejecutarse desde la raíz del repositorio.

## 2. Entorno utilizado

- Sistema operativo: Linux mediante WSL.
- Python: 3.14.4.
- Entorno virtual: .venv-linux.

Activación del entorno existente:

    source .venv-linux/bin/activate

Versiones de las dependencias principales verificadas:

| Dependencia | Versión |
|---|---|
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| pyarrow | 26.0.1 |
| scikit-learn | 1.9.1 |
| lightgbm | 4.7.0 |
| yfinance | 1.8.0 |
| requests | 2.34.2 |
| python-dotenv | 1.2.4 |
| loguru | 0.8.0 |
| pytest | 9.2.0 |
| mcp | 2.3.0 |
| claude-agent-sdk | 0.2.189 |
| pydantic | 2.14.4 |
| jsonschema | 4.26.0 |
| uvicorn | 0.54.0 |
| starlette | 1.10.0 |

Comprobar la instalación:

    python --version
    python -m pip check

Las versiones anteriores corresponden al entorno utilizado durante
las pruebas y no constituyen un archivo de bloqueo completo
de todas las dependencias del proyecto.

## 3. Datos requeridos

Para reproducir las simulaciones completas se requiere disponer
de los datos históricos locales utilizados en la evaluación.

Directorios principales:

    data/market/
    data/rates/banxico/
    data/estrategias_corregidas/
    data/estrategias_homologadas/
    data/eval/

Los datos de mercado proceden de Yahoo Finance.

Las tasas de interés proceden del Sistema de Información
Económica del Banco de México.

No se deben incluir credenciales ni tokens de acceso
en los archivos versionados.

La actualización de los datos históricos puede modificar
los resultados respecto a la evaluación documentada.

## 4. Reproducción rápida

Este procedimiento utiliza los resultados de evaluación
existentes y no requiere volver a entrenar modelos.

### 4.1. Verificar archivos de entrada

    ls -lh data/estrategias_corregidas/
    ls -lh data/estrategias_homologadas/
    ls -lh data/eval/strategies_comparison_real.json

### 4.2. Generar las capas Medallion

    python -m scripts.medallion_estrategias

El procedimiento genera:

- Bronze: archivos originales y manifiesto de integridad.
- Silver: archivos validados y reporte de calidad.
- Gold: producto consolidado de evaluación.

Producto final:

    data/eval/medallion/gold/evaluacion_multiples_estrategias.json

### 4.3. Consultar la capa semántica

Resumen general:

    python -c "from src.agents_mcp.domains.evaluacion_semantica import resumen_estrategias, serializar; print(serializar(resumen_estrategias()))"

Consultar SMA:

    python -c "from src.agents_mcp.domains.evaluacion_semantica import consultar_estrategia, serializar; print(serializar(consultar_estrategia('sma_50_200')))"

Comparar por Sharpe:

    python -c "from src.agents_mcp.domains.evaluacion_semantica import comparar_estrategias, serializar; print(serializar(comparar_estrategias('sharpe_neto')))"

Consultar decisión:

    python -c "from src.agents_mcp.domains.evaluacion_semantica import consultar_decision, serializar; print(serializar(consultar_decision()))"

Estas consultas no requieren una API de inteligencia artificial.

## 5. Reproducción completa de las evaluaciones

Este procedimiento puede consumir más tiempo porque las
estrategias Top-K requieren entrenamiento de modelos.

Los comandos regeneran archivos de resultados existentes.
Se recomienda conservar una copia de los resultados anteriores
antes de ejecutar el procedimiento.

### 5.1. Preparar directorios

    mkdir -p data/estrategias_corregidas
    mkdir -p data/estrategias_homologadas

### 5.2. Estrategia Top-K 2

    python -u -m scripts.backtest_topk \
        --universe bmv \
        --top-k 2 \
        --min-p-up 0.56 \
        --score p_up \
        --weight equal \
        --horizon 5 \
        --commission 0.0025 \
        --slippage 0.0005 \
        --output data/estrategias_corregidas/topk_2.json

### 5.3. Estrategia Top-K 5

    python -u -m scripts.backtest_topk \
        --universe bmv \
        --top-k 5 \
        --min-p-up 0.56 \
        --score p_up \
        --weight equal \
        --horizon 5 \
        --commission 0.0025 \
        --slippage 0.0005 \
        --output data/estrategias_corregidas/topk_5.json

### 5.4. Estrategia Top-K 10

    python -u -m scripts.backtest_topk \
        --universe bmv \
        --top-k 10 \
        --min-p-up 0.56 \
        --score p_up \
        --weight equal \
        --horizon 5 \
        --commission 0.0025 \
        --slippage 0.0005 \
        --output data/estrategias_corregidas/topk_10.json

### 5.5. Variante exploratoria Top-K 5 Ranking

    python -u -m scripts.backtest_topk \
        --universe bmv \
        --top-k 5 \
        --min-p-up 0.0 \
        --score p_up \
        --weight equal \
        --horizon 5 \
        --commission 0.0025 \
        --slippage 0.0005 \
        --output data/estrategias_corregidas/topk_5_ranking.json

Esta variante tiene carácter exploratorio y no constituye
una validación independiente.

### 5.6. Estrategias tradicionales

    python -u -m scripts.backtest_traditional

Archivos generados:

    data/estrategias_corregidas/sma_portfolio.json
    data/estrategias_corregidas/rsi_portfolio.json

### 5.7. Homologación diaria de Top-K

    python -u -m scripts.homologate_topk

Reconstruye las curvas diarias y verifica que se conserve
el rendimiento y el capital final de las simulaciones originales.

### 5.8. Benchmark NAFTRAC

    python -u -m scripts.backtest_benchmark

Archivo generado:

    data/estrategias_homologadas/buy_hold_naftrac.json

### 5.9. Consolidación

    python -u -m scripts.consolidate_strategies

Archivos principales:

    data/estrategias_homologadas/strategies_catalog.json
    data/estrategias_homologadas/strategies_summary.json

### 5.10. Validación y comparación

    python -u -m scripts.validate_strategies \
        --catalog data/estrategias_homologadas/strategies_catalog.json \
        --max-drawdown 0.30 \
        --output data/eval/strategies_comparison_real.json

### 5.11. Generación Medallion

    python -u -m scripts.medallion_estrategias

## 6. Resultados de referencia

Periodo: 2023-05-10 a 2026-10-30.

Capital inicial: 100,000 MXN.

Sesiones de evaluación: 876.

| Estrategia | Rendimiento neto | Sharpe neto | Capital final MXN |
|---|---:|---:|---:|
| SMA 50/200 | 12.46% | 0.4213 | 112,463.84 |
| RSI 14 | 11.03% | 0.5008 | 111,029.44 |
| Top-K 2 | 0.00% | 0.0000 | 100,000.00 |
| Top-K 5 | 0.00% | 0.0000 | 100,000.00 |
| Top-K 10 | 0.00% | 0.0000 | 100,000.00 |
| Top-K 5 Ranking | -45.25% | -0.8734 | 54,745.13 |
| Benchmark NAFTRAC | 24.54% | 0.4803 | 124,535.33 |

Resultado esperado con los datos de referencia:

- Calidad del producto: VALIDO.
- Estrategias evaluadas: 6.
- Estrategias aprobadas: 0.
- Estrategias no aprobadas: 6.
- Decisión global: NO_BET_LIVE.

Las simulaciones tradicionales y el benchmark fueron
ejecutados nuevamente y reprodujeron las métricas anteriores.

La reproducción completa de los modelos ML puede depender
de las versiones de software y de los datos utilizados.

## 7. Pruebas automatizadas

    python -m pytest -v \
        tests/test_strategy_validator.py \
        tests/test_strategy_comparator.py \
        tests/test_traditional_backtest.py \
        tests/test_strategy_homologation.py \
        tests/test_medallion_evaluacion.py \
        tests/test_medallion_estrategias.py \
        tests/test_evaluacion_semantica.py \
        tests/test_mcp_estrategias.py \
        tests/test_agents_mcp_evaluacion_lab.py

Estas pruebas comprueban la aportación de evaluación y riesgo
y su integración con los componentes correspondientes.

No sustituyen las pruebas completas de todo el repositorio.

## 8. Servidor MCP

Configurar un token temporal seguro mediante variables de entorno.

    export MCP_REQUIRE_AUTH_TOKEN=1
    read -rs MCP_AUTH_TOKEN
    export MCP_AUTH_TOKEN
    python -m mcp_server.server

El endpoint local predeterminado es:

    http://localhost:8080/mcp

No exponer el servidor a Internet durante las pruebas locales.

Consultar mcp_server/README.md para conocer las herramientas,
argumentos, autenticación y ejemplos de consultas.

## 9. Consideraciones de reproducibilidad

- Mantener los mismos datos históricos.
- Mantener el universo de instrumentos utilizado.
- Conservar las versiones de las dependencias.
- Mantener los costos y parámetros de simulación.
- No ajustar parámetros utilizando los resultados de prueba.
- No confundir resultados históricos con ganancias reales.
- Las fechas de generación y hashes pueden cambiar al regenerar
  archivos, aunque las métricas financieras sean equivalentes.
- Los scripts sin interfaz de argumentos pueden ejecutar su
  procedimiento completo incluso si reciben la opción --help.

## 10. Documentación complementaria

- docs/evaluacion_riesgo.md
- mcp_server/README.md

Los resultados tienen fines educativos y no constituyen
una recomendación de inversión.

## Anexo: restauracion del historial de Banxico

Para evitar versionar 6799 archivos JSON individuales, se entrega
el historial utilizado en el archivo:

    data/rates/banxico_historico.zip

Caracteristicas del conjunto de datos:

- Fuente: Banco de Mexico, API SIE.
- Inicio: 2008-01-02.
- Fin: 2026-11-23.
- Archivos JSON: 6799.
- Archivos invalidos: 0.
- Archivos vacios: 0.
- Tamano original: 377469 bytes.
- Tamano ZIP: 1114427 bytes.

Series utilizadas:

| Variable | Serie Banxico |
|---|---|
| TIIE 28 dias | SF43783 |
| TIIE 91 dias | SF60649 |
| CETES 28 dias | SF43936 |
| CETES 364 dias | SF43945 |
| Tasa objetivo | SF61745 |

### Verificacion del archivo

Desde la raiz del repositorio, ejecutar:

    sha256sum data/rates/banxico_historico.zip

El resultado esperado es:

    aa4b85ad115993809025f59cdf4bfb5607c5ee44f93fc776a50eb7a38b1f47ab

### Restauracion de los datos

En una copia nueva del repositorio, ejecutar:

    python -m zipfile -e data/rates/banxico_historico.zip data/rates/

Verificar la cantidad de archivos recuperados:

    find data/rates/banxico/ -type f -name '*.json' | wc -l

Resultado esperado:

    6799

La extraccion debe realizarse antes de ejecutar el backtesting
Top-K desde cero.

Los algoritmos conservan su mecanismo original de lectura de
archivos JSON individuales. No se requiere modificar el modelo
para utilizar el historial restaurado.

La extraccion puede sobrescribir archivos existentes con el
mismo nombre. Se recomienda realizarla en una copia nueva
del repositorio.

El ZIP fue verificado byte por byte contra los 6799 archivos
originales. Esto garantiza la integridad de la copia, pero no
constituye una auditoria independiente de los valores publicados.

No se incluyen tokens ni credenciales de acceso a la API.

---

## Evidencia de reproducción histórica e integración con upstream

### Identificación de la versión reproducida

- Commit histórico: `2afeb12b00e08e39fc94d15c72b950b806f50409`.
- Commit de upstream integrado: `f5fbbb81371c42a9c29f4850e4361c25691682f8`.
- Rama de entrega: `entrega/multiples-estrategias`.
- Entorno utilizado: WSL y Python 3.14.4.

La reproducción se realizó en un worktree independiente del
repositorio principal, utilizando el commit histórico de la
entrega y el entorno Python previamente configurado.

### Restauración de fuentes históricas

Se utilizaron los 95 archivos Parquet de mercado correspondientes
al commit histórico y se restauraron los datos de Banxico desde
el archivo versionado:

`data/rates/banxico_historico.zip`

Resultados de la restauración:

- Archivos JSON restaurados: 6799.
- Archivos JSON inválidos: 0.
- Archivos JSON vacíos: 0.
- Fecha inicial: 2008-01-02.
- Fecha final: 2026-11-23.
- SHA-256 del ZIP:

  `aa4b85ad115993809025f59cdf4bfb5607c5ee44f93fc776a50eb7a38b1f47ab`

### Reproducción de simulaciones financieras

| Simulación | Resultado |
|---|---|
| SMA 50/200 | Idéntico |
| RSI 14 | Idéntico |
| Benchmark NAFTRAC | Idéntico |
| Top-K 2 | Idéntico |
| Top-K 5 | Idéntico |
| Top-K 10 | Idéntico |
| Top-K Ranking | Idéntico |

Las siete simulaciones fueron ejecutadas nuevamente y sus
resultados JSON comparados contra los archivos del commit
histórico.

La comparación excluyó únicamente el campo generated_at.
No se aplicaron tolerancias numéricas para obtener coincidencias.

### Homologación y consolidación

Se reconstruyeron las cuatro variantes Top-K y el catálogo
consolidado de seis estrategias.

Condiciones financieras comunes:

- Capital inicial: 100000 MXN.
- Inicio: 2023-05-10.
- Fin: 2026-10-30.
- Sesiones bursátiles: 876.
- Puntos de valoración: 877.
- Frecuencia de evaluación: diaria.
- Periodos por año: 252.
- Comisión por lado: 0.25%.
- Slippage por lado: 0.05%.

Los nueve archivos homologados fueron verificados.

El catálogo presentó seis diferencias correspondientes
exclusivamente a campos generated_at anidados.
No se encontraron otras diferencias.

### Resultados financieros reproducidos

| Estrategia | Retorno neto | Sharpe anualizado (base diaria) | Capital final MXN |
|---|---:|---:|---:|
| NAFTRAC | 24.54% | 0.4803 | 124535.33 |
| SMA 50/200 | 12.46% | 0.4213 | 112463.84 |
| RSI 14 | 11.03% | 0.5008 | 111029.44 |
| Top-K 2 | 0.00% | 0.0000 | 100000.00 |
| Top-K 5 | 0.00% | 0.0000 | 100000.00 |
| Top-K 10 | 0.00% | 0.0000 | 100000.00 |
| Top-K Ranking | -45.25% | -0.8734 | 54745.13 |

### Validación financiera reproducida

Se reconstruyó el reporte strategies_comparison_real.json.

La única diferencia detectada fue generated_at.

- Estrategias evaluadas: 6.
- Estrategias aprobadas: 0.
- Estrategias no aprobadas: 6.
- Diferencias adicionales: 0.

En total se realizaron 17 verificaciones de resultados,
correspondientes a 16 archivos distintos:

- 7 resultados de simulaciones.
- 9 archivos homologados y consolidados.
- 1 reporte de validación.

El benchmark NAFTRAC se contabilizó tanto en las simulaciones
como en los archivos homologados.

Todos coincidieron con los originales, excluyendo únicamente
las marcas temporales de generación.

### Reconstrucción de Medallion

Bronze:

- Archivos incorporados: 10.

Silver:

- Estado: VALIDO.
- Estrategias validadas: 6.
- Errores: 0.
- Advertencias: 9.

Gold:

- Producto: evaluacion_multiples_estrategias.
- Calidad: VALIDO.
- Estrategias aprobadas: 0.
- Estrategias no aprobadas: 6.
- Decisión: NO_BET_LIVE.

Las advertencias de calidad se conservaron y no fueron
eliminadas para obtener una validación artificial.

### Verificación de la capa semántica

Se ejecutaron directamente las siguientes funciones Python
contra el producto Gold reconstruido:

| Consulta | Resultado |
|---|---|
| resumen_estrategias | OK |
| consultar_estrategia | OK |
| comparar_estrategias | OK |
| consultar_benchmark | OK |
| consultar_validacion | OK |
| consultar_calidad | OK |
| consultar_decision | OK |

Las siete consultas utilizaron el producto
evaluacion_multiples_estrategias, versión 1.0.0.

### Integración de datos con upstream

Durante la integración se identificaron 31 conflictos:

- 30 archivos Parquet.
- 1 documento Markdown.

En los 30 archivos Parquet en conflicto se conservaron
las 32 correcciones históricas de nuestra entrega y las
211 filas adicionales provenientes de upstream.

La auditoría general de los 95 archivos Parquet identificó:

- 92 archivos sin diferencias hasta el corte histórico.
- 3 archivos con registros históricos adicionales.
- 712 registros posteriores al corte histórico.
- 715 registros adicionales en total.
- 0 registros históricos modificados.
- 0 registros históricos eliminados.

Los tres registros históricos adicionales corresponden a
NAFTRAC.MX, POSADASA.MX y VISTAA.MX.

Por este motivo, para reproducir exactamente los resultados
originales se utilizó el commit histórico completo, en lugar
de limitar únicamente las fechas de los datos integrados.

### Alcance y limitaciones

- La reproducción utilizó un entorno Python existente.
- No constituye una certificación de instalación desde cero.
- Los archivos Gold se reconstruyeron y verificaron mediante
  sus controles de consistencia y consultas semánticas.
- No se afirma igualdad binaria de los productos Medallion.
- Las pruebas directas de funciones semánticas no sustituyen
  las pruebas de integración del protocolo MCP.
- Los resultados corresponden a simulaciones históricas.
- Top-K Ranking es una variante exploratoria.
- Ninguna estrategia fue aprobada para operar con dinero real.
- La integración final con upstream se verificó posteriormente
  mediante las pruebas documentadas en el siguiente apartado.

---

## Verificación final de integración y reproducibilidad

### 1. Identificación de la versión integrada

La verificación final se realizó sobre el commit integrado
`237df46`, utilizando la rama `entrega/finalizacion`.

Se conservó el trabajo de Evaluación y Riesgo junto con las
actualizaciones incorporadas desde upstream.

La reproducción histórica de las simulaciones corresponde al
commit histórico identificado en el apartado anterior. Las
pruebas de integración corresponden al commit integrado.

### 2. Entorno de ejecución

Se preparó un entorno virtual independiente en una segunda
computadora utilizando WSL y Ubuntu.

| Componente | Versión |
|---|---|
| Python | 3.14.4 |
| Java OpenJDK | 21.0.13.1 |
| NumPy | 2.5.4 |
| Pandas | 3.1.1 |
| PyArrow | 26.0.1 |
| PySpark | 4.3.0 |
| Scikit-learn | 1.9.1 |
| LightGBM | 4.7.0 |
| YFinance | 1.8.0 |
| Pytest | 9.2.0 |
| Pydantic | 2.14.4 |
| JSON Schema | 4.26.0 |
| SQLGlot | 30.35.0 |
| MCP | 2.3.0 |
| Claude Agent SDK | 0.2.189 |
| Uvicorn | 0.54.0 |
| Starlette | 1.10.0 |
| AnyIO | 4.16.0 |

El entorno virtual utilizado fue `.venv-linux`.

La comprobación `python -m pip check` no detectó dependencias
incompatibles según los metadatos de los paquetes instalados.

Se verificaron las importaciones requeridas, el procesamiento
distribuido local de Spark, la ejecución de funciones Python
en trabajadores de Spark y la conversión de DataFrames a Pandas.

La instalación se limitó a las dependencias necesarias para
el alcance de las pruebas. No constituye una certificación
de todas las dependencias y dominios del repositorio.

### 3. Restauración de datos de Banxico

Se restauró nuevamente el archivo versionado:

`data/rates/banxico_historico.zip`

Resultados:

- Archivos JSON restaurados: 6799.
- Archivos inválidos: 0.
- Archivos vacíos: 0.
- Fecha inicial: 2008-01-02.
- Fecha final: 2026-11-23.
- Integridad SHA-256 del ZIP: verificada.

La restauración no requirió credenciales de acceso a Banxico.

### 4. Ejecución aislada del ETL de Mercado

Para evitar modificaciones accidentales sobre el repositorio
principal, el ETL se ejecutó en una copia independiente.

Se copiaron los módulos de código fuente y los archivos
Parquet originales, sin reutilizar las salidas generadas
por ejecuciones anteriores.

Entradas verificadas mediante SHA-256:

| Dominio | Archivos Parquet |
|---|---:|
| Mercado histórico | 95 |
| Mercado intradía | 36 |
| Total | 131 |

Los 131 archivos conservaron su integridad durante las pruebas.

#### 4.1. Bronze

Comando ejecutado:

```bash
python -u -m src.etl.bronze.mercado.etl_bronze
```

| Indicador | Resultado |
|---|---:|
| CSV de mercado | 94 |
| Registros de mercado | 495329 |
| CSV intradía | 36 |
| Registros intradía | 1390350 |
| Total de registros | 1885679 |
| Tiempo de ejecución | 14.509 s |
| Código de salida | 0 |

El archivo ALFAA.MX.parquet estaba vacío y fue omitido
por el pipeline.

Se verificaron los conteos por archivo, la estructura de
los CSV y la correspondencia con los metadatos Parquet.

#### 4.2. Silver

Comando ejecutado:

```bash
python -u -m src.etl.silver.mercado.etl_silver
```

| Indicador | Resultado |
|---|---:|
| Registros de mercado | 495329 |
| Registros intradía | 1390350 |
| Total de registros | 1885679 |
| Registros perdidos respecto a Bronze | 0 |
| Tiempo de ejecución | 16.823 s |
| Código de salida | 0 |

Se verificaron las llaves, los valores numéricos, las fechas,
los indicadores binarios y la ausencia de registros duplicados.

Indicadores de calidad del mercado histórico:

| Indicador | Registros |
|---|---:|
| Volumen cero | 11414 |
| Precio no positivo | 8 |
| OHLC incoherente | 497 |
| Salto superior al 25% | 314 |
| Cierre repetido | 12148 |

Indicadores de calidad intradía:

| Indicador | Registros |
|---|---:|
| Festivo | 30889 |
| Posible falla de fuente | 126874 |
| Día plano | 204638 |
| Día incompleto | 33683 |

Se identificaron 46 fechas distintas asociadas con posibles
fallas de fuente.

Los indicadores pueden superponerse y no deben sumarse como
si fueran categorías mutuamente excluyentes.

La identificación de festivos y fallas de fuente utiliza
heurísticas del pipeline; no constituye una confirmación
independiente de eventos reales.

#### 4.3. Gold

Comando ejecutado:

```bash
python -u -m src.etl.gold.mercado.etl_gold
```

| Indicador | Resultado |
|---|---:|
| Registros históricos | 495329 |
| Registros históricos válidos | 489396 |
| Registros históricos no válidos | 5933 |
| Registros sin tipo de cambio disponible | 43082 |
| Registros intradía conservados | 1232587 |
| Registros intradía excluidos | 157763 |
| Resúmenes diarios intradía | 15668 |
| Resúmenes diarios completos | 15176 |
| Resúmenes diarios incompletos | 492 |
| Resúmenes diarios planos | 601 |
| Archivos CSV generados | 166 |
| Tiempo de ejecución | 20.830 s |
| Código de salida | 0 |

Se verificaron las transformaciones monetarias, los
rendimientos, las variables objetivo, las llaves y los
resúmenes diarios mediante cálculos independientes.

Las observaciones sin tipo de cambio disponible se
conservaron con los valores derivados correspondientes
ausentes, según las reglas del pipeline.

El tiempo acumulado de ejecución de Bronze, Silver y Gold
fue de 52.162 segundos, sin incluir las auditorías.

### 5. Verificación de SQLite

El ETL Gold generó la base de datos:

`data/market/gold/mercado.db`

La base se generó dentro de la copia aislada.

| Tabla | Registros | Columnas |
|---|---:|---:|
| precios_historicos | 495329 | 13 |
| tipo_cambio | 5974 | 2 |
| intradia_diario | 15668 | 11 |
| Total | 516971 | |

El total representa filas almacenadas entre las tres tablas,
no observaciones financieras únicas.

Resultados de auditoría:

- Tamaño de la base: 78127104 bytes.
- PRAGMA integrity_check: ok.
- Tablas esperadas: 3 de 3.
- Llaves primarias: verificadas.
- Registros duplicados por llave primaria: 0.
- Índice idx_precios_fecha: verificado.
- Correspondencia de datos CSV y SQLite: verificada.
- Consultas SQL de comprobación: 5 de 5 exitosas.

La base contiene los resúmenes diarios intradía, no la
totalidad de las barras individuales de cinco minutos.

### 6. Reconstrucción del Medallion de Evaluación

Comando ejecutado en el repositorio integrado:

```bash
python -u -m scripts.medallion_estrategias
```

Se utilizaron los resultados financieros previamente
generados y versionados. Esta reconstrucción no requirió
volver a entrenar los modelos ni repetir los backtests.

| Capa | Archivos JSON | Resultado |
|---|---:|---|
| Bronze | 11 | Verificado |
| Silver | 4 | VALIDO |
| Gold | 1 | VALIDO |
| Total | 16 | Verificado |

Bronze incorporó diez archivos de entrada y un manifiesto.
Se verificó la igualdad SHA-256 de las diez entradas
respecto a sus fuentes en el repositorio integrado.

Silver conservó la trazabilidad de las fuentes y presentó:

- Estrategias validadas: 6.
- Errores: 0.
- Advertencias: 9.
- Benchmark validado: sí.

Gold consolidó el producto
`evaluacion_multiples_estrategias`, versión `1.0.0`.

- Estrategias evaluadas: 6.
- Estrategias aprobadas: 0.
- Estrategias no aprobadas: 6.
- Decisión: NO_BET_LIVE.

La validez técnica de los datos no implica la aprobación
financiera de las estrategias.

### 7. Pruebas automatizadas de integración

Se ejecutaron las siguientes pruebas:

```bash
python -m pytest -v \
    tests/test_strategy_validator.py \
    tests/test_strategy_comparator.py \
    tests/test_traditional_backtest.py \
    tests/test_strategy_homologation.py \
    tests/test_medallion_evaluacion.py \
    tests/test_medallion_estrategias.py \
    tests/test_evaluacion_semantica.py \
    tests/test_mcp_estrategias.py \
    tests/test_agents_mcp_evaluacion_lab.py \
    tests/test_evaluacion.py \
    --tb=short
```

| Módulo | Pruebas aprobadas |
|---|---:|
| Validador de estrategias | 8 |
| Comparador de estrategias | 5 |
| Backtesting tradicional | 10 |
| Homologación | 10 |
| Medallion Evaluación | 12 |
| Medallion Estrategias | 16 |
| Evaluación semántica | 30 |
| MCP Estrategias | 34 |
| Agentes MCP Evaluación y Lab | 8 |
| Evaluación general | 11 |
| Total | 144 |

Resultados en la segunda computadora:

- Pruebas aprobadas: 144.
- Pruebas fallidas: 0.
- Pruebas omitidas: 0.
- Código de salida: 0.

Las mismas 144 pruebas habían sido aprobadas previamente
en la primera computadora, con un tiempo de 84.34 segundos.

Los tiempos son observaciones de ejecución y no representan
un benchmark controlado de hardware.

### 8. Verificación de la capa semántica

Se ejecutaron directamente las siete funciones de Evaluación
contra el producto Gold reconstruido.

| Función | Resultado |
|---|---|
| resumen_estrategias | OK |
| consultar_estrategia | OK |
| comparar_estrategias | OK |
| consultar_benchmark | OK |
| consultar_validacion | OK |
| consultar_calidad | OK |
| consultar_decision | OK |

Las siete respuestas utilizaron el mismo producto Gold
y la versión de esquema 1.0.0.

### 9. Verificación del protocolo MCP mediante HTTP

Se inició el servidor MCP utilizando el transporte
Streamable HTTP y autenticación Bearer.

Configuración local de prueba:

```bash
export HOST=127.0.0.1
export PORT=8080
export MCP_AUTH_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export MCP_REQUIRE_AUTH_TOKEN=1

python -u -m mcp_server.server
```

Las variables de dirección y puerto utilizadas por el
servidor son HOST y PORT.

La dirección de escucha se verificó mediante `ss`.

Resultados:

| Verificación | Resultado |
|---|---|
| Dirección de escucha | 127.0.0.1:8080 |
| Solicitud sin token | HTTP 401 |
| Solicitud con token incorrecto | HTTP 401 |
| Solicitud con token correcto | HTTP 200 |
| Inicialización MCP | OK |
| Versión del protocolo | 2025-06-18 |
| Notificación initialized | HTTP 202 |
| Solicitud tools/list | HTTP 200 |
| Herramientas registradas | 14 |
| Herramientas duplicadas | 0 |
| Herramientas originales conservadas | 7 |
| Herramientas de Evaluación registradas | 7 |

Se ejecutaron las siete herramientas de Evaluación mediante
solicitudes JSON-RPC tools/call.

| Herramienta | HTTP | Igualdad con función directa |
|---|---:|---|
| resumen_estrategias | 200 | Exacta |
| consultar_estrategia | 200 | Exacta |
| comparar_estrategias | 200 | Exacta |
| consultar_benchmark | 200 | Exacta |
| consultar_validacion | 200 | Exacta |
| consultar_calidad | 200 | Exacta |
| consultar_decision | 200 | Exacta |

La igualdad se verificó comparando los objetos Python
obtenidos de las respuestas HTTP con las respuestas
de las funciones semánticas directas.

Se comprobó mediante SHA-256 que el archivo Gold no fue
modificado durante las consultas.

Finalmente se detuvo el servidor, se verificó la liberación
del puerto 8080 y se eliminaron las variables temporales
de autenticación utilizadas durante las pruebas.

No se incluyen tokens ni identificadores de sesión reales
en esta documentación.

### 10. Alcance y limitaciones de la verificación final

- Las 144 pruebas corresponden al alcance seleccionado de
  Evaluación e integración, no a toda la suite del repositorio.
- El registro de las siete herramientas originales de MCP
  no implica que todas sus funcionalidades hayan sido probadas.
- Las consultas HTTP no requirieron una conversación con
  un modelo de lenguaje ni un despliegue en producción.
- No se verificó aislamiento completo de red durante
  la reproducción histórica.
- PySpark presentó advertencias sobre bibliotecas nativas,
  resolución del hostname y compatibilidad con Pandas 3.
- Las advertencias no impidieron completar las ejecuciones
  verificadas, pero no se garantiza compatibilidad universal.
- Las variables objetivo representan la siguiente observación
  disponible según las reglas del pipeline, no necesariamente
  la siguiente sesión oficial o barra exacta de cinco minutos.
- La evaluación histórica está sujeta a sesgo de supervivencia.
- No se incluyen dividendos ni intereses sobre efectivo.
- Las posiciones abiertas se valoran al cierre.
- La variante Top-K Ranking es exploratoria y requiere
  validación independiente.
- Los resultados históricos no garantizan rendimientos futuros.
- Ninguna estrategia fue aprobada para operar con dinero real.
- Los resultados generados del ETL de Mercado permanecieron
  en la copia aislada y no se incorporaron al repositorio.
- No se afirma igualdad binaria entre los productos Gold
  históricos y los reconstruidos en la versión integrada.

### 11. Resultado de la integración

La verificación técnica del alcance de Evaluación y Riesgo
concluyó satisfactoriamente.

Se comprobaron la reproducibilidad histórica, la integración
de datos, la ejecución del ETL de Mercado, la consistencia
de SQLite, la reconstrucción Medallion, las pruebas
automatizadas y las consultas semánticas mediante MCP.

El repositorio permaneció limpio durante las pruebas,
antes de incorporar esta actualización documental.

La decisión financiera se mantiene en NO_BET_LIVE.
