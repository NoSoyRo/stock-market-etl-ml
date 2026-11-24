# Spec: agente de dominio Features/ML (`features_ml`)

Estado: listo para implementar · Fecha: 2026-11-22

**Alcance:** solo `src/agents_mcp/domains/features_ml.py` y los módulos, catálogo y tests nuevos del agente de ML.
**No se modifica:** `orchestrator.py`, `mcp_server/`, `quant_senior.py`, `domain_agent.py` ni los demás dominios (son de otros equipos).

## 1. Objetivo y rol

Adaptar el `features_ml_agent` existente (no crear uno nuevo) para que sea el **auditor y explicador de ML** del sistema: responde qué modelos hay, qué evidencia tienen, cuántas pruebas se han hecho y qué dicen las señales registradas, con honestidad estadística.

Por qué este rol (hechos del repo):
- El ML **no decide el libro**: lo decide una regla fija (`src/agent/core_satellite.py`) y Haiku solo veta (`docs/SISTEMA_END_TO_END.md` §1, §9).
- La evidencia es negativa y está documentada: `data/eval/agent_scorecard.md`, reportes `n0`–`n4` y `avances/2026-11-19_ultimo_intento_ml.md` (350 trials, DSR máximo 0.018). Lo único que queda es un IC ≈ 0.03 de Ridge-rank, no explotable tras costos.
- Un agente que recomienda sería deshonesto con ese historial; uno que audita y explica es consistente con él.

**No hace:** recomendar compra/venta, afinar parámetros mirando backtests, entrenar modelos, escribir en disco, llamar a la red.

## 2. Restricciones de compatibilidad

| Consumidor | Qué usa | Obligación |
|---|---|---|
| `orchestrator.py`, `mcp_server/server.py` | `features_ml_agent` (objeto con `.ask()`) | Mantener el nombre y la interfaz `DomainAgent`. `name="features-ml"` no cambia. |
| `quant_senior.py` | `pronostico_precio.handler({...})` y `recomendar_portafolio.handler({})`, leyendo `result["content"][0]["text"]` | Ambos nombres siguen exportados, con `.handler` y el mismo envoltorio `{"content":[{"type":"text","text":...}]}`. El `text` pasa a ser JSON (§4); `quant_senior` lo trata como string, así que sigue funcionando. |

Verificado por grep: nadie fuera de `features_ml.py` importa `query_features_ml` ni `schema_features_ml`.

## 3. Arquitectura

```
src/agents_mcp/domains/
  features_ml.py     # DomainAgent + registro de tools (delgado)
  ml_service.py      # lógica pura, sin SDK ni LLM, testeable (nuevo)
  ml_catalog.yaml    # catálogo curado de modelos y veredictos (nuevo)
tests/test_agents_mcp_features_ml.py   # nuevo
```

- `ml_service.py` expone funciones síncronas que devuelven `dict`. Las tools las invocan con `asyncio.to_thread` (patrón de `noticias.py`).
- Fábricas `create_ml_tools(root=ROOT, *, service=None)` y `create_ml_agent(root=ROOT)`, igual que `create_news_tools` / `create_news_agent`. `features_ml_agent = create_ml_agent()` se mantiene a nivel de módulo.
- Subclase `MLDomainAgent(DomainAgent)` que sobreescribe `build_options()` y añade `max_turns=MAX_TURNS` y `max_budget_usd=MAX_BUDGET_USD` a `ClaudeAgentOptions` (campos soportados por claude-agent-sdk 0.2.180). Así no se toca `domain_agent.py`.
- Las tools de lectura se declaran con `annotations=ToolAnnotations(readOnlyHint=True)` (`from mcp.types import ToolAnnotations`).
- Lectura de YAML con `pyyaml` (ya en `requirements.txt`).
- Modelo: `claude-haiku-5-5` (default de `DomainAgent`).

## 4. Contrato de respuesta de toda tool

El `text` es JSON (`ensure_ascii=False`, `allow_nan=False`, usando `json_safe` de `src.semantic.models` como en `noticias.py`):

```json
{
  "ok": true,
  "as_of": "2026-10-31",
  "fuentes": ["data/eval/n4_combinacion.json"],
  "n_trials_global": 350,
  "evidencia": "senal_debil_no_explotable",
  "advertencias": [],
  "datos": {}
}
```

Errores: `{"ok": false, "error_tipo": "...", "mensaje": "..."}` con `error_tipo` en `modelo_desconocido | archivo_ausente | argumento_invalido | corte_futuro | timeout`. Las tools nunca propagan excepciones.

`evidencia` toma uno de estos valores: `sin_skill`, `senal_debil_no_explotable`, `explotable`, `no_evaluado`, `demo_no_vigente`. Sale del catálogo, nunca la decide el LLM. `explotable` solo puede aparecer si el catálogo la declara con fuente; hoy ninguna entrada la tiene. Las tools que no tratan de un modelo concreto (`registro_experimentos`, `calcular_dsr`, `estado_track_record`) devuelven `evidencia: null`.

## 5. Tools

| Tool | Entrada | Devuelve | Fuente |
|---|---|---|---|
| `schema_features_ml` (existe) | — | Inventario, reglas, qué está en producción vs investigación y las etiquetas de evidencia. Ahora en JSON. | catálogo |
| `registro_experimentos` | `familia?` (str), `prefijo?` (str), `limite` (int 1-50, default 10) | N global y por familia (hashes únicos), `ts` del último trial y los últimos `limite` trials. | `data/eval/trial_log.jsonl` vía `trial_stats.n_trials` y `trials.count_trials` |
| `veredicto_modelo` | `modelo` (enum = ids del catálogo) | Etiqueta de evidencia, estado, resumen, métricas resueltas, límites conocidos y `as_of` del reporte. | catálogo + `data/eval/*.json` |
| `senales_recientes` | `fecha?` (YYYY-MM-DD), `corte?` (ISO-8601 con zona) | Régimen, apuestas, top 10, `hash` y `stale`. Marcadas como informativas: no deciden el libro. | `data/signals/history/*.json` |
| `estado_track_record` | — | Nº de señales, rango de fechas y cuántas ya cumplieron el horizonte de 20 días hábiles (`numpy.busday_offset`). Solo conteos. | `data/signals/history/index.json` |
| `calcular_dsr` | `sharpe_anual` (float), `anios` (float > 0), `n_trials?` (int; default N global), `skew?` (default 0), `kurtosis?` (default 3, no excedente) | `dsr`, `sr0_anual`, `n_trials` usado y lectura (`dsr > 0.95` = evidencia). | `src/ml/trial_stats.py` (`expected_max_sharpe`, `probabilistic_sharpe_ratio`); **no** el de `purged_cv.py` |
| `plan_experimento` | `hipotesis`, `familia`, `config` (objeto), `criterio_exito` | Borrador de preregistro en texto, si el `config_hash` ya existe en el log y cuánto subiría N. No escribe ni ejecuta nada. | `trial_stats.config_hash`, `trial_log.jsonl` |
| `pronostico_precio` (existe, degradada) | `ticker`, `horizonte_dias?` (default 42) | Los 4 modelos de `scripts/forecast_horizons.py` (`run_sin_ancla`), ejecutados con `to_thread` y timeout, en `datos.modelos[]` con retorno %, MAE del modelo y del naive y etiqueta; más promedio y dispersión como números. `evidencia: sin_skill`. Se describe como diagnóstico experimental, no pronóstico. | `scripts/forecast_horizons.py` |
| `recomendar_portafolio` (existe, degradada) | — | Pesos y métricas del JSON, con `as_of` = mtime del archivo, `evidencia: demo_no_vigente` y advertencia de que la regla vigente del libro está en el dominio `lab`. | `data/eval/forecast_demo/portfolio_optimo_cov.json` |

`query_features_ml` se elimina; la cubren `registro_experimentos` y `veredicto_modelo`.

**Frescura (`stale`).** Una señal es `stale` si su fecha es más de 3 días calendario anterior al corte (por defecto, hoy en UTC). Los reportes de `data/eval/` no tienen regla de `stale`: su `as_of` es el `generated_at` si existe y, si no, el mtime del archivo. Las señales pueden traer `regime: "?"`; se reporta tal cual.

**Corte.** `corte` debe traer zona horaria y no ser futuro; si lo es, `corte_futuro`. Con `fecha` explícita nunca se sustituye por otra: si no existe ese archivo, `archivo_ausente`. Sin `fecha` se usa la última señal con fecha ≤ corte.

## 6. Catálogo (`ml_catalog.yaml`)

Ubicación: `src/agents_mcp/domains/ml_catalog.yaml`.

Una entrada por modelo con: `id`, `nombre`, `estado` (`produccion` | `investigacion` | `descartado`), `evidencia`, `resumen`, `fuentes[]`, `metricas` y `limites[]`.

Los reportes de `data/eval/` tienen estructuras distintas entre sí, así que cada métrica se declara con ruta de puntos y se lee del archivo en cada consulta; el valor nunca se copia al catálogo:

```yaml
- id: ridge_rank
  nombre: Ridge sobre ranks de features
  estado: investigacion
  evidencia: senal_debil_no_explotable
  resumen: IC por fecha ≈ 0.03 (t ≈ 4.4) pero alfa neto negativo con costo realista.
  fuentes: [data/eval/n1b_target.json, data/eval/n4_combinacion.json]
  metricas:
    ic_h5:        {archivo: data/eval/n1b_target.json, ruta: cfg.ridge_rank_h5.ic}
    t_nw_h5:      {archivo: data/eval/n1b_target.json, ruta: cfg.ridge_rank_h5.t_nw}
    folds_ic_pos: {archivo: data/eval/n1b_target.json, ruta: cfg.ridge_rank_h5.folds_ic_pos}
    dsr_global:   {archivo: data/eval/n1b_target.json, ruta: cfg.ridge_rank_h5.dsr_global.dsr}
    ic_s1_n4:     {archivo: data/eval/n4_combinacion.json, ruta: meta.ic_score.S1.ic}
  limites: [No cubre el costo realista (mediana ≈ 0.54 %/lado)., IC 2024+ ≈ 0.024.]
```

Reglas:
- Si una ruta no resuelve o el archivo falta, esa métrica sale `null` con una advertencia y la tool no falla.
- Las 10 entradas son: `supermodel`, `ridge_rank`, `lightgbm_ranking`, `woe`, `lstm_tcn`, `regimen_hmm`, `ising`, `noticias_sentimiento`, `reversion_5d`, `vol_targeting`.
- Veredictos y `resumen` se toman de `avances/2026-11-19_ultimo_intento_ml.md`, `avances/2026-11-05_panel_ideas_ml.md`, `data/eval/agent_scorecard.{md,json}` y `docs/SISTEMA_END_TO_END.md` §9.
- Estado: `supermodel` = `produccion` (corre diario con `scripts/record_signals.py`, pero no decide el libro); el resto = `investigacion` o `descartado` según esos documentos.
- Un test verifica que todas las rutas del catálogo resuelven contra los archivos reales del repo.

## 7. Disciplina que impone el agente

Va en `SYSTEM_PROMPT` y se prueba:

1. Toda cifra de desempeño lleva N de trials y fuente. El DSR se calcula siempre con el N global.
2. Se mide con IC por fecha, entrada Open t+1 y costo realista (mediana ≈ 0.54 %/lado; 0.30 % es optimista). No usa AUC agrupado como argumento.
3. Ante "la mejor configuración" o "afínalo hasta que salga": rechaza, explica que cada variante sube N y ofrece `plan_experimento` (preregistro).
4. Distingue siempre producción, investigación y descartado.
5. Nunca da orden de compra/venta ni presenta una salida de modelo sin su etiqueta de evidencia.
6. Si el `as_of` es viejo o `stale`, lo dice. Ausencia de dato no significa dato neutral.
7. Backtests y DSR de estrategias completas → dominio `evaluacion`. Decisión del libro → `lab`. Noticias → `noticias`. No los duplica.

## 8. Límites operativos

Constantes al inicio de `features_ml.py`:

```python
MAX_TURNS = 6
MAX_BUDGET_USD = 0.25
TIMEOUT_PRONOSTICO_S = 60
```

- Ninguna tool entrena ni corre `SuperModel`.
- `pronostico_precio` es la única con cómputo: se ejecuta con `asyncio.to_thread` bajo `asyncio.wait_for(..., TIMEOUT_PRONOSTICO_S)`; al vencer devuelve `error_tipo: timeout`.
- Sin escritura en disco ni red.

## 9. Pruebas

Archivo `tests/test_agents_mcp_features_ml.py`. Sin LLM ni red. Usa el patrón de `tests/test_news_domain_agent.py`: servicio y tools aislados con `create_ml_tools(root=tmp_path)`, invocados con `.handler`, y `pytest.importorskip("claude_agent_sdk")`.

- `registro_experimentos`: N por familia = hashes únicos; el global coincide con `trials.count_trials`.
- `veredicto_modelo`: modelo desconocido → `modelo_desconocido`; la etiqueta viene del catálogo; ruta rota → `null` + advertencia.
- `senales_recientes`: `corte` futuro → `corte_futuro`; señal de más de 3 días → `stale: true`; `fecha` inexistente → `archivo_ausente`.
- `estado_track_record`: cuenta bien las señales cerradas con fechas de prueba.
- `calcular_dsr`: coincide con `trial_stats.deflated_sharpe` sobre una serie equivalente; a mayor N, menor DSR.
- `plan_experimento`: detecta un hash existente; no crea ni modifica archivos (comparar el directorio antes y después).
- `pronostico_precio` y `recomendar_portafolio`: existen, tienen `.handler`, devuelven `content[0].text` parseable como JSON y `quant_senior` sigue importándolos.
- `pronostico_precio` corre fuera del event loop (monkeypatch lento) y devuelve `timeout` al vencer.
- Contrato: toda tool devuelve `ok`, `fuentes`, `evidencia`, `n_trials_global`.
- Catálogo: toda `ruta` de `ml_catalog.yaml` resuelve contra los archivos reales de `data/eval/`.
- Prompt: `SYSTEM_PROMPT` contiene las 7 reglas del §7.
- Compatibilidad: `from src.agents_mcp.orchestrator import build_options` sigue funcionando (importa `features_ml_agent`).

### Criterios de aceptación

- `features_ml_agent` sigue importable y con `.ask()`.
- `orchestrator.py`, `mcp_server/`, `quant_senior.py` y `domain_agent.py` no tienen cambios en el diff.
- Toda tool devuelve el contrato del §4.
- Pasan los tests nuevos y los de agentes existentes (`test_agents_mcp_fundamentales.py`, `test_news_domain_agent.py`, `test_news_orchestrator_integration.py`, `test_investor_agent.py`).
- Ninguna tool escribe en disco ni entrena.

## 10. Orden de implementación

1. `ml_service.py` + `ml_catalog.yaml` + sus tests (servicio y catálogo), sin tocar `features_ml.py`.
2. Tools nuevas, `create_ml_tools`, `create_ml_agent` y `MLDomainAgent`; reescribir `schema_features_ml`; eliminar `query_features_ml`.
3. Degradar `pronostico_precio` y `recomendar_portafolio` conservando `.handler`.
4. Tests de compatibilidad y de prompt; correr toda la suite de agentes.
5. Verificación manual con el SDK real: una pregunta de auditoría (por ejemplo, "¿qué evidencia hay de Ridge-rank?") y una de "afínalo hasta que dé Sharpe > 1", que debe terminar en rechazo con oferta de `plan_experimento`.
