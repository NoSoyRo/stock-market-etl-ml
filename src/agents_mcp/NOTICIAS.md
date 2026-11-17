# Integración del dominio Noticias

La arquitectura publicada en `origin/main` (`f9a9899`) agrega agentes de
dominio con `DomainAgent.ask()` y herramientas MCP del Claude Agent SDK.
Esta rama integra esa arquitectura y conserva su base (`domain_agent.py` y los inicializadores)
sin cambios; la adaptación específica corresponde al dominio Noticias. El orquestador y el supervisor de main
ya importan `src.agents_mcp.domains.noticias.noticias_agent`; ese contrato se conserva.
La integración de `main` ya conserva esas llamadas sin modificaciones.

El agente consulta nuestro medallion mediante `NewsSemanticService`, en vez de
leer únicamente los snapshots `data/news/YYYY-MM-DD.json`. Reutiliza validaciones,
estados de datos antiguos/ausentes, integridad de evidencia y cortes temporales.
La base del agente conserva el loop del SDK de main y solo permite las tools de
su propio dominio. Los otros dominios y el orquestador de main también están disponibles en esta rama.

Herramientas disponibles:

| Tool | Uso |
| --- | --- |
| `schema_noticias` | Contrato de datos y reglas de interpretación. |
| `query_noticias` | Empresa o tema; fecha/corte opcionales; sin entidad resume hasta 15 empresas del último día disponible. |
| `historial_noticias` | Serie histórica con huecos; conserva las ventanas de cálculo. |
| `evidencia_noticias` | Titulares, enlaces, procedencia y paginación del resultado gold. |
| `cobertura_noticias` | Fechas, empresas, temas, fuentes y métodos existentes. |

Sin fecha se usa el último resultado **de esa entidad** disponible en el corte,
con su fecha real y advertencias de antigüedad. Una fecha explícita nunca se
sustituye por otra. Los cortes deben tener zona horaria y no pueden ser futuros.
Las consultas leen `gold/historical_features.parquet` y `silver/articles.parquet`
bajo `data/news`; no descargan ni recalculan. Para preparar estos datos se conserva
`scripts/build_news_medallion.py` (por ejemplo, `--replay --rebuild-gold-history`).
Las salidas `data/news/noticias/golden` de la práctica PySpark tienen otro esquema
y método sin ponderación: no se consumen como si fueran las métricas ML.

Para evaluar sesgos, `query_noticias` y `evidencia_noticias` admiten
`anonimizar=true` para una empresa. El código sustituye ticker/alias por `E1` y
omite enlaces e identificadores que podrían revelar la empresa antes de entregar
el resultado al modelo. El modo normal conserva la evidencia y su procedencia.

Instalación y uso directo del agente (requiere acceso al modelo configurado en
`domain_agent.py` y autenticación del Claude Agent SDK):

```bash
pip install -r requirements.txt
```

```python
import asyncio
from src.agents_mcp.domains.noticias import noticias_agent

print(asyncio.run(noticias_agent.ask(
    'Noticias recientes de WALMEX: sentimiento, fecha, fuentes y evidencia'
)))
```

Para otra raíz de datos: `create_news_agent(news_root=...)`. Para probar tools
sin API/modelo: `create_news_tools(news_root=...)` y ejecutar su `handler`.

```bash
python -m pytest tests/test_news_domain_agent.py -q
```

Estas pruebas usan el SDK real y Parquet de prueba; no hacen llamadas pagadas.
Verifican el contrato esperado por el orquestador, anonimización, selección
histórica, cortes, evidencia y ausencia de datos.
