"""Servicio de solo lectura para el agente de dominio Features/ML.

La lógica vive aquí para poder probarla sin Claude Agent SDK ni llamadas de red.
No entrena modelos, no ejecuta SuperModel y no escribe archivos.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.ml import trial_stats
from src.ml.trials import count_trials


class MLService:
    """Consultas auditables sobre catálogo, trials, señales y diagnósticos ML."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.catalog_path = self.root / "src" / "agents_mcp" / "domains" / "ml_catalog.yaml"
        self.trial_log_path = self.root / "data" / "eval" / "trial_log.jsonl"
        self.signals_dir = self.root / "data" / "signals" / "history"

    def _n_global(self) -> int:
        return count_trials(path=str(self.trial_log_path))

    def _base(
        self,
        *,
        ok: bool = True,
        as_of: str | None = None,
        fuentes: list[str] | None = None,
        evidencia: str | None = None,
        advertencias: list[str] | None = None,
        datos: dict[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": ok,
            "as_of": as_of,
            "fuentes": fuentes or [],
            "n_trials_global": self._n_global(),
            "evidencia": evidencia,
            "advertencias": advertencias or [],
            "datos": datos or {},
        }
        out.update(extra)
        return out

    def _error(self, tipo: str, mensaje: str, **kwargs: Any) -> dict[str, Any]:
        return self._base(ok=False, error_tipo=tipo, mensaje=mensaje, **kwargs)

    def _load_catalog(self) -> list[dict[str, Any]]:
        if not self.catalog_path.exists():
            return []
        data = yaml.safe_load(self.catalog_path.read_text(encoding="utf-8")) or {}
        models = data.get("modelos", data if isinstance(data, list) else [])
        return list(models or [])

    @staticmethod
    def _dot_get(data: Any, route: str) -> Any:
        cur = data
        for part in route.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                raise KeyError(route)
        return cur

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _report_as_of(self, path: Path, data: dict[str, Any] | None = None) -> str:
        data = data or {}
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        candidates = [
            data.get("generated_at"),
            data.get("execution_date"),
            meta.get("generated_at"),
            meta.get("execution_date"),
        ]
        for value in candidates:
            if value:
                return str(value)
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

    def _trial_rows(self) -> list[dict[str, Any]]:
        if not self.trial_log_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.trial_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    @staticmethod
    def _trial_name(row: dict[str, Any]) -> str:
        result = row.get("result") or {}
        return str(result.get("name") or row.get("family") or "")

    def schema(self) -> dict[str, Any]:
        models = self._load_catalog()
        if not models:
            return self._error(
                "archivo_ausente",
                "No se encontró o está vacío el catálogo ML.",
                fuentes=["src/agents_mcp/domains/ml_catalog.yaml"],
            )
        return self._base(
            fuentes=["src/agents_mcp/domains/ml_catalog.yaml"],
            datos={
                "rol": "auditor_y_explicador_ml",
                "modelos": [
                    {
                        "id": m.get("id"),
                        "nombre": m.get("nombre"),
                        "estado": m.get("estado"),
                        "evidencia": m.get("evidencia"),
                    }
                    for m in models
                ],
                "estados": ["produccion", "investigacion", "descartado"],
                "evidencias": [
                    "sin_skill",
                    "senal_debil_no_explotable",
                    "explotable",
                    "no_evaluado",
                    "demo_no_vigente",
                ],
                "reglas": [
                    "Solo lectura: no entrena, no escribe y no ejecuta SuperModel.",
                    "Las cifras de desempeño deben acompañarse por N de trials y fuente.",
                    "No emite órdenes de compra/venta.",
                ],
            },
        )

    def registro_experimentos(
        self,
        familia: str | None = None,
        prefijo: str | None = None,
        limite: int = 10,
    ) -> dict[str, Any]:
        if not isinstance(limite, int) or not 1 <= limite <= 50:
            return self._error("argumento_invalido", "limite debe estar entre 1 y 50.")
        rows = self._trial_rows()
        if not self.trial_log_path.exists():
            return self._error(
                "archivo_ausente",
                "No existe data/eval/trial_log.jsonl.",
                fuentes=["data/eval/trial_log.jsonl"],
            )

        families = sorted({str(r.get("family") or "") for r in rows if r.get("family")})
        por_familia = {
            fam: trial_stats.n_trials(fam, self.trial_log_path)
            for fam in families
        }

        selected = rows
        if familia:
            selected = [r for r in selected if r.get("family") == familia]
        if prefijo:
            selected = [
                r
                for r in selected
                if self._trial_name(r).startswith(prefijo)
                or str(r.get("family") or "").startswith(prefijo)
            ]

        ultimos = [
            {
                "ts": row.get("ts"),
                "family": row.get("family"),
                "name": self._trial_name(row),
                "hash": row.get("hash"),
            }
            for row in selected[-limite:]
        ]

        return self._base(
            as_of=rows[-1].get("ts") if rows else None,
            fuentes=["data/eval/trial_log.jsonl"],
            datos={
                "n_por_familia": por_familia,
                "filtro_familia": familia,
                "filtro_prefijo": prefijo,
                "n_coincidencias": len(selected),
                "ultimo_ts": selected[-1].get("ts") if selected else None,
                "ultimos": ultimos,
            },
        )

    def veredicto_modelo(self, modelo: str) -> dict[str, Any]:
        catalog = {str(m.get("id")): m for m in self._load_catalog()}
        if modelo not in catalog:
            return self._error(
                "modelo_desconocido",
                f"Modelo desconocido: {modelo!r}.",
                fuentes=["src/agents_mcp/domains/ml_catalog.yaml"],
                datos={"modelos_validos": sorted(catalog)},
            )

        item = catalog[modelo]
        warnings: list[str] = []
        metrics: dict[str, Any] = {}
        as_of_values: list[str] = []

        for metric_name, spec in (item.get("metricas") or {}).items():
            rel = Path(str(spec.get("archivo") or ""))
            route = str(spec.get("ruta") or "")
            path = self.root / rel
            if not path.exists():
                metrics[metric_name] = None
                warnings.append(f"No existe {rel.as_posix()} para {metric_name}.")
                continue
            try:
                data = self._read_json(path)
                metrics[metric_name] = self._dot_get(data, route)
                as_of_values.append(self._report_as_of(path, data))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                metrics[metric_name] = None
                warnings.append(
                    f"No se pudo resolver {metric_name} en {rel.as_posix()}:{route} ({exc})."
                )

        sources = [str(x) for x in item.get("fuentes") or []]
        if not as_of_values:
            for rel_str in sources:
                path = self.root / rel_str
                if path.exists():
                    try:
                        data = self._read_json(path) if path.suffix == ".json" else {}
                        as_of_values.append(self._report_as_of(path, data))
                        break
                    except (OSError, json.JSONDecodeError):
                        continue

        return self._base(
            as_of=max(as_of_values) if as_of_values else None,
            fuentes=sources,
            evidencia=item.get("evidencia"),
            advertencias=warnings,
            datos={
                "id": item.get("id"),
                "nombre": item.get("nombre"),
                "estado": item.get("estado"),
                "resumen": item.get("resumen"),
                "metricas": metrics,
                "limites": list(item.get("limites") or []),
            },
        )

    @staticmethod
    def _parse_cutoff(
        corte: str | None,
        now: datetime | None = None,
    ) -> tuple[datetime | None, str | None]:
        now = now or datetime.now(timezone.utc)
        if corte is None:
            return now, None
        try:
            parsed = datetime.fromisoformat(corte.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None, "corte debe ser ISO-8601 con zona horaria."
        if parsed.tzinfo is None:
            return None, "corte debe incluir zona horaria."
        parsed = parsed.astimezone(timezone.utc)
        if parsed > now.astimezone(timezone.utc):
            return None, "corte_futuro"
        return parsed, None

    def senales_recientes(
        self,
        fecha: str | None = None,
        corte: str | None = None,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        cutoff, error = self._parse_cutoff(corte, now=now)
        if error == "corte_futuro":
            return self._error("corte_futuro", "El corte no puede estar en el futuro.")
        if error:
            return self._error("argumento_invalido", error)
        assert cutoff is not None

        selected_path: Path | None = None
        signal_date: date | None = None

        if fecha is not None:
            try:
                signal_date = date.fromisoformat(fecha)
            except (TypeError, ValueError):
                return self._error(
                    "argumento_invalido",
                    "fecha debe tener formato YYYY-MM-DD.",
                )
            selected_path = self.signals_dir / f"{fecha}.json"
            if not selected_path.exists():
                return self._error(
                    "archivo_ausente",
                    f"No existe señal para {fecha}.",
                    fuentes=[f"data/signals/history/{fecha}.json"],
                )
            if signal_date > cutoff.date():
                return self._error(
                    "argumento_invalido",
                    "La fecha solicitada es posterior al corte.",
                    fuentes=[f"data/signals/history/{fecha}.json"],
                )
        else:
            candidates: list[tuple[date, Path]] = []
            if self.signals_dir.exists():
                for path in self.signals_dir.glob("????-??-??.json"):
                    try:
                        d = date.fromisoformat(path.stem)
                    except ValueError:
                        continue
                    if d <= cutoff.date():
                        candidates.append((d, path))
            if candidates:
                signal_date, selected_path = max(candidates, key=lambda x: x[0])

        if selected_path is None or signal_date is None:
            return self._error(
                "archivo_ausente",
                "No hay señales registradas hasta el corte solicitado.",
                fuentes=["data/signals/history/"],
            )

        try:
            rec = self._read_json(selected_path)
        except (OSError, json.JSONDecodeError) as exc:
            return self._error(
                "archivo_ausente",
                f"No se pudo leer la señal: {exc}",
                fuentes=[str(selected_path.relative_to(self.root))],
            )

        age_days = (cutoff.date() - signal_date).days
        rel = str(selected_path.relative_to(self.root))
        return self._base(
            as_of=str(rec.get("timestamp_utc") or signal_date.isoformat()),
            fuentes=[rel],
            advertencias=[
                "Señal informativa de Features/ML: no decide el libro vigente."
            ],
            datos={
                "date": signal_date.isoformat(),
                "regime": rec.get("regime"),
                "bets": rec.get("bets") or [],
                "top10": rec.get("top10_ranking") or [],
                "hash": rec.get("hash"),
                "stale": age_days > 3,
                "edad_dias": age_days,
            },
        )

    def estado_track_record(self, *, now: datetime | None = None) -> dict[str, Any]:
        path = self.signals_dir / "index.json"
        if not path.exists():
            return self._error(
                "archivo_ausente",
                "No existe el índice del track record.",
                fuentes=["data/signals/history/index.json"],
            )
        try:
            data = self._read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            return self._error(
                "archivo_ausente",
                f"No se pudo leer el track record: {exc}",
                fuentes=["data/signals/history/index.json"],
            )

        today = (now or datetime.now(timezone.utc)).date()
        valid_dates: list[date] = []
        closed = 0
        for entry in data.get("entries") or []:
            try:
                d = date.fromisoformat(str(entry.get("date")))
            except ValueError:
                continue
            valid_dates.append(d)
            close_np = np.busday_offset(np.datetime64(d), 20, roll="forward")
            close_date = date.fromisoformat(str(close_np))
            if close_date <= today:
                closed += 1

        return self._base(
            as_of=str(
                data.get("last_updated")
                or (max(valid_dates).isoformat() if valid_dates else None)
            ),
            fuentes=["data/signals/history/index.json"],
            datos={
                "n_signales": len(valid_dates),
                "fecha_inicio": min(valid_dates).isoformat() if valid_dates else None,
                "fecha_fin": max(valid_dates).isoformat() if valid_dates else None,
                "horizonte_dias_habiles": 20,
                "n_horizonte_cumplido": closed,
                "n_pendientes": len(valid_dates) - closed,
            },
        )

    def calcular_dsr(
        self,
        sharpe_anual: float,
        anios: float,
        n_trials: int | None = None,
        skew: float = 0.0,
        kurtosis: float = 3.0,
    ) -> dict[str, Any]:
        try:
            sr_ann = float(sharpe_anual)
            years = float(anios)
            skew = float(skew)
            kurtosis = float(kurtosis)
            trials = self._n_global() if n_trials is None else int(n_trials)
        except (TypeError, ValueError):
            return self._error("argumento_invalido", "Parámetros numéricos inválidos.")
        if years <= 0 or trials < 1 or kurtosis <= 0:
            return self._error(
                "argumento_invalido",
                "anios debe ser > 0, n_trials >= 1 y kurtosis > 0.",
            )

        periods = 252
        n_obs = max(int(round(years * periods)), 2)
        sr_period = sr_ann / math.sqrt(periods)
        sr0_ann = trial_stats.expected_max_sharpe(trials, years)
        sr0_period = sr0_ann / math.sqrt(periods)
        dsr = trial_stats.probabilistic_sharpe_ratio(
            sr_period,
            n_obs,
            sr0_period,
            skew,
            kurtosis,
        )
        return self._base(
            fuentes=["src/ml/trial_stats.py", "data/eval/trial_log.jsonl"],
            datos={
                "sharpe_anual": sr_ann,
                "anios": years,
                "n_obs_aprox": n_obs,
                "n_trials": trials,
                "sr0_anual": float(sr0_ann),
                "dsr": float(dsr),
                "umbral_evidencia": 0.95,
                "lectura": (
                    "evidencia" if dsr > 0.95 else "sin_evidencia_suficiente"
                ),
            },
        )

    def plan_experimento(
        self,
        hipotesis: str,
        familia: str,
        config: dict[str, Any],
        criterio_exito: str,
    ) -> dict[str, Any]:
        if not all(
            isinstance(x, str) and x.strip()
            for x in [hipotesis, familia, criterio_exito]
        ):
            return self._error(
                "argumento_invalido",
                "hipotesis, familia y criterio_exito deben ser texto no vacío.",
            )
        if not isinstance(config, dict):
            return self._error("argumento_invalido", "config debe ser un objeto.")

        h = trial_stats.config_hash(config)
        rows = self._trial_rows()
        duplicate = any(
            r.get("family") == familia and r.get("hash") == h for r in rows
        )
        n_family = trial_stats.n_trials(familia, self.trial_log_path)
        n_global = self._n_global()
        return self._base(
            fuentes=["src/ml/trial_stats.py", "data/eval/trial_log.jsonl"],
            datos={
                "preregistro": {
                    "hipotesis": hipotesis.strip(),
                    "familia": familia.strip(),
                    "config": config,
                    "config_hash": h,
                    "criterio_exito": criterio_exito.strip(),
                },
                "ya_existe": duplicate,
                "n_familia_actual": n_family,
                "n_familia_despues": n_family + (0 if duplicate else 1),
                "n_global_actual": n_global,
                "n_global_despues_estimado": n_global + (0 if duplicate else 1),
                "nota": "Borrador de preregistro: no se escribió ni ejecutó nada.",
            },
        )

    def pronostico_precio(
        self,
        ticker: str,
        horizonte_dias: int = 42,
    ) -> dict[str, Any]:
        """Diagnóstico experimental; no es una recomendación ni señal operativa."""
        ticker = str(ticker or "").strip().upper()
        try:
            horizonte_dias = int(horizonte_dias)
        except (TypeError, ValueError):
            return self._error(
                "argumento_invalido",
                "horizonte_dias debe ser entero.",
            )
        if not ticker:
            return self._error("argumento_invalido", "ticker es obligatorio.")

        from scripts.forecast_horizons import HORIZONS, TICKERS, run_sin_ancla

        if ticker not in TICKERS:
            return self._error(
                "argumento_invalido",
                f"{ticker} no está en el catálogo local de símbolos.",
                fuentes=["scripts/forecast_horizons.py", "src/data/store.py"],
                evidencia="sin_skill",
            )

        horizon_label, horizon_days = min(
            HORIZONS.items(),
            key=lambda kv: abs(kv[1] - horizonte_dias),
        )
        model_names = (
            "lightgbm_ranking",
            "momentum_12_1",
            "woe_score",
            "regimen_hmm",
        )
        results: list[dict[str, Any]] = []
        warnings: list[str] = []
        cutoffs: list[str] = []

        for model in model_names:
            try:
                run = run_sin_ancla(ticker, model, horizon_label, horizon_days)
            except Exception as exc:
                warnings.append(f"{model}: error ({exc}).")
                continue
            if run is None or run.get("daily") is None or len(run["daily"]) == 0:
                warnings.append(f"{model}: datos insuficientes.")
                continue
            price0 = float(run["price0"])
            final_price = float(run["daily"]["price_model"].iloc[-1])
            ret = (final_price / price0 - 1.0) * 100.0
            cutoffs.append(str(run["cutoff_date"]))
            results.append(
                {
                    "modelo": model,
                    "retorno_pct": ret,
                    "mae_modelo": float(run["mae_model"]),
                    "mae_naive": float(run["mae_naive"]),
                    "etiqueta": "diagnostico_experimental",
                }
            )

        if not results:
            return self._error(
                "archivo_ausente",
                f"Ningún diagnóstico pudo ejecutarse para {ticker}.",
                fuentes=["scripts/forecast_horizons.py"],
                evidencia="sin_skill",
                advertencias=warnings,
            )

        returns = [r["retorno_pct"] for r in results]
        return self._base(
            as_of=max(cutoffs) if cutoffs else None,
            fuentes=["scripts/forecast_horizons.py"],
            evidencia="sin_skill",
            advertencias=[
                "Diagnóstico experimental: no es pronóstico operativo ni orden de compra/venta.",
                *warnings,
            ],
            datos={
                "ticker": ticker,
                "horizonte": horizon_label,
                "horizonte_sesiones": horizon_days,
                "modelos": results,
                "retorno_promedio_pct": float(np.mean(returns)),
                "dispersion_pp": float(max(returns) - min(returns)),
            },
        )

    def recomendar_portafolio(self) -> dict[str, Any]:
        path = (
            self.root
            / "data"
            / "eval"
            / "forecast_demo"
            / "portfolio_optimo_cov.json"
        )
        rel = "data/eval/forecast_demo/portfolio_optimo_cov.json"
        if not path.exists():
            return self._error(
                "archivo_ausente",
                f"No existe {rel}.",
                fuentes=[rel],
                evidencia="demo_no_vigente",
            )
        try:
            data = self._read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            return self._error(
                "archivo_ausente",
                f"No se pudo leer {rel}: {exc}",
                fuentes=[rel],
                evidencia="demo_no_vigente",
            )
        return self._base(
            as_of=datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat(),
            fuentes=[rel],
            evidencia="demo_no_vigente",
            advertencias=[
                "Demo histórica no vigente. La decisión actual del libro pertenece al dominio Lab.",
                "No debe interpretarse como recomendación de inversión.",
            ],
            datos={
                "pesos": data.get("pesos") or {},
                "lambda": data.get("lambda"),
                "retorno_portafolio_pct": data.get("retorno_portafolio_pct"),
                "riesgo_portafolio_pct": data.get("riesgo_portafolio_pct"),
            },
        )
