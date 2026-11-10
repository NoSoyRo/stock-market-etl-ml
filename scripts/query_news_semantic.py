#!/usr/bin/env python
"""CLI JSON de solo lectura para la capa semántica de noticias."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if importlib.util.find_spec("pandas") is None:
    site_packages = (
        ROOT
        / "venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if site_packages.is_dir():
        sys.path.insert(0, str(site_packages))

from src.semantic.models import dumps
from src.semantic.repository import DEFAULT_NEWS_ROOT
from src.semantic.service import NewsSemanticService


def _entity_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker")
    group.add_argument("--topic")


def _entity(args: argparse.Namespace) -> tuple[str, str]:
    if args.ticker is not None:
        return "ticker", args.ticker
    return "topic", args.topic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-root", type=Path, default=DEFAULT_NEWS_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    company = subparsers.add_parser("empresa")
    company.add_argument("--ticker", required=True)
    company.add_argument("--fecha")
    company.add_argument("--window-days", type=int)
    company.add_argument("--corte", help="Corte ISO-8601 con zona horaria.")

    topic = subparsers.add_parser("tema")
    topic.add_argument("--topic", required=True)
    topic.add_argument("--fecha")
    topic.add_argument("--window-days", type=int)
    topic.add_argument("--corte", help="Corte ISO-8601 con zona horaria.")

    history = subparsers.add_parser("historial")
    _entity_arguments(history)
    history.add_argument("--inicio", required=True)
    history.add_argument("--fin", required=True)
    history.add_argument("--window-days", type=int)

    compare = subparsers.add_parser("comparar")
    _entity_arguments(compare)
    compare.add_argument("--fecha-a", required=True)
    compare.add_argument("--fecha-b", required=True)

    evidence = subparsers.add_parser("evidencia")
    _entity_arguments(evidence)
    evidence.add_argument("--fecha", required=True)
    evidence.add_argument("--periodo", choices=["reciente", "anterior", "ambos"], default="reciente")
    evidence.add_argument("--limit", type=int, default=20)
    evidence.add_argument("--offset", type=int, default=0)
    evidence.add_argument("--corte", help="Corte ISO-8601 con zona horaria.")

    subparsers.add_parser("cobertura")
    return parser


def _exit_code(status: str) -> int:
    if status == "invalid_input":
        return 2
    if status in {"data_unavailable", "data_integrity_error"}:
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    service = NewsSemanticService(args.news_root)
    if args.command == "empresa":
        result = service.consultar_empresa(
            args.ticker, fecha=args.fecha, window_days=args.window_days, corte=args.corte
        )
    elif args.command == "tema":
        result = service.consultar_tema(
            args.topic, fecha=args.fecha, window_days=args.window_days, corte=args.corte
        )
    elif args.command == "historial":
        entity_type, entity_id = _entity(args)
        result = service.consultar_historial(
            entity_type,
            entity_id,
            args.inicio,
            args.fin,
            window_days=args.window_days,
        )
    elif args.command == "comparar":
        entity_type, entity_id = _entity(args)
        result = service.comparar_fechas(
            entity_type, entity_id, args.fecha_a, args.fecha_b
        )
    elif args.command == "evidencia":
        entity_type, entity_id = _entity(args)
        result = service.consultar_evidencia(
            entity_type,
            entity_id,
            args.fecha,
            periodo=args.periodo,
            limit=args.limit,
            offset=args.offset,
            corte=args.corte,
        )
    else:
        result = service.consultar_cobertura()
    print(dumps(result))
    return _exit_code(result["status"])


if __name__ == "__main__":
    raise SystemExit(main())
