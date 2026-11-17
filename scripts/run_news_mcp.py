#!/usr/bin/env python3
"""Punto de entrada para iniciar el servidor MCP de Noticias Segmentadas.

Uso:
    python scripts/run_news_mcp.py [--transport stdio] [--news-root RUTA]
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mcp.server import main

if __name__ == "__main__":
    main()

