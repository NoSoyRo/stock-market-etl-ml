#!/usr/bin/env python3
"""Run the local news dashboard and its server-side AI agent."""
from pathlib import Path
import argparse
import logging
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')
from src.dashboard.news import create_app
from src.semantic.repository import DEFAULT_NEWS_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--news-root', type=Path, default=DEFAULT_NEWS_ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    create_app(news_root=args.news_root).run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
