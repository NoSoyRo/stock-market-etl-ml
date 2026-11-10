"""Import existing news into the lake and build gold, without network calls."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.data.news_medallion import (
    DEFAULT_ROOT, build_gold, capture, normalize, rebuild_bronze, rebuild_gold_history,
)
from src.ml.news_sources import ENTITY_ALIASES


def migrate(root):
    """Content-addressed import manifest makes repeated legacy imports idempotent."""
    candidates = [(p, 'databursatil') for p in sorted((root / 'history' / 'databursatil').glob('????-??-??.json'))]
    candidates += [(root / 'noticias_trump.csv', 'trump'), (root / 'history' / 'articles.parquet', 'gdelt')]
    report = []
    for path, source in candidates:
        if not path.exists():
            continue
        digest = sha256(path.read_bytes()).hexdigest()
        marker = root / 'bronze' / '_imports' / (digest + '.json')
        if marker.exists():
            imported = json.loads(marker.read_text())
            normalize(root / imported['bronze_path'], root=root, tickers=ENTITY_ALIASES)
            continue
        if path.suffix == '.json':
            payload = json.loads(path.read_text())
            fetched = payload.get('fetched_at')  # Unknown legacy availability is assigned import time.
        else:
            df = pd.read_csv(path).fillna('') if path.suffix == '.csv' else pd.read_parquet(path)
            payload = df.to_dict(orient='records')
            fetched = None
        bronze = capture(source, payload, root=root, fetched_at=fetched,
                         representation='legacy_import', origin=str(path.relative_to(root)))
        normalize(bronze, root=root, tickers=ENTITY_ALIASES)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({'bronze_path': str(bronze.relative_to(root))}))
        report.append(str(path.relative_to(root)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--migrate', action='store_true', help='Import legacy snapshots, articles and Trump CSV')
    parser.add_argument('--as-of', help='UTC cutoff; default now')
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--replay', action='store_true', help='Rebuild silver from all bronze captures')
    parser.add_argument('--bronze-only', action='store_true',
                        help='Build bronze/news.parquet from JSON captures without rebuilding silver/gold')
    parser.add_argument('--rebuild-gold-history', action='store_true',
                        help='Recalculate historical daily cuts from the first known observation')
    args = parser.parse_args()
    if args.migrate:
        print('Imported:', migrate(args.root))
    if args.bronze_only:
        result = rebuild_bronze(root=args.root)
        print(f'Bronze: {len(result)} records → {args.root / "bronze/news.parquet"}')
        return
    if args.replay:
        for path in sorted((args.root / 'bronze').glob('*/*/*.json')):
            normalize(path, root=args.root, tickers=ENTITY_ALIASES)
    builder = rebuild_gold_history if args.rebuild_gold_history else build_gold
    result = builder(root=args.root, as_of=args.as_of, days_back=args.days)
    print(f'Gold: {len(result)} groups → {args.root / "gold"}')


if __name__ == '__main__':
    main()
