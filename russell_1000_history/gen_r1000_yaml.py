"""Generate a price-retrieval asset list from the newest dated R1000 snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

from db_config import config as conn_params
from db_utils import get_connection

MIN_SYMBOLS = 900
MAX_SYMBOLS = 1150


def fetch_symbols(code: str, as_of: date) -> tuple[date, list[str]]:
    """Return symbols from the newest source snapshot no later than ``as_of``."""
    with get_connection(conn_params()) as (_, cur):
        cur.execute(
            """SELECT MAX(s.as_of_date) AS snapshot_date
               FROM market_indices.seed_snapshots s
               JOIN market_indices.indices i USING (index_id)
               WHERE i.code = %s AND s.as_of_date <= %s""",
            (code, as_of),
        )
        snapshot_date = cur.fetchone()["snapshot_date"]
        if snapshot_date is None:
            raise RuntimeError(f"No {code} snapshot on or before {as_of}")
        cur.execute(
            """SELECT s.symbol FROM market_indices.seed_snapshots s
               JOIN market_indices.indices i USING (index_id)
               WHERE i.code = %s AND s.as_of_date = %s ORDER BY s.symbol""",
            (code, snapshot_date),
        )
        symbols = [row["symbol"] for row in cur.fetchall()]
    if not MIN_SYMBOLS <= len(symbols) <= MAX_SYMBOLS:
        raise RuntimeError(f"Suspicious {code} snapshot size {len(symbols)} on {snapshot_date}")
    return snapshot_date, symbols


def render_assets_yaml(symbols: list[str]) -> str:
    """Render a portable asset-list YAML document without requiring PyYAML."""
    return "\n".join(["equities:", *[f"  - {json.dumps(symbol)}" for symbol in symbols], "", "crypto: []", ""])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", default="R1000")
    parser.add_argument("--asof", type=date.fromisoformat, default=date.today())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    snapshot_date, symbols = fetch_symbols(args.code, args.asof)
    if args.asof - snapshot_date > timedelta(days=7):
        raise RuntimeError(f"Stale {args.code} snapshot {snapshot_date} for {args.asof}")
    content = render_assets_yaml(symbols)
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    else:
        sys.stdout.write(content)


if __name__ == "__main__":
    main()
