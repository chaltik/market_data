#!/usr/bin/env python3
"""Remove post-lifecycle price rows registered in equity_lifecycle_events.yaml.

Use --apply only after reviewing the printed row counts. This is deliberately
generic: it applies verified registry cutoffs without encoding strategy logic.
"""

from __future__ import annotations

import argparse

from data_retrieval import equity_lifecycle_events
from db_config import config
from db_utils import get_connection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="delete rows instead of reporting them")
    args = parser.parse_args()
    events = equity_lifecycle_events()
    if not events:
        print("No lifecycle events registered.")
        return
    with get_connection(config()) as (_, cur):
        for symbol, event in sorted(events.items()):
            cutoff = event["last_trading_date"]
            cur.execute(
                "SELECT count(*) AS n, min(ts::date) AS first_day, max(ts::date) AS last_day "
                "FROM price_data.equities_us_daily WHERE symbol=%s AND ts::date>%s",
                (symbol, cutoff),
            )
            row = cur.fetchone()
            print(f"{symbol}: {row['n']} rows after {cutoff} ({row['first_day']}..{row['last_day']})")
            if event.get("successor_symbol"):
                print(f"  successor database symbol: {event['successor_symbol']}")
            if args.apply and row["n"]:
                cur.execute(
                    "DELETE FROM price_data.equities_us_daily WHERE symbol=%s AND ts::date>%s",
                    (symbol, cutoff),
                )
                print(f"{symbol}: deleted {cur.rowcount} post-lifecycle rows")


if __name__ == "__main__":
    main()
