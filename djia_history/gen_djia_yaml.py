"""Export current DJIA symbols for the shared daily price loader."""

import argparse
from datetime import date

import psycopg2

from db_config import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asof", default=date.today().isoformat())
    parser.add_argument("--output", required=True)
    parser.add_argument("--db-config", help="Path to database.ini, if not in repository root")
    args = parser.parse_args()
    params = config(filename=args.db_config) if args.db_config else config()
    with psycopg2.connect(**params) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT symbol FROM market_indices.constituents_asof('DJIA', %s) ORDER BY symbol", (args.asof,))
            symbols = [row[0] for row in cur.fetchall()]
    if len(symbols) != 30 or len(set(symbols)) != 30:
        raise ValueError(f"Expected 30 distinct DJIA symbols as of {args.asof}; got {len(symbols)}")
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("equities:\n" + "".join(f"  - {symbol}\n" for symbol in symbols) + "crypto: []\n")


if __name__ == "__main__":
    main()
