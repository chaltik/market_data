"""Generate a data_retrieval.py asset YAML for the SP500 universe."""

import argparse
import re
import sys
from datetime import date
from typing import List

from db_config import config as conn_params
from db_utils import get_connection

TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,7}$")


def _format_yaml(symbols: List[str]) -> str:
    equities = sorted({symbol for symbol in symbols if TICKER_RE.fullmatch(symbol)})
    return "\n".join(["equities:", *[f"  - {symbol}" for symbol in equities], "", "crypto: []", ""])


def fetch_symbols(code: str, mode: str, as_of: date) -> List[str]:
    with get_connection(conn_params()) as (conn, cur):
        if mode == "all":
            cur.execute(
                """
                SELECT symbol
                FROM market_indices.index_constituents ic
                JOIN market_indices.indices i USING (index_id)
                WHERE i.code = %(code)s
                GROUP BY symbol ORDER BY symbol
                """,
                {"code": code},
            )
        else:
            cur.execute(
                "SELECT symbol FROM market_indices.constituents_asof(%(code)s, %(asof)s) ORDER BY symbol",
                {"code": code, "asof": as_of},
            )
        return [row["symbol"].upper() for row in cur.fetchall()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="SP500")
    parser.add_argument("--mode", choices=["all", "asof"], default="asof")
    parser.add_argument("--asof", help="YYYY-MM-DD; defaults to today for asof mode")
    parser.add_argument("--output")
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    as_of = date.fromisoformat(args.asof) if args.asof else date.today()
    text = _format_yaml(fetch_symbols(args.code, args.mode, as_of))
    if args.stdout or not args.output:
        sys.stdout.write(text)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
