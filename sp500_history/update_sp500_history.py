"""Daily SP500 approximate-universe maintenance from State Street SPY."""

import argparse
import logging
from datetime import date, timedelta

from db_config import config as conn_params
from db_utils import get_connection

from .build_sp500_history import (
    INDEX_CODE,
    MAX_UNIVERSE_ROWS,
    MIN_UNIVERSE_ROWS,
    SPY_HOLDINGS_XLSX,
    SPY_SOURCE_TITLE,
    Source,
    fetch_spy_holdings,
    upsert_source,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default=INDEX_CODE)
    parser.add_argument("--asof", help="Snapshot date YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    as_of = date.fromisoformat(args.asof) if args.asof else date.today()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    holdings = fetch_spy_holdings()
    if not MIN_UNIVERSE_ROWS <= len(holdings) <= MAX_UNIVERSE_ROWS:
        raise RuntimeError(f"Refusing SPY update with {len(holdings)} symbols")
    current = {holding.symbol: holding for holding in holdings}
    params = conn_params()

    with get_connection(params) as (conn, cur):
        cur.execute("SELECT index_id FROM market_indices.indices WHERE code = %(code)s", {"code": args.code})
        row = cur.fetchone()
        if not row:
            raise LookupError(f"Index code {args.code} not found; run the historical builder or DDL first")
        index_id = row["index_id"]

    with get_connection(params) as (conn, cur):
        cur.execute(
            """
            SELECT MAX(as_of_date) AS prev_asof
            FROM market_indices.seed_snapshots
            WHERE index_id = %(id)s AND as_of_date < %(asof)s
            """,
            {"id": index_id, "asof": as_of},
        )
        previous = cur.fetchone()["prev_asof"]
        previous_symbols = set()
        if previous:
            cur.execute(
                "SELECT symbol FROM market_indices.seed_snapshots WHERE index_id = %(id)s AND as_of_date = %(asof)s",
                {"id": index_id, "asof": previous},
            )
            previous_symbols = {row["symbol"] for row in cur.fetchall()}
        if previous_symbols and len(current) < int(0.6 * len(previous_symbols)):
            raise RuntimeError(f"Refusing suspicious SPY size drop: {len(previous_symbols)} -> {len(current)}")

    with get_connection(params) as (conn, cur):
        source_id = upsert_source(cur, Source("ETF", SPY_SOURCE_TITLE, SPY_HOLDINGS_XLSX, as_of, None))
        for holding in holdings:
            cur.execute(
                """
                INSERT INTO market_indices.seed_snapshots
                    (index_id, as_of_date, symbol, company_name, sector, weight, source_id)
                VALUES (%(id)s, %(asof)s, %(symbol)s, %(name)s, %(sector)s, %(weight)s, %(source)s)
                ON CONFLICT (index_id, as_of_date, symbol) DO UPDATE SET
                    company_name = EXCLUDED.company_name, sector = EXCLUDED.sector,
                    weight = EXCLUDED.weight, source_id = EXCLUDED.source_id
                """,
                {"id": index_id, "asof": as_of, "symbol": holding.symbol,
                 "name": holding.company_name, "sector": holding.sector,
                 "weight": holding.weight, "source": source_id},
            )

    current_symbols = set(current)
    if not previous:
        start_date = as_of
        added, removed, continued = current_symbols, set(), set()
    else:
        start_date = min(previous + timedelta(days=1), as_of)
        added = current_symbols - previous_symbols
        removed = previous_symbols - current_symbols
        continued = current_symbols & previous_symbols

    if removed:
        with get_connection(params) as (conn, cur):
            for symbol in sorted(removed):
                cur.execute(
                    """
                    UPDATE market_indices.index_constituents
                    SET end_date = %(end)s
                    WHERE index_id = %(id)s AND symbol = %(symbol)s AND end_date IS NULL
                    """,
                    {"id": index_id, "symbol": symbol, "end": start_date},
                )

    if added:
        with get_connection(params) as (conn, cur):
            for symbol in sorted(added):
                holding = current[symbol]
                cur.execute(
                    """
                    INSERT INTO market_indices.index_constituents
                        (index_id, symbol, company_name, sector, start_date, end_date, last_verified_asof)
                    VALUES (%(id)s, %(symbol)s, %(name)s, %(sector)s, %(start)s, NULL, %(asof)s)
                    ON CONFLICT ON CONSTRAINT uq_constituent_period DO NOTHING
                    """,
                    {"id": index_id, "symbol": symbol, "name": holding.company_name,
                     "sector": holding.sector, "start": start_date, "asof": as_of},
                )

    if continued:
        with get_connection(params) as (conn, cur):
            for symbol in sorted(continued):
                holding = current[symbol]
                cur.execute(
                    """
                    UPDATE market_indices.index_constituents
                    SET company_name = COALESCE(company_name, %(name)s),
                        sector = COALESCE(sector, %(sector)s),
                        last_verified_asof = GREATEST(COALESCE(last_verified_asof, %(asof)s), %(asof)s)
                    WHERE index_id = %(id)s AND symbol = %(symbol)s AND end_date IS NULL
                    """,
                    {"id": index_id, "symbol": symbol, "name": holding.company_name,
                     "sector": holding.sector, "asof": as_of},
                )

    logging.info("SP500 diff: added=%d removed=%d continued=%d", len(added), len(removed), len(continued))


if __name__ == "__main__":
    main()
