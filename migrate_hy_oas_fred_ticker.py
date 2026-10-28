"""One-off, guarded HY OAS key migration. Dry-run unless --apply is given.

Keep the release name, dates, and values unchanged; only rename series_name
from the display label to FRED's exact series ID. Backups are transactional.
"""

import argparse

import psycopg2

from db_config import config
from hy_oas import RELEASE_NAME, SERIES_ID

OLD_NAME = "HY OAS"
EXPECTED_ROWS = 7788


def preflight(cur):
    cur.execute(
        """SELECT series_name, fred_series_id FROM eco.release_content
           WHERE release_name = %s ORDER BY series_name""", (RELEASE_NAME,)
    )
    metadata = cur.fetchall()
    if metadata != [(OLD_NAME, SERIES_ID)]:
        raise RuntimeError(f"Unexpected HY OAS metadata: {metadata}")
    cur.execute(
        """SELECT series_name, count(*) FROM eco.release_data
           WHERE release_name = %s GROUP BY series_name""", (RELEASE_NAME,)
    )
    counts = cur.fetchall()
    if counts != [(OLD_NAME, EXPECTED_ROWS)]:
        raise RuntimeError(f"Unexpected HY OAS data counts: {counts}")
    return counts[0][1]


def migrate(cur):
    cur.execute("LOCK TABLE eco.release_content, eco.release_data IN SHARE ROW EXCLUSIVE MODE")
    count = preflight(cur)
    cur.execute("CREATE SCHEMA IF NOT EXISTS data_repair_backup")
    cur.execute(
        """CREATE TABLE data_repair_backup.hy_oas_content_before_ticker_20261028 AS
           SELECT * FROM eco.release_content WHERE release_name = %s
             AND series_name = %s""", (RELEASE_NAME, OLD_NAME)
    )
    cur.execute(
        """CREATE TABLE data_repair_backup.hy_oas_data_before_ticker_20261028 AS
           SELECT * FROM eco.release_data WHERE release_name = %s
             AND series_name = %s""", (RELEASE_NAME, OLD_NAME)
    )
    cur.execute(
        """INSERT INTO eco.release_content
           (release_id, release_name, series_name, bls_series_id,
            fred_series_id, source, country_code)
           SELECT release_id, release_name, %s, bls_series_id,
                  fred_series_id, source, country_code
           FROM eco.release_content
           WHERE release_name = %s AND series_name = %s""",
        (SERIES_ID, RELEASE_NAME, OLD_NAME),
    )
    if cur.rowcount != 1:
        raise RuntimeError("Ticker metadata insert did not affect exactly one row")
    cur.execute(
        """UPDATE eco.release_data SET series_name = %s
           WHERE release_name = %s AND series_name = %s""",
        (SERIES_ID, RELEASE_NAME, OLD_NAME),
    )
    if cur.rowcount != count:
        raise RuntimeError(f"Expected {count} moved rows, got {cur.rowcount}")
    cur.execute(
        """DELETE FROM eco.release_content
           WHERE release_name = %s AND series_name = %s""",
        (RELEASE_NAME, OLD_NAME),
    )
    if cur.rowcount != 1:
        raise RuntimeError("Legacy metadata removal did not affect exactly one row")
    cur.execute(
        """SELECT count(*) FROM eco.release_data
           WHERE release_name = %s AND series_name = %s""",
        (RELEASE_NAME, SERIES_ID),
    )
    if cur.fetchone()[0] != count:
        raise RuntimeError("Migrated row count failed final verification")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--db-config", default="database.ini")
    args = parser.parse_args()
    with psycopg2.connect(**config(args.db_config)) as connection:
        with connection.cursor() as cur:
            if args.apply:
                count = migrate(cur)
                print(f"Migrated {count} HY OAS rows to {SERIES_ID}; backups retained")
            else:
                count = preflight(cur)
                print(f"Dry run: {count} HY OAS rows ready to move to {SERIES_ID}")


if __name__ == "__main__":
    main()
