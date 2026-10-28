"""One-time, guarded cleanup of the legacy HY OAS ALFRED alias.

Dry-run by default. --apply preserves the affected rows in
data_repair_backup before changing eco; it never changes canonical values.
"""

import argparse
from datetime import date
from decimal import Decimal

import psycopg2

from db_config import config
from hy_oas import RELEASE_NAME, SERIES_ID, SERIES_NAME, VINTAGE_SOURCE

LEGACY_NAME = SERIES_ID
OBS_DATE = date(2026, 10, 26)
NEXT_DATE = date(2026, 10, 27)
CORRECTED_DATE = date(2026, 10, 28)
BACKUP_SCHEMA = "data_repair_backup"
BACKUP_TABLES = (
    "hy_oas_legacy_release_data_20261028",
    "hy_oas_legacy_release_content_20261028",
    "hy_oas_schedule_20261026_before_repair",
)


def preflight(cur):
    cur.execute("""
        SELECT count(*),
               count(*) FILTER (WHERE c.reference_date IS NOT NULL),
               count(*) FILTER (WHERE c.reference_date IS NULL AND l.value IS NULL),
               count(*) FILTER (WHERE c.reference_date IS NULL AND l.value IS NOT NULL),
               count(*) FILTER (WHERE c.reference_date IS NOT NULL
                                AND l.value IS DISTINCT FROM c.value)
        FROM eco.release_data l
        LEFT JOIN eco.release_data c
          ON c.release_name = l.release_name
         AND c.series_name = %s AND c.reference_date = l.reference_date
        WHERE l.release_name = %s AND l.series_name = %s
    """, (SERIES_NAME, RELEASE_NAME, LEGACY_NAME))
    counts = cur.fetchone()
    if counts != (792, 784, 8, 0, 0):
        raise RuntimeError(f"Legacy HY OAS rows changed; refusing repair: {counts}")

    cur.execute("""
        SELECT series_name, fred_series_id FROM eco.release_content
        WHERE release_name = %s AND fred_series_id = %s
        ORDER BY series_name
    """, (RELEASE_NAME, SERIES_ID))
    metadata = cur.fetchall()
    if metadata != [(LEGACY_NAME, SERIES_ID), (SERIES_NAME, SERIES_ID)]:
        raise RuntimeError(f"HY OAS metadata changed; refusing repair: {metadata}")

    cur.execute("""
        SELECT release_id, release_date, value
        FROM eco.release_data
        WHERE release_name = %s AND series_name = %s AND reference_date = %s
    """, (RELEASE_NAME, SERIES_NAME, OBS_DATE))
    canonical = cur.fetchone()
    if canonical != (20261028, CORRECTED_DATE, Decimal("3.30")):
        raise RuntimeError(f"Canonical October 26 row changed: {canonical}")

    cur.execute("""
        SELECT reference_date, release_id, release_date, release_date_source,
               release_date_verified, first_fetched_at
        FROM eco.release_schedule
        WHERE release_name = %s AND reference_date IN (%s, %s)
        ORDER BY reference_date
    """, (RELEASE_NAME, OBS_DATE, NEXT_DATE))
    schedule = cur.fetchall()
    if (len(schedule) != 2 or schedule[0][:5] !=
            (OBS_DATE, 20261026, OBS_DATE, "FRED/ALFRED vintage date", True)
            or schedule[0][5] is not None
            or schedule[1][:3] != (NEXT_DATE, 20261028, CORRECTED_DATE)
            or schedule[1][5] is None):
        raise RuntimeError(f"HY OAS schedule changed; refusing repair: {schedule}")
    return counts, schedule[1][5]


def repair(cur, first_fetched_at):
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {BACKUP_SCHEMA}")
    for table in BACKUP_TABLES:
        cur.execute("SELECT to_regclass(%s)", (f"{BACKUP_SCHEMA}.{table}",))
        if cur.fetchone()[0] is not None:
            raise RuntimeError(f"Backup table already exists: {BACKUP_SCHEMA}.{table}")
    cur.execute(f"""
        CREATE TABLE {BACKUP_SCHEMA}.{BACKUP_TABLES[0]} AS
        SELECT * FROM eco.release_data
        WHERE release_name = %s AND series_name = %s
    """, (RELEASE_NAME, LEGACY_NAME))
    cur.execute(f"""
        CREATE TABLE {BACKUP_SCHEMA}.{BACKUP_TABLES[1]} AS
        SELECT * FROM eco.release_content
        WHERE release_name = %s AND series_name = %s
    """, (RELEASE_NAME, LEGACY_NAME))
    cur.execute(f"""
        CREATE TABLE {BACKUP_SCHEMA}.{BACKUP_TABLES[2]} AS
        SELECT * FROM eco.release_schedule
        WHERE release_name = %s AND reference_date = %s
    """, (RELEASE_NAME, OBS_DATE))

    cur.execute("""
        DELETE FROM eco.release_data
        WHERE release_name = %s AND series_name = %s
    """, (RELEASE_NAME, LEGACY_NAME))
    if cur.rowcount != 792:
        raise RuntimeError(f"Expected 792 legacy rows deleted, got {cur.rowcount}")
    cur.execute("""
        DELETE FROM eco.release_content
        WHERE release_name = %s AND series_name = %s
    """, (RELEASE_NAME, LEGACY_NAME))
    if cur.rowcount != 1:
        raise RuntimeError(f"Expected one legacy metadata row deleted, got {cur.rowcount}")
    cur.execute("""
        UPDATE eco.release_schedule
        SET release_id = 20261028,
            release_date = %s,
            release_date_source = %s,
            release_date_verified = FALSE,
            first_fetched_at = %s
        WHERE release_name = %s AND reference_date = %s
    """, (CORRECTED_DATE, VINTAGE_SOURCE, first_fetched_at,
          RELEASE_NAME, OBS_DATE))
    if cur.rowcount != 1:
        raise RuntimeError(f"Expected one schedule row corrected, got {cur.rowcount}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--db-config", default="database.ini")
    args = parser.parse_args()
    with psycopg2.connect(**config(args.db_config)) as conn:
        conn.set_session(readonly=not args.apply)
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '5s'")
            cur.execute("SET LOCAL statement_timeout = '30s'")
            counts, first_fetched_at = preflight(cur)
            print(f"Preflight passed: {counts[0]} legacy rows, {counts[1]} matching "
                  f"canonical values, {counts[2]} null-only dates")
            if args.apply:
                repair(cur, first_fetched_at)
                print(f"Repaired HY OAS; backup tables are in {BACKUP_SCHEMA}")
            else:
                print("Dry run only; pass --apply for the scoped cleanup")


if __name__ == "__main__":
    main()
