"""Backfill and refresh FRED's US high-yield option-adjusted spread.

The archive contains observation dates, not publication timestamps. Its
release_date is therefore an explicitly inferred *availability date*: the next
NYSE session after the observation. Recent FRED observations are also bounded
by the actual fetch time relative to the 11:30 ET decision cutoff. Consumers
making that intraday decision may use release_date <= their trade date.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from bisect import bisect_right
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal
import psycopg2
import requests
from dotenv import load_dotenv

from db_config import config

SERIES_ID = "BAMLH0A0HYM2"
RELEASE_NAME = "ICE BofA US High Yield OAS"
SERIES_NAME = "HY OAS"
ARCHIVE_SOURCE = "inferred_next_nyse_session_from_archive"
FRED_SOURCE = "inferred_next_nyse_session_after_fred_fetch"
VINTAGE_SOURCE = "inferred_next_nyse_session_after_alfred_vintage"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
NY_TZ = ZoneInfo("America/New_York")
DECISION_CUTOFF_ET = time(11, 30)
LOG = logging.getLogger(__name__)


def next_nyse_session(day: date) -> date:
    """Return the first NYSE trading date strictly after day."""
    calendar = mcal.get_calendar("NYSE")
    days = calendar.valid_days(
        start_date=day + timedelta(days=1), end_date=day + timedelta(days=14)
    )
    if days.empty:
        raise ValueError(f"No NYSE session found after {day}")
    return days[0].date()


def parse_value(raw: str | None) -> Decimal | None:
    if raw is None or not raw.strip() or raw.strip() == ".":
        return None
    try:
        value = Decimal(raw.strip())
    except InvalidOperation as exc:
        raise ValueError(f"Invalid {SERIES_ID} value: {raw!r}") from exc
    if not value.is_finite():
        raise ValueError(f"Invalid {SERIES_ID} value: {raw!r}")
    return value


def read_archive(path: Path) -> list[tuple[date, Decimal]]:
    observations: dict[date, Decimal] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["date", SERIES_ID]:
            raise ValueError(f"Unexpected archive columns: {reader.fieldnames}")
        for row in reader:
            observation_date = date.fromisoformat(row["date"])
            value = parse_value(row[SERIES_ID])
            if value is None:
                continue
            if observation_date in observations:
                raise ValueError(f"Duplicate observation: {observation_date}")
            observations[observation_date] = value
    return sorted(observations.items())


def fetch_fred(start: date, api_key: str) -> list[tuple[date, Decimal, date | None]]:
    observations: dict[date, tuple[Decimal, date | None]] = {}
    offset = 0
    use_vintages = True
    while True:
        response = requests.get(
            FRED_URL,
            params={
                "series_id": SERIES_ID,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start.isoformat(),
                **({
                    "realtime_start": "1776-07-04",
                    "realtime_end": "9999-12-31",
                } if use_vintages else {}),
                "limit": 100000,
                "offset": offset,
            },
            timeout=60,
        )
        if response.status_code == 400 and use_vintages:
            LOG.warning("ALFRED vintages unavailable; using FRED observations")
            use_vintages = False
            offset = 0
            observations.clear()
            continue
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("observations", [])
        for row in batch:
            value = parse_value(row.get("value"))
            if value is not None:
                observation_date = date.fromisoformat(row["date"])
                vintage_date = (
                    date.fromisoformat(row["realtime_start"]) if use_vintages else None
                )
                previous = observations.get(observation_date)
                if previous is None or (
                    vintage_date is not None and previous[1] is not None
                    and vintage_date < previous[1]
                ):
                    # Keep the first published value, not a later revision.
                    observations[observation_date] = (value, vintage_date)
        offset += len(batch)
        if not batch or offset >= int(payload.get("count", offset)):
            break
    return [(day, value, vintage) for day, (value, vintage)
            in sorted(observations.items())]


def ensure_provenance_columns(cur) -> None:
    # Production's eco schema predates this change; the migration is safe to
    # run repeatedly and does not alter existing release dates.
    cur.execute(
        "ALTER TABLE eco.release_schedule "
        "ADD COLUMN IF NOT EXISTS release_date_source TEXT"
    )
    cur.execute(
        "ALTER TABLE eco.release_schedule "
        "ADD COLUMN IF NOT EXISTS release_date_verified BOOLEAN NOT NULL DEFAULT FALSE"
    )
    cur.execute(
        "ALTER TABLE eco.release_schedule "
        "ADD COLUMN IF NOT EXISTS first_fetched_at TIMESTAMPTZ"
    )


def latest_observation(cur) -> date | None:
    cur.execute(
        """
        SELECT max(d.reference_date)
        FROM eco.release_data d
        JOIN eco.release_content c USING (release_name, series_name)
        WHERE c.fred_series_id = %s
          AND d.release_name = %s AND d.series_name = %s
          AND d.value IS NOT NULL
        """,
        (SERIES_ID, RELEASE_NAME, SERIES_NAME),
    )
    return cur.fetchone()[0]


def store(
    cur,
    observations: list[tuple[date, Decimal] | tuple[date, Decimal, date | None]],
    *,
    source: str,
    fetched_at: datetime,
) -> int:
    if not observations:
        return 0
    ensure_provenance_columns(cur)
    cur.execute(
        """
        INSERT INTO eco.release_content
            (release_id, release_name, series_name, fred_series_id, source, country_code)
        VALUES (%s, %s, %s, %s, %s, 'USA')
        ON CONFLICT (release_name, series_name) DO UPDATE
        SET fred_series_id = EXCLUDED.fred_series_id,
            source = EXCLUDED.source
        """,
        (int(observations[0][0].strftime("%Y%m%d")), RELEASE_NAME, SERIES_NAME,
         SERIES_ID, "FRED archive / FRED API"),
    )
    fetch_day = fetched_at.astimezone(NY_TZ).date()
    calendar = mcal.get_calendar("NYSE")
    sessions = [
        timestamp.date()
        for timestamp in calendar.valid_days(
            start_date=min(observations[0][0], fetch_day),
            end_date=max(observations[-1][0], fetch_day) + timedelta(days=14),
        )
    ]

    def following_session(day: date) -> date:
        position = bisect_right(sessions, day)
        if position >= len(sessions):
            raise ValueError(f"No NYSE session found after {day}")
        return sessions[position]

    def visible_after_fetch(when: datetime) -> date:
        eastern = when.astimezone(NY_TZ)
        day = eastern.date()
        if day in sessions and eastern.time() < DECISION_CUTOFF_ET:
            return day
        return following_session(day)

    existing_fetches = {}
    if source == FRED_SOURCE:
        cur.execute(
            """
            SELECT reference_date, first_fetched_at
            FROM eco.release_schedule
            WHERE release_name = %s
            """,
            (RELEASE_NAME,),
        )
        existing_fetches = dict(cur.fetchall())

    written = 0
    for observation in observations:
        observation_date, value = observation[:2]
        vintage_date = observation[2] if len(observation) == 3 else None
        available_on = following_session(observation_date)
        row_source = source
        if source == FRED_SOURCE:
            first_fetch = existing_fetches.get(observation_date) or fetched_at
            if vintage_date is not None:
                available_on = max(available_on, following_session(vintage_date))
                row_source = VINTAGE_SOURCE
                # A recent vintage date does not prove that the value was
                # present before today's 11:30 ET strategy decision.
                if vintage_date >= fetch_day - timedelta(days=14):
                    available_on = max(available_on, visible_after_fetch(first_fetch))
            else:
                available_on = max(available_on, visible_after_fetch(first_fetch))
        release_id = int(available_on.strftime("%Y%m%d"))
        cur.execute(
            """
            INSERT INTO eco.release_schedule
                (release_id, release_name, release_date,
                 reference_date, release_date_source, release_date_verified,
                 first_fetched_at)
            VALUES (%s, %s, %s, %s, %s, FALSE, %s)
            ON CONFLICT (release_name, reference_date) DO UPDATE
            SET release_date = EXCLUDED.release_date,
                release_id = EXCLUDED.release_id,
                release_date_source = EXCLUDED.release_date_source
            WHERE eco.release_schedule.release_date_source = %s
              AND EXCLUDED.release_date_source = %s
               OR (eco.release_schedule.release_date_source = %s
                   AND EXCLUDED.release_date_source = %s
                   AND eco.release_schedule.release_date < EXCLUDED.release_date)
            """,
            (release_id, RELEASE_NAME, available_on, observation_date,
             row_source, fetched_at, FRED_SOURCE, VINTAGE_SOURCE,
             VINTAGE_SOURCE, VINTAGE_SOURCE),
        )
        cur.execute(
            """
            INSERT INTO eco.release_data
                (release_id, release_name, series_name, release_date,
                 reference_date, value)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (release_name, series_name, reference_date) DO UPDATE
            SET release_date = EXCLUDED.release_date,
                release_id = EXCLUDED.release_id,
                value = EXCLUDED.value
            WHERE eco.release_data.release_date > EXCLUDED.release_date
               OR (eco.release_data.release_date < EXCLUDED.release_date
                   AND EXISTS (
                       SELECT 1 FROM eco.release_schedule s
                       WHERE s.release_name = EXCLUDED.release_name
                         AND s.reference_date = EXCLUDED.reference_date
                         AND s.release_date_source = %s
                         AND s.release_date = EXCLUDED.release_date
                   ))
            """,
            (release_id, RELEASE_NAME, SERIES_NAME, available_on,
             observation_date, value, VINTAGE_SOURCE),
        )
        written += cur.rowcount
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("backfill", "refresh"))
    parser.add_argument("--archive", type=Path, help="CSV archive for backfill")
    parser.add_argument("--since", type=date.fromisoformat,
                        help="Override incremental FRED start date (YYYY-MM-DD)")
    parser.add_argument("--db-config", default="database.ini")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name(".env"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv(args.env_file)

    with psycopg2.connect(**config(args.db_config)) as connection:
        with connection.cursor() as cur:
            if args.action == "backfill":
                if args.archive is None:
                    parser.error("--archive is required for backfill")
                observations = read_archive(args.archive)
                source = ARCHIVE_SOURCE
            else:
                api_key = os.environ.get("FRED_API_KEY")
                if not api_key:
                    parser.error("FRED_API_KEY is required for refresh")
                newest = latest_observation(cur)
                start = args.since or (
                    newest - timedelta(days=14) if newest else date(1996, 12, 31)
                )
                observations = fetch_fred(start, api_key)
                source = FRED_SOURCE
            # Capture when the complete provider response was actually received;
            # a request crossing 11:30 ET must not claim pre-decision access.
            fetched_at = datetime.now(timezone.utc)
            written = store(cur, observations, source=source, fetched_at=fetched_at)
            newest = latest_observation(cur)
    LOG.info(
        "%s: %s nonmissing observations fetched, %s inserted or corrected, newest stored %s",
        SERIES_ID, len(observations), written, newest,
    )


if __name__ == "__main__":
    main()
