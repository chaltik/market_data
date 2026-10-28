import tempfile
import unittest
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd

os.environ.setdefault("TIINGO_API_KEY", "test-only")
from data_retrieval import save_eco_from_fred_to_db
from hy_oas import (ARCHIVE_SOURCE, FRED_SOURCE, VINTAGE_SOURCE,
                    latest_observation, next_nyse_session, read_archive, store)


class RecordingCursor:
    def __init__(self):
        self.statements = []
        self.rowcount = 1

    def execute(self, sql, params=None):
        self.statements.append((sql, params))

    def fetchall(self):
        return []

    def fetchone(self):
        return (date(2026, 10, 27),)


class HyOasTest(unittest.TestCase):
    def test_latest_observation_ignores_legacy_alias(self):
        cursor = RecordingCursor()
        self.assertEqual(latest_observation(cursor), date(2026, 10, 27))
        sql, params = cursor.statements[-1]
        self.assertIn("d.series_name = %s", sql)
        self.assertEqual(params, ("BAMLH0A0HYM2", "ICE BofA US High Yield OAS", "HY OAS"))

    def test_generic_writer_refuses_hy_oas_alias(self):
        frame = pd.DataFrame([{
            "fred_series_id": "BAMLH0A0HYM2", "series_name": "BAMLH0A0HYM2",
            "release_name": "ICE BofA US High Yield OAS", "release_id": 20261028,
            "release_date": date(2026, 10, 28),
            "reference_date": date(2026, 10, 27), "value": Decimal("3.18"),
        }])
        with self.assertRaisesRegex(ValueError, "must be loaded with hy_oas.py"):
            save_eco_from_fred_to_db(frame)

    def test_archive_skips_blanks_and_preserves_numeric_weekend_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.csv"
            path.write_text(
                "date,BAMLH0A0HYM2\n"
                "2026-01-30,2.9\n"
                "2026-01-31,2.88\n"
                "2026-02-01,\n"
                "2026-02-02,.\n"
            )
            observations = read_archive(path)
        self.assertEqual([day for day, _ in observations], [
            date(2026, 1, 30), date(2026, 1, 31)
        ])
        self.assertEqual(next_nyse_session(date(2026, 1, 31)), date(2026, 2, 2))

    def test_archive_and_fresh_fetch_have_distinct_availability(self):
        observations = [(date(2026, 1, 30), Decimal("2.9"))]
        fetched_at = datetime(2026, 2, 10, 1, 0, tzinfo=timezone.utc)
        archive = RecordingCursor()
        store(archive, observations, source=ARCHIVE_SOURCE, fetched_at=fetched_at)
        fresh = RecordingCursor()
        store(fresh, observations, source=FRED_SOURCE, fetched_at=fetched_at)

        def release_date(cursor):
            return next(params[3] for sql, params in cursor.statements
                        if "INSERT INTO eco.release_data" in sql)

        self.assertEqual(release_date(archive), date(2026, 2, 2))
        # Fetched after the US session on February 9; no intraday lookahead.
        self.assertEqual(release_date(fresh), date(2026, 2, 10))

        vintage = RecordingCursor()
        store(
            vintage,
            [(date(2026, 1, 30), Decimal("2.9"), date(2026, 1, 30))],
            source=FRED_SOURCE,
            fetched_at=datetime(2026, 3, 10, 1, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(release_date(vintage), date(2026, 2, 2))
        self.assertTrue(any(
            params and VINTAGE_SOURCE in params
            for _, params in vintage.statements
        ))

    def test_recent_vintage_fetched_after_cutoff_waits_for_next_session(self):
        fetched_at = datetime(2026, 10, 24, 3, 0, tzinfo=timezone.utc)
        cursor = RecordingCursor()
        store(
            cursor,
            [(date(2026, 10, 22), Decimal("3.4"), date(2026, 10, 22))],
            source=FRED_SOURCE,
            fetched_at=fetched_at,
        )
        row = next(params for sql, params in cursor.statements
                   if "INSERT INTO eco.release_data" in sql)
        self.assertEqual(row[3], date(2026, 10, 26))


if __name__ == "__main__":
    unittest.main()
