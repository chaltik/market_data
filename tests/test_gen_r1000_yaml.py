from contextlib import contextmanager
from datetime import date

import pytest

from russell_1000_history import gen_r1000_yaml as generator


def test_render_assets_yaml_quotes_symbols_and_has_required_sections():
    assert generator.render_assets_yaml(["AAPL", "BRK.B"]) == (
        'equities:\n  - "AAPL"\n  - "BRK.B"\n\ncrypto: []\n'
    )


def test_fetch_symbols_uses_latest_snapshot_not_later_than_requested_date(monkeypatch):
    class Cursor:
        def __init__(self):
            self.queries = []

        def execute(self, query, params):
            self.queries.append((query, params))

        def fetchone(self):
            return {"snapshot_date": date(2026, 11, 16)}

        def fetchall(self):
            return [{"symbol": f"S{i:04d}"} for i in range(generator.MIN_SYMBOLS)]

    cursor = Cursor()

    @contextmanager
    def connection(_):
        yield None, cursor

    monkeypatch.setattr(generator, "get_connection", connection)
    monkeypatch.setattr(generator, "conn_params", lambda: {})
    snapshot_date, symbols = generator.fetch_symbols("R1000", date(2026, 11, 17))
    assert snapshot_date == date(2026, 11, 16)
    assert len(symbols) == generator.MIN_SYMBOLS
    assert cursor.queries[0][1] == ("R1000", date(2026, 11, 17))


def test_fetch_symbols_rejects_an_implausibly_small_snapshot(monkeypatch):
    class Cursor:
        def execute(self, *_):
            pass

        def fetchone(self):
            return {"snapshot_date": date(2026, 11, 16)}

        def fetchall(self):
            return [{"symbol": "AAPL"}]

    @contextmanager
    def connection(_):
        yield None, Cursor()

    monkeypatch.setattr(generator, "get_connection", connection)
    monkeypatch.setattr(generator, "conn_params", lambda: {})
    with pytest.raises(RuntimeError, match="Suspicious R1000 snapshot size 1"):
        generator.fetch_symbols("R1000", date(2026, 11, 17))
