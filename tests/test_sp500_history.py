import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from sp500_history.build_sp500_history import (
    Holding,
    build_transitions,
    compute_intervals,
    parse_history_csv,
    parse_spy_holdings,
)


class Sp500HistoryTests(unittest.TestCase):
    def test_history_parser_and_ticker_rename(self):
        original = [f"A{i}" for i in range(450)]
        symbols_a = ",".join(original)
        symbols_b = ",".join("NEW10" if symbol == "A10" else symbol for symbol in original)
        csv = f"date,tickers\n1996-01-02,\"{symbols_a}\"\n1996-02-01,\"{symbols_b}\"\n"
        snapshots = parse_history_csv(csv)
        transitions = build_transitions(snapshots)
        self.assertIn(("REMOVE", "A10", date(1996, 2, 1)), transitions)
        self.assertIn(("ADD", "NEW10", date(1996, 2, 1)), transitions)

        indexed = [(action, symbol, when, index) for index, (action, symbol, when) in enumerate(transitions)]
        intervals = compute_intervals(indexed)
        self.assertIn(("A10", date(1996, 1, 2), date(1996, 2, 1), 10, 450), intervals)
        self.assertIn(("NEW10", date(1996, 2, 1), None, 451, None), intervals)

    def test_spy_parser_finds_metadata_header(self):
        rows = [["Fund Name:", "SPY", "", ""], ["As of", "2026-10-16", "", ""],
                ["Name", "Ticker", "Weight", "Sector"]]
        rows.extend([[f"Company {i}", f"A{i}", "1", "Technology"] for i in range(450)])
        with patch(
            "sp500_history.build_sp500_history.pd.read_excel",
            return_value=pd.DataFrame(rows),
        ):
            holdings = parse_spy_holdings(b"xlsx")
        self.assertEqual(len(holdings), 450)
        self.assertEqual(holdings[0], Holding("A0", "Company 0", "Technology", 1.0))

    def test_validation_rejects_non_large_cap_snapshot(self):
        csv = "date,tickers\n1996-01-02,\"AAPL,MSFT\"\n"
        with self.assertRaisesRegex(ValueError, "expected 450-550"):
            parse_history_csv(csv)


if __name__ == "__main__":
    unittest.main()
