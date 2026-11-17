import unittest
from unittest.mock import patch

from russell_1000_history.update_r1000_history import (
    load_excluded_symbols,
    normalize_symbol,
    parse_holdings,
)


class R1000HoldingsTest(unittest.TestCase):
    def test_current_holdings_header_and_unreported_small_weight(self):
        content = ('Fund Holdings as of,"Nov 16, 2026"\n'
                   'Ticker,Name,Type,Sector,Asset Class,Market Value,Market Weight\n'
                   'AAPL,Apple,EQUITY,Tech,Equity,990,99.00\n'
                   'SMALL,Small,EQUITY,Tech,Equity,10,-\n')
        with patch('russell_1000_history.update_r1000_history.MIN_ROWS', 1):
            _, holdings, _ = parse_holdings(content)
        self.assertEqual([holding.weight for holding in holdings], [99., 1.])

    def test_share_classes_and_rights(self):
        self.assertEqual(normalize_symbol("BRK B"), "BRK.B")
        self.assertEqual(normalize_symbol("BF A"), "BF.A")
        self.assertIsNone(normalize_symbol("UWMC-R-W"))

    def test_rejects_missing_file_date(self):
        with self.assertRaisesRegex(ValueError, "holdings date"):
            parse_holdings("Ticker,Name,Sector,Asset Class\nAAPL,Apple,Tech,Equity\n")

    def test_rejects_too_few_equities(self):
        content = (
            'iShares Russell 1000 ETF\nFund Holdings as of,"Oct 23, 2026"\n'
            "Ticker,Name,Sector,Asset Class,Weight (%)\n"
            "AAPL,Apple,Tech,Equity,1.0\nCASH,Cash,Cash,Cash,1.0\n"
        )
        with self.assertRaisesRegex(ValueError, "valid equities"):
            parse_holdings(content)

    def test_retired_holding_exclusion_is_versioned(self):
        self.assertIn("HOLX", load_excluded_symbols())


if __name__ == "__main__":
    unittest.main()
