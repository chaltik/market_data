import unittest
from datetime import date
from types import SimpleNamespace

from ib_etf_close_fallback import exact_bar


class IbEtfFallbackTest(unittest.TestCase):
    def test_requires_exact_completed_session(self):
        prior = SimpleNamespace(date=date(2026, 10, 30), close=100.0)
        completed = SimpleNamespace(date=date(2026, 11, 2), close=101.0)
        self.assertIs(exact_bar([prior, completed], date(2026, 11, 2)), completed)
        with self.assertRaisesRegex(ValueError, "0 bars"):
            exact_bar([prior], date(2026, 11, 2))
        with self.assertRaisesRegex(ValueError, "2 bars"):
            exact_bar([completed, completed], date(2026, 11, 2))

    def test_rejects_nonpositive_close(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            exact_bar([SimpleNamespace(date=date(2026, 11, 2), close=0)],
                      date(2026, 11, 2))


if __name__ == "__main__":
    unittest.main()
