import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

os.environ.setdefault("TIINGO_API_KEY", "test-only")

import data_retrieval
from data_retrieval import _completed_equity_bars
from required_etf_daily_check import latest_completed_close, refresh_completed_bar, verify_required


class RequiredEtfCheckTest(unittest.TestCase):
    def test_nyse_completed_session_before_and_after_close(self):
        before = datetime(2026, 10, 27, 16, 40, tzinfo=timezone.utc)  # 12:40 ET
        after = datetime(2026, 10, 27, 21, 0, tzinfo=timezone.utc)   # 17:00 ET
        self.assertEqual(latest_completed_close(before).date().isoformat(), "2026-10-26")
        self.assertEqual(latest_completed_close(after).date().isoformat(), "2026-10-27")

    def test_provisional_today_is_dropped(self):
        frame = pd.DataFrame({"ts": [pd.Timestamp("2026-10-26 20:00Z"),
                                      pd.Timestamp("2026-10-27 20:00Z")],
                              "close": [1.0, 2.0]})
        complete = _completed_equity_bars(frame, now="2026-10-27 16:40Z")
        self.assertEqual(complete["close"].tolist(), [1.0])

    def test_retry_and_fail_visible(self):
        calls = []
        def refresh(symbol, close_ts):
            calls.append(symbol)
            return symbol == "IVV" or calls.count(symbol) == 2
        close_ts = verify_required(symbols=("IVV", "IVE"), attempts=2,
                                   delay_seconds=0, now="2026-10-27 21:00Z",
                                   refresh=refresh, sleep=lambda _: None)
        self.assertEqual(close_ts.date().isoformat(), "2026-10-27")
        self.assertEqual(calls.count("IVV"), 1)
        self.assertEqual(calls.count("IVE"), 2)

        with self.assertRaisesRegex(RuntimeError, "Required ETF bars missing"):
            verify_required(symbols=("IVE",), attempts=2, delay_seconds=0,
                            now="2026-10-27 21:00Z", refresh=lambda *_: False,
                            sleep=lambda _: None)

    def test_factor_etf_uses_tiingo_raw_and_adjusted_columns(self):
        close_ts = pd.Timestamp("2026-10-26 20:00Z")
        frame = pd.DataFrame({"ts": [close_ts], "symbol": ["IVV"],
                              "close": [101.0], "adj_close": [99.0]})
        with patch.object(data_retrieval, "fetch_equity_prices", return_value=frame) as fetch, \
             patch.object(data_retrieval, "save_equity_prices_bulk") as save, \
             patch("required_etf_daily_check.MIN_FACTOR_HISTORY_ROWS", 1), \
             patch("required_etf_daily_check.check_history_coverage"), \
             patch("required_etf_daily_check.stored_completed_bar", return_value=True):
            self.assertTrue(refresh_completed_bar("IVV", close_ts))
        self.assertTrue(fetch.call_args.kwargs["require_adj_close"])
        self.assertEqual(fetch.call_args.kwargs["price_provider"], "tiingo")
        self.assertTrue(fetch.call_args.kwargs["strict_provider"])
        self.assertEqual(fetch.call_args.kwargs["start_date"], "2000-01-01")
        saved = save.call_args.args[0]
        self.assertEqual(saved.iloc[0]["close"], 101.0)
        self.assertEqual(saved.iloc[0]["adj_close"], 99.0)

    def test_stale_nonempty_provider_response_does_not_pass(self):
        close_ts = pd.Timestamp("2026-10-26 20:00Z")
        old = pd.DataFrame({"ts": [pd.Timestamp("2026-10-23 20:00Z")],
                            "symbol": ["IVV"], "close": [100.0], "adj_close": [99.0]})
        with patch.object(data_retrieval, "fetch_equity_prices", return_value=old), \
             patch.object(data_retrieval, "save_equity_prices") as save:
            self.assertFalse(refresh_completed_bar("IVV", close_ts))
        save.assert_not_called()

    def test_uwm_uses_tiingo_full_history(self):
        close_ts = pd.Timestamp("2026-10-26 20:00Z")
        frame = pd.DataFrame({"ts": [close_ts], "symbol": ["UWM"],
                              "close": [56.0], "adj_close": [56.0]})
        with patch.object(data_retrieval, "fetch_equity_prices", return_value=frame) as fetch, \
             patch.object(data_retrieval, "save_equity_prices_bulk") as save, \
             patch("required_etf_daily_check.MIN_UWM_HISTORY_ROWS", 1), \
             patch("required_etf_daily_check.check_history_coverage"), \
             patch("required_etf_daily_check.stored_completed_bar", return_value=True):
            self.assertTrue(refresh_completed_bar("UWM", close_ts))
        self.assertEqual(fetch.call_args.kwargs["price_provider"], "tiingo")
        self.assertTrue(fetch.call_args.kwargs["strict_provider"])
        self.assertEqual(fetch.call_args.kwargs["start_date"], "2000-01-01")
        save.assert_called_once()

    def test_strict_tiingo_does_not_fall_back_to_yahoo(self):
        with patch.object(data_retrieval, "get_latest_ts_for_symbol", return_value=None), \
             patch.object(data_retrieval.ti_client, "get_dataframe", side_effect=RuntimeError("unavailable")), \
             patch.object(data_retrieval.yf, "Ticker") as yahoo:
            with self.assertRaisesRegex(RuntimeError, "Yahoo fallback disabled"):
                data_retrieval.fetch_equity_prices(
                    "IVV", start_date="2026-10-26", end_date="2026-10-26",
                    price_provider="tiingo", strict_provider=True)
        yahoo.assert_not_called()


if __name__ == "__main__":
    unittest.main()
