import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner

os.environ.setdefault("TIINGO_API_KEY", "test-only")

import data_retrieval as retrieval


POLICY_FILE = Path(retrieval.__file__).with_name("required_etf_assets.yaml")


class RequiredEtfCheckTest(unittest.TestCase):
    def setUp(self):
        self.policy = retrieval.load_etf_close_policy(POLICY_FILE)

    def test_symbols_and_thresholds_come_from_yaml(self):
        self.assertEqual(len(self.policy["symbols"]), 26)
        self.assertEqual(self.policy["full_history_min_rows"]["UWM"], 4900)
        self.assertEqual(self.policy["symbols"][:5], ("IVV", "IVW", "IVE", "IWM", "UWM"))

    def test_nyse_completed_session_before_and_after_close(self):
        before = datetime(2026, 10, 27, 16, 40, tzinfo=timezone.utc)
        after = datetime(2026, 10, 27, 21, 0, tzinfo=timezone.utc)
        self.assertEqual(retrieval.latest_completed_etf_close(before).date().isoformat(), "2026-10-26")
        self.assertEqual(retrieval.latest_completed_etf_close(after).date().isoformat(), "2026-10-27")

    def test_stale_nonempty_provider_response_does_not_pass(self):
        close_ts = pd.Timestamp("2026-10-26 20:00Z")
        old = pd.DataFrame({"ts": [pd.Timestamp("2026-10-23 20:00Z")],
                            "symbol": ["IVV"], "close": [100.0], "adj_close": [99.0]})
        with patch.object(retrieval, "fetch_equity_prices", return_value=old), \
             patch.object(retrieval, "save_equity_prices_bulk") as save:
            self.assertFalse(retrieval.refresh_completed_etf_bar("IVV", close_ts, self.policy))
        save.assert_not_called()

    def test_factor_and_replica_use_strict_tiingo_with_yaml_rules(self):
        close_ts = pd.Timestamp("2026-10-26 20:00Z")
        for symbol, expected_start in (("IVV", "2000-01-01"), ("GLD", "2026-09-21")):
            frame = pd.DataFrame({"ts": [close_ts], "symbol": [symbol],
                                  "close": [101.0], "adj_close": [99.0]})
            policy = dict(self.policy)
            policy["full_history_min_rows"] = {"IVV": 1}
            policy["min_recent_history_rows"] = 1
            with patch.object(retrieval, "fetch_equity_prices", return_value=frame) as fetch, \
                 patch.object(retrieval, "save_equity_prices_bulk") as save, \
                 patch.object(retrieval, "check_etf_history_coverage"), \
                 patch.object(retrieval, "stored_completed_etf_bar", return_value=True):
                self.assertTrue(retrieval.refresh_completed_etf_bar(symbol, close_ts, policy))
            self.assertEqual(fetch.call_args.kwargs["price_provider"], "tiingo")
            self.assertTrue(fetch.call_args.kwargs["strict_provider"])
            self.assertTrue(fetch.call_args.kwargs["require_adj_close"])
            self.assertEqual(fetch.call_args.kwargs["start_date"], expected_start)
            save.assert_called_once()

    def test_retry_and_independent_audit(self):
        calls = []
        def refresh(symbol, close_ts):
            calls.append(symbol)
            return symbol == "IVV" or calls.count(symbol) == 2
        retrieval.verify_required_etf_closes(("IVV", "IVE"), self.policy, attempts=2,
                                             delay_seconds=0, now="2026-10-27 21:00Z",
                                             refresh=refresh, sleep=lambda _: None)
        self.assertEqual(calls.count("IVV"), 1)
        self.assertEqual(calls.count("IVE"), 2)
        present = set(self.policy["symbols"]) - {"GLD"}
        def repair(symbol, close_ts):
            present.add(symbol)
            return True
        with patch.object(retrieval, "verify_required_etf_closes", wraps=retrieval.verify_required_etf_closes):
            retrieval.audit_required_etf_closes(self.policy, attempts=1,
                                                now="2026-10-27 21:00Z",
                                                stored=lambda symbol, _: symbol in present,
                                                refresh=repair)
        self.assertIn("GLD", present)

    def test_cli_modes_require_explicit_yaml(self):
        runner = CliRunner()
        with patch.object(retrieval, "verify_required_etf_closes") as verify:
            result = runner.invoke(retrieval.main, ["--assets_file", str(POLICY_FILE),
                                                    "--etf-close-check", "--attempts", "1"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(verify.call_args.args[0]), 26)
        result = runner.invoke(retrieval.main, ["--repair-missing"])
        self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
