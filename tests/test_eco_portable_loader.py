import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner

os.environ.setdefault("TIINGO_API_KEY", "test-only")

import data_retrieval as retrieval


class PortableEcoLoaderTest(unittest.TestCase):
    def test_direct_fred_ids_need_no_untracked_yaml(self):
        with patch.object(retrieval.Path, "is_file", return_value=False):
            self.assertEqual(retrieval.resolve_eco_series("GDP,CPIAUCSL,GDP"),
                             [("GDP", "GDP", "GDP"),
                              ("CPIAUCSL", "CPIAUCSL", "CPIAUCSL")])

    def test_explicit_catalog_rejects_unknown_id(self):
        with TemporaryDirectory() as directory:
            catalog = Path(directory) / "series.yaml"
            catalog.write_text("group:\n  - fred_series_id: GDP\n"
                               "    series_name: GDP\n    release_name: GDP\n")
            with self.assertRaisesRegex(Exception, "not in"):
                retrieval.resolve_eco_series("CPIAUCSL", catalog)

    def test_hy_oas_uses_its_dedicated_loader(self):
        with patch.object(retrieval.Path, "is_file", return_value=False):
            with self.assertRaisesRegex(Exception, "hy_oas.py"):
                retrieval.resolve_eco_series("BAMLH0A0HYM2")

    def test_eco_only_skips_assets_yaml_and_tiingo(self):
        frame = pd.DataFrame({"fred_series_id": ["GDP"], "series_name": ["GDP"],
                              "release_name": ["GDP"], "release_id": [20261001],
                              "release_date": [pd.Timestamp("2026-10-01")],
                              "reference_date": [pd.Timestamp("2026-07-01")],
                              "value": [1.0]})
        with patch.dict(os.environ, {"FRED_API_KEY": "test-only"}), \
             patch.object(retrieval, "fetch_eco_from_fred", return_value=frame) as fetch, \
             patch.object(retrieval, "save_eco_from_fred_to_db") as save:
            result = CliRunner().invoke(retrieval.main, ["--eco-only", "--eco-series", "GDP",
                                                        "--assets_file", "/does/not/exist.yaml"])
        self.assertEqual(result.exit_code, 0, result.output)
        fetch.assert_called_once()
        save.assert_called_once()

    def test_failed_series_returns_nonzero_exit(self):
        with patch.dict(os.environ, {"FRED_API_KEY": "test-only"}), \
             patch.object(retrieval, "fetch_eco_from_fred", side_effect=RuntimeError("API down")):
            result = CliRunner().invoke(retrieval.main, ["--eco-only", "--eco-series", "GDP"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("GDP", result.output)


if __name__ == "__main__":
    unittest.main()
