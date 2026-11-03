import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "required_etf_repair.sh"


class RequiredEtfRepairTest(unittest.TestCase):
    def test_ib_is_optional_after_tiingo_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            policy = root / "policy.yaml"
            policy.write_text("symbols: [TEST]\n")
            calls = root / "calls"
            fake_python = root / "python"
            fake_python.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$1\" >> \"$CALLS\"\n"
                "case \"$1\" in *data_retrieval.py) exit 1;; esac\n"
                "exit 0\n"
            )
            fake_python.chmod(0o755)
            env = dict(os.environ, MARKET_DATA_PYTHON=str(fake_python), CALLS=str(calls))
            env.pop("IB_ETF_FALLBACK", None)
            no_ib = subprocess.run([str(SCRIPT), str(policy)], env=env,
                                   capture_output=True, text=True)
            self.assertEqual(no_ib.returncode, 1)
            self.assertIn("IB fallback is disabled", no_ib.stderr)
            self.assertEqual(len(calls.read_text().splitlines()), 1)

            env["IB_ETF_FALLBACK"] = "1"
            with_ib = subprocess.run([str(SCRIPT), str(policy)], env=env,
                                     capture_output=True, text=True)
            self.assertEqual(with_ib.returncode, 0)
            self.assertTrue(calls.read_text().splitlines()[-1].endswith("ib_etf_close_fallback.py"))


if __name__ == "__main__":
    unittest.main()
