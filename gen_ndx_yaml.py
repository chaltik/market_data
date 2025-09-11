# gen_ndx_yaml.py
# --------------------------------------------
# Generate a YAML of equities to feed data_retrieval.py
# Modes:
#   --mode all   -> all historical NDX constituents (distinct over time)
#   --mode asof  -> constituents as of a date (uses constituents_asof)
#
# Examples:
#   python gen_ndx_yaml.py --mode all --output ndx_all.yaml
#   python gen_ndx_yaml.py --mode asof --asof 2025-09-08 --stdout
#
import argparse
import re
import sys
from datetime import date
from typing import List

from db_utils import get_connection
from db_config import config as conn_params

TICKER_RE = re.compile(r"^[A-Z\.]{1,6}$")

def _fmt_yaml(equities: List[str]) -> str:
    equities = sorted({s for s in equities if TICKER_RE.match(s)})
    lines = ["equities:"]
    lines += [f"  - {s}" for s in equities]
    # keep crypto key to satisfy parsers expecting it; leave empty
    lines += ["", "crypto: []", ""]
    return "\n".join(lines)

def fetch_all_symbols(code: str) -> List[str]:
    params = conn_params()
    with get_connection(params) as (conn, cur):
        cur.execute("""
            SELECT ic.symbol
            FROM market_indices.index_constituents ic
            JOIN market_indices.indices i ON i.index_id = ic.index_id
            WHERE i.code = %(code)s
            GROUP BY ic.symbol
            ORDER BY ic.symbol
        """, {"code": code})
        return [r["symbol"].upper() for r in cur.fetchall()]

def fetch_asof_symbols(code: str, asof: date) -> List[str]:
    params = conn_params()
    with get_connection(params) as (conn, cur):
        cur.execute("""
            SELECT symbol
            FROM market_indices.constituents_asof(%(code)s, %(asof)s)
            ORDER BY symbol
        """, {"code": code, "asof": asof})
        return [r["symbol"].upper() for r in cur.fetchall()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="NDX", help="Index code (default: NDX)")
    ap.add_argument("--mode", choices=["all", "asof"], default="asof")
    ap.add_argument("--asof", help="YYYY-MM-DD (default: today for --mode asof)")
    ap.add_argument("--output", help="Write YAML to this file")
    ap.add_argument("--stdout", action="store_true", help="Write YAML to stdout")
    args = ap.parse_args()

    if args.mode == "asof":
        asof = date.fromisoformat(args.asof) if args.asof else date.today()
        syms = fetch_asof_symbols(args.code, asof)
    else:
        syms = fetch_all_symbols(args.code)

    yaml_text = _fmt_yaml(syms)

    if args.stdout or not args.output:
        sys.stdout.write(yaml_text)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(yaml_text)

if __name__ == "__main__":
    main()