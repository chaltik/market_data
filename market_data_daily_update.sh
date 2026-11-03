#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -ne 1 || ! -f "$1" ]]; then
  echo "Usage: $0 REQUIRED_ETF_ASSETS.yaml" >&2
  exit 2
fi
REQUIRED_ETF_ASSETS="$(realpath "$1")"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYENV_VERSION="market_data"

PYENV="/home/captain/.pyenv/bin/pyenv"
PYTHON=("$PYENV" exec python)

# Refresh the Nasdaq-100 universe from the latest QQQ holdings, then pull
# prices for the current constituents. A failed universe check stops the job.
ndx_tmpfile="$(mktemp -t ndx_today_XXXX.yaml)"
combined_tmpfile="$(mktemp -t market_data_today_XXXX.yaml)"
etf_tmpfile="$(mktemp -t market_data_etfs_XXXX.yaml)"
trap 'rm -f "$ndx_tmpfile" "$combined_tmpfile" "$etf_tmpfile"' EXIT

asof="$(TZ=America/Los_Angeles date +%F)"
"${PYTHON[@]}" -m nasdaq_100_history.update_ndx_history --code NDX --asof "$asof"
"${PYTHON[@]}" -m nasdaq_100_history.gen_ndx_yaml \
  --code NDX --mode asof --asof "$asof" --output "$ndx_tmpfile"

# Refresh the approximate S&P 500 universe from State Street's SPY holdings,
# then pull prices into the same price_data.equities_us_daily table.
sp500_tmpfile="$(mktemp -t sp500_today_XXXX.yaml)"
djia_tmpfile="$(mktemp -t djia_today_XXXX.yaml)"
trap 'rm -f "$ndx_tmpfile" "$sp500_tmpfile" "$djia_tmpfile" "$combined_tmpfile" "$etf_tmpfile"' EXIT
"${PYTHON[@]}" -m sp500_history.update_sp500_history --code SP500 --asof "$asof"
"${PYTHON[@]}" -m sp500_history.gen_sp500_yaml \
  --code SP500 --mode asof --asof "$asof" --output "$sp500_tmpfile"

# DIA publishes a dated workbook. The loader rejects stale or incomplete files.
"${PYTHON[@]}" -m djia_history.load_djia --daily
"${PYTHON[@]}" -m djia_history.gen_djia_yaml --asof "$asof" --output "$djia_tmpfile"

# Merge stock-index constituents. ETFs are loaded in a separate pass.
"${PYTHON[@]}" - "$ndx_tmpfile" "$sp500_tmpfile" "$djia_tmpfile" \
  "$ROOT/commodity_etf_assets.yaml" "$REQUIRED_ETF_ASSETS" "$combined_tmpfile" "$etf_tmpfile" <<'PY'
import sys
import yaml

stocks = set()
for path in sys.argv[1:4]:
    with open(path) as handle:
        assets = yaml.safe_load(handle) or {}
    stocks.update(str(symbol) for symbol in assets.get("equities", []))

with open(sys.argv[4]) as handle:
    generic = yaml.safe_load(handle) or {}
with open(sys.argv[5]) as handle:
    required = yaml.safe_load(handle) or {}
required_etfs = set(required.get("equities", []))
generic_etfs = set(generic.get("equities", [])) - required_etfs
if stocks & (generic_etfs | required_etfs):
    raise SystemExit("ETF symbols overlap an index constituent universe")

with open(sys.argv[6], "w") as handle:
    yaml.safe_dump({"equities": sorted(stocks)}, handle, sort_keys=False)
with open(sys.argv[7], "w") as handle:
    yaml.safe_dump({"equities": sorted(generic_etfs)}, handle, sort_keys=False)
PY

# data_retrieval.py loads .env itself and writes to the existing
# price_data.equities_us storage tables. yfinance is explicit here to keep
# the scheduled job within Tiingo's Starter limits.
"${PYTHON[@]}" "$ROOT/data_retrieval.py" \
  --assets_file "$combined_tmpfile" \
  --assets-only \
  --price-provider yfinance
"${PYTHON[@]}" "$ROOT/data_retrieval.py" \
  --assets_file "$etf_tmpfile" \
  --assets-only \
  --price-provider yfinance

# The broad price pass can receive an older nonempty Yahoo response. Refetch
# these live-book ETFs and fail the cron if the last completed close is absent.
# All factor and replica-book ETFs use Tiingo EOD.
"${PYTHON[@]}" "$ROOT/data_retrieval.py" \
  --assets_file "$REQUIRED_ETF_ASSETS" --etf-close-check
