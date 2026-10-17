#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
trap 'rm -f "$ndx_tmpfile"' EXIT

asof="$(TZ=America/Los_Angeles date +%F)"
"${PYTHON[@]}" -m nasdaq_100_history.update_ndx_history --code NDX --asof "$asof"
"${PYTHON[@]}" -m nasdaq_100_history.gen_ndx_yaml \
  --code NDX --mode asof --asof "$asof" --output "$ndx_tmpfile"
"${PYTHON[@]}" "$ROOT/data_retrieval.py" \
  --assets_file "$ndx_tmpfile" \
  --assets-only

# Refresh the approximate S&P 500 universe from State Street's SPY holdings,
# then pull prices into the same price_data.equities_us_daily table.
sp500_tmpfile="$(mktemp -t sp500_today_XXXX.yaml)"
trap 'rm -f "$ndx_tmpfile" "$sp500_tmpfile"' EXIT
"${PYTHON[@]}" -m sp500_history.update_sp500_history --code SP500 --asof "$asof"
"${PYTHON[@]}" -m sp500_history.gen_sp500_yaml \
  --code SP500 --mode asof --asof "$asof" --output "$sp500_tmpfile"
"${PYTHON[@]}" "$ROOT/data_retrieval.py" \
  --assets_file "$sp500_tmpfile" \
  --assets-only

# data_retrieval.py loads .env itself and uses the existing Tiingo-first,
# yfinance-fallback provider path and price_data.equities_us storage tables.
"${PYTHON[@]}" "$ROOT/data_retrieval.py" \
  --assets_file "$ROOT/commodity_etf_assets.yaml" \
  --assets-only
