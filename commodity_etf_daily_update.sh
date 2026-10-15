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
# prices for the current constituents.  Keep this before the commodity run so
# a failed universe check prevents a misleading partial scheduled update.
ndx_tmpfile="$(mktemp -t ndx_today_XXXX.yaml)"
trap 'rm -f "$ndx_tmpfile"' EXIT

ndx_asof="$(TZ=America/Los_Angeles date +%F)"
"${PYTHON[@]}" -m nasdaq_100_history.update_ndx_history --code NDX --asof "$ndx_asof"
"${PYTHON[@]}" -m nasdaq_100_history.gen_ndx_yaml \
  --code NDX \
  --mode asof \
  --asof "$ndx_asof" \
  --output "$ndx_tmpfile"

# data_retrieval.py loads .env itself and uses the existing Tiingo-first,
# yfinance-fallback provider path and price_data.equities_us storage tables.
"${PYTHON[@]}" "$ROOT/data_retrieval.py" \
  --assets_file "$ndx_tmpfile" \
  --assets-only

# Refresh the existing commodity/ETF universe in the same scheduled run.
"${PYTHON[@]}" "$ROOT/data_retrieval.py" \
  --assets_file "$ROOT/commodity_etf_assets.yaml" \
  --assets-only
