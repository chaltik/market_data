#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYENV_VERSION="3.12.8/envs/strader"

# data_retrieval.py loads .env itself and uses the existing Tiingo-first,
# yfinance-fallback provider path and price_data.equities_us storage tables.
exec /home/captain/.pyenv/bin/pyenv exec python "$ROOT/data_retrieval.py" \
  --assets_file "$ROOT/commodity_etf_assets.yaml" \
  --assets-only
