#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -ne 1 || ! -f "$1" ]]; then
  echo "Usage: $0 REQUIRED_ETF_ASSETS.yaml" >&2
  exit 2
fi
REQUIRED_ETF_ASSETS="$(realpath "$1")"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYENV_VERSION=market_data

if ! /home/captain/.pyenv/bin/pyenv exec python "$ROOT/data_retrieval.py" \
  --assets_file "$REQUIRED_ETF_ASSETS" --etf-close-check \
  --repair-missing --attempts 3 --delay-seconds 120; then
  echo "Tiingo repair failed; trying read-only IB historical closes" >&2
  /home/captain/.pyenv/versions/strader/bin/python "$ROOT/ib_etf_close_fallback.py" \
    "$REQUIRED_ETF_ASSETS"
fi
