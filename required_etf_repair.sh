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
PYTHON="${MARKET_DATA_PYTHON:-python}"

if ! "$PYTHON" "$ROOT/data_retrieval.py" \
  --assets_file "$REQUIRED_ETF_ASSETS" --etf-close-check \
  --repair-missing --attempts 3 --delay-seconds 120; then
  if [[ "${IB_ETF_FALLBACK:-0}" != "1" ]]; then
    echo "Required ETF closes remain missing after Tiingo repair. IB fallback is disabled; set IB_ETF_FALLBACK=1 to enable it." >&2
    exit 1
  fi
  echo "Tiingo repair failed; trying read-only IB historical closes" >&2
  "$PYTHON" "$ROOT/ib_etf_close_fallback.py" \
    "$REQUIRED_ETF_ASSETS"
fi
