#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYENV_VERSION=market_data

/home/captain/.pyenv/bin/pyenv exec python "$ROOT/data_retrieval.py" \
  --assets_file "$ROOT/required_etf_assets.yaml" --etf-close-check \
  --repair-missing --attempts 3 --delay-seconds 120
