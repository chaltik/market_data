#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYENV_VERSION=market_data

/home/captain/.pyenv/bin/pyenv exec python "$ROOT/required_etf_daily_check.py" \
  --repair-missing --attempts 3 --delay-seconds 120
