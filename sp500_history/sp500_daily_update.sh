#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYENV_VERSION="market_data"

PYENV="/home/captain/.pyenv/bin/pyenv"
PYTHON=("$PYENV" exec python)

tmpfile="$(mktemp -t sp500_today_XXXX).yaml"
trap 'rm -f "$tmpfile"' EXIT

asof="$(TZ=America/Los_Angeles date +%F)"
"${PYTHON[@]}" -m sp500_history.update_sp500_history --code SP500 --asof "$asof"
"${PYTHON[@]}" -m sp500_history.gen_sp500_yaml \
  --code SP500 --mode asof --asof "$asof" --output "$tmpfile"
"${PYTHON[@]}" data_retrieval.py --assets_file "$tmpfile" --assets-only
