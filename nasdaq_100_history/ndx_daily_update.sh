#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYENV_VERSION="market_data"

PYENV="/home/captain/.pyenv/bin/pyenv"
PYTHON=("$PYENV" exec python)

tmpfile="$(mktemp -t ndx_today_XXXX).yaml"
trap 'rm -f "$tmpfile"' EXIT

asof="$(TZ=America/Los_Angeles date +%F)"
"${PYTHON[@]}" -m nasdaq_100_history.update_ndx_history --code NDX --asof "$asof"
"${PYTHON[@]}" -m nasdaq_100_history.gen_ndx_yaml \
  --code NDX --mode asof --asof "$asof" --output "$tmpfile"
"${PYTHON[@]}" data_retrieval.py --assets_file "$tmpfile" --assets-only
