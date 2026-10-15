#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

tmpfile="$(mktemp -t ndx_today_XXXX).yaml"
trap 'rm -f "$tmpfile"' EXIT

PYTHONPATH=. python -m nasdaq_100_history.update_ndx_history
PYTHONPATH=. python -m nasdaq_100_history.gen_ndx_yaml --mode asof --asof "$(date +%F)" --output "$tmpfile"
eval "$(cat .env)" PYTHONPATH=. python data_retrieval.py --assets_file "$tmpfile"
