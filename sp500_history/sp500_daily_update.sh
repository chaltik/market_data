#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

tmpfile="$(mktemp -t sp500_today_XXXX).yaml"
trap 'rm -f "$tmpfile"' EXIT

PYTHONPATH=. python -m sp500_history.update_sp500_history
PYTHONPATH=. python -m sp500_history.gen_sp500_yaml --mode asof --asof "$(date +%F)" --output "$tmpfile"
eval "$(cat .env)" PYTHONPATH=. python data_retrieval.py --assets_file "$tmpfile"
