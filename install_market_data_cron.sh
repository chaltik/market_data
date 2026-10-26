#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
schedule="$root/market_data_daily.cron"
job="$root/market_data_daily_update.sh"
fred_job="$root/hy_oas_daily_update.sh"
deployment_root="/home/captain/projects/market_data"
expected_fred="30 7 * * * $deployment_root/hy_oas_daily_update.sh >> $deployment_root/logs/hy_oas_daily.log 2>&1"
expected="0 18 * * * $deployment_root/market_data_daily_update.sh >> $deployment_root/logs/market_data_daily.log 2>&1"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--check" ) ]]; then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi

if [[ ! -x "$job" ]]; then
  echo "Daily update script is missing or not executable: $job" >&2
  exit 1
fi
if [[ ! -x "$fred_job" ]]; then
  echo "FRED update script is missing or not executable: $fred_job" >&2
  exit 1
fi
if ! grep -Fqx "$expected" "$schedule"; then
  echo "Cron schedule does not call the expected daily update script." >&2
  exit 1
fi
if ! grep -Fqx "$expected_fred" "$schedule"; then
  echo "Cron schedule does not call the expected FRED update script." >&2
  exit 1
fi

if [[ ${1:-} == "--check" ]]; then
  echo "Cron schedule and update script match."
  exit 0
fi

if [[ "$root" != "$deployment_root" ]]; then
  echo "Checkout is $root; cron is configured for $deployment_root" >&2
  exit 1
fi

crontab "$schedule"
if ! diff -q "$schedule" <(crontab -l) >/dev/null; then
  echo "Installed crontab does not match $schedule" >&2
  exit 1
fi
echo "Installed and verified $schedule"
