#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
schedule="$root/market_data_daily.cron"
job="$root/market_data_daily_update.sh"
fred_job="$root/hy_oas_daily_update.sh"
repair_job="$root/required_etf_repair.sh"
deployment_root="/home/captain/projects/market_data"
if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$1" != "--check" ) ]]; then
  echo "Usage: $0 [--check] REQUIRED_ETF_ASSETS.yaml" >&2
  exit 2
fi
required_etf_assets="$(realpath "${!#}")"
if [[ ! -f "$required_etf_assets" ]]; then
  echo "Required ETF assets file is missing: $required_etf_assets" >&2
  exit 2
fi
expected_fred="30 7 * * * $deployment_root/hy_oas_daily_update.sh >> $deployment_root/logs/hy_oas_daily.log 2>&1"
expected="0 18 * * * $deployment_root/market_data_daily_update.sh $required_etf_assets >> $deployment_root/logs/market_data_daily.log 2>&1"
expected_repair_evening="15 19 * * * $deployment_root/required_etf_repair.sh $required_etf_assets >> $deployment_root/logs/required_etf_repair.log 2>&1"
expected_repair_morning="0 7 * * * $deployment_root/required_etf_repair.sh $required_etf_assets >> $deployment_root/logs/required_etf_repair.log 2>&1"

if [[ ! -x "$job" ]]; then
  echo "Daily update script is missing or not executable: $job" >&2
  exit 1
fi
if [[ ! -x "$fred_job" ]]; then
  echo "FRED update script is missing or not executable: $fred_job" >&2
  exit 1
fi
if [[ ! -x "$repair_job" ]]; then
  echo "ETF repair script is missing or not executable: $repair_job" >&2
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
for expected_repair in "$expected_repair_evening" "$expected_repair_morning"; do
  if ! grep -Fqx "$expected_repair" "$schedule"; then
    echo "Cron schedule does not call the expected ETF repair script." >&2
    exit 1
  fi
done

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
