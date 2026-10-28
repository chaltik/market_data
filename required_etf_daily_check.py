"""Require fresh, completed NYSE closes for the live ETF set."""

import argparse
import logging
import time
from functools import partial
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import yaml

from db_config import config
from db_utils import get_connection

FACTOR_SYMBOLS = ("IVV", "IVW", "IVE", "IWM", "UWM")
with Path(__file__).with_name("commodity_etf_assets.yaml").open() as handle:
    REPLICA_SYMBOLS = tuple(symbol for symbol in yaml.safe_load(handle)["equities"]
                            if symbol not in FACTOR_SYMBOLS)
REQUIRED_SYMBOLS = tuple(dict.fromkeys((*FACTOR_SYMBOLS, *REPLICA_SYMBOLS)))
MIN_FACTOR_HISTORY_ROWS = 6000
MIN_UWM_HISTORY_ROWS = 4900
MIN_REPLICA_HISTORY_ROWS = 500  # IBIT launched in 2024.
MIN_RECENT_HISTORY_ROWS = 10
RECENT_HISTORY_DAYS = 35
SETTLEMENT_DELAY = timedelta(minutes=30)


def latest_completed_close(now=None):
    """Return the most recent NYSE session close old enough for daily bars."""
    now = now or datetime.now(timezone.utc)
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    ny_date = now.tz_convert("America/New_York").date()
    schedule = mcal.get_calendar("NYSE").schedule(
        start_date=ny_date - timedelta(days=14), end_date=ny_date)
    completed = schedule[schedule["market_close"] <= now - SETTLEMENT_DELAY]
    if completed.empty:
        raise RuntimeError("No completed NYSE session found in the last 14 days")
    return completed["market_close"].iloc[-1]


def stored_completed_bar(symbol, close_ts):
    """Verify the exact session, not just a nonempty historical response."""
    with get_connection(config()) as (_, cur):
        cur.execute(
            """SELECT close, adj_close FROM price_data.equities_us_daily
               WHERE symbol = %s AND ts = %s""", (symbol, close_ts.to_pydatetime()))
        row = cur.fetchone()
    return bool(row and row["close"] is not None and row["adj_close"] is not None
                and row["close"] > 0 and row["adj_close"] > 0)


def check_history_coverage(symbol, frame):
    """Prevent a partial Tiingo response from leaving old Yahoo-only dates."""
    source_dates = set(pd.to_datetime(frame["ts"]).dt.tz_convert("America/New_York").dt.date)
    with get_connection(config()) as (_, cur):
        cur.execute(
            """SELECT (ts AT TIME ZONE 'America/New_York')::date AS trade_date
               FROM price_data.equities_us_daily WHERE symbol = %s""", (symbol,))
        old_dates = {row["trade_date"] for row in cur.fetchall()}
    missing = old_dates - source_dates
    if missing:
        raise RuntimeError(f"Tiingo history for {symbol} omits {len(missing)} existing dates")


def refresh_completed_bar(symbol, close_ts, full_history=False):
    """Use Tiingo EOD; refresh replica history weekly and recent bars nightly."""
    from data_retrieval import fetch_equity_prices, save_equity_prices_bulk

    provider = "tiingo"
    session_date = close_ts.tz_convert("America/New_York").date().isoformat()
    # The five factor histories are small enough to re-read every night.
    # Replica ETFs get a 35-day window nightly and a full refresh on Sundays.
    full_history = full_history or symbol in FACTOR_SYMBOLS
    start_date = ("2000-01-01" if full_history else
                  (close_ts.tz_convert("America/New_York").date() -
                   timedelta(days=RECENT_HISTORY_DAYS)).isoformat())
    frame = fetch_equity_prices(symbol, start_date=start_date, end_date=session_date,
                                require_adj_close=True, price_provider=provider,
                                strict_provider=True)
    if frame.empty:
        logging.warning("%s: %s returned no completed %s bar", symbol, provider, session_date)
        return False
    matches = frame[pd.to_datetime(frame["ts"], utc=True) == close_ts.tz_convert("UTC")]
    if len(matches) != 1:
        logging.warning("%s: %s returned %d bars, none uniquely matches %s",
                        symbol, provider, len(frame), close_ts)
        return False
    minimum_rows = (MIN_UWM_HISTORY_ROWS if symbol == "UWM" else
                    MIN_FACTOR_HISTORY_ROWS if symbol in FACTOR_SYMBOLS else
                    MIN_REPLICA_HISTORY_ROWS if full_history else MIN_RECENT_HISTORY_ROWS)
    if len(frame) < minimum_rows or frame["ts"].duplicated().any():
        raise RuntimeError(f"Suspicious Tiingo history for {symbol}: {len(frame)} rows")
    if (frame["adj_close"].isna().any() or (frame["adj_close"] <= 0).any()
            or frame["close"].isna().any() or (frame["close"] <= 0).any()):
        raise RuntimeError(f"Tiingo price history incomplete for {symbol}")
    if full_history:
        check_history_coverage(symbol, frame)
    save_equity_prices_bulk(frame)
    if not stored_completed_bar(symbol, close_ts):
        logging.error("%s: %s close did not persist", symbol, session_date)
        return False
    row = matches.iloc[0]
    logging.info("Verified %s %s via %s: raw close=%s adjusted close=%s",
                 symbol, session_date, provider, row["close"], row["adj_close"])
    return True


def verify_required(symbols=REQUIRED_SYMBOLS, attempts=3, delay_seconds=60,
                    now=None, refresh=refresh_completed_bar, sleep=time.sleep):
    close_ts = latest_completed_close(now)
    logging.info("Required ETF session: %s", close_ts)
    pending = set(symbols)
    for attempt in range(1, attempts + 1):
        for symbol in sorted(pending):
            try:
                if refresh(symbol, close_ts):
                    pending.remove(symbol)
            except Exception:
                logging.exception("%s refresh attempt %d failed", symbol, attempt)
        if not pending:
            return close_ts
        if attempt < attempts:
            logging.warning("Missing completed ETF bars %s; retrying in %s seconds",
                            sorted(pending), delay_seconds)
            sleep(delay_seconds)
    raise RuntimeError(f"Required ETF bars missing for {close_ts}: {sorted(pending)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--delay-seconds", type=int, default=60)
    parser.add_argument("--full-history", action="store_true",
                        help="Rebuild all selected ETF histories from Tiingo")
    parser.add_argument("--replica-only", action="store_true",
                        help="Check only the 21 ETF replica-book symbols")
    args = parser.parse_args()
    if args.attempts < 1 or args.delay_seconds < 0:
        parser.error("attempts must be >= 1 and delay-seconds must be >= 0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sunday = datetime.now(ZoneInfo("America/Los_Angeles")).weekday() == 6
    verify_required(symbols=REPLICA_SYMBOLS if args.replica_only else REQUIRED_SYMBOLS,
                    attempts=args.attempts, delay_seconds=args.delay_seconds,
                    refresh=partial(refresh_completed_bar,
                                    full_history=args.full_history or sunday))


if __name__ == "__main__":
    main()
