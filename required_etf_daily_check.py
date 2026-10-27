"""Require fresh, completed NYSE closes for the live factor and UWM ETFs."""

import argparse
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pandas_market_calendars as mcal

from db_config import config
from db_utils import get_connection

REQUIRED_SYMBOLS = ("IVV", "IVW", "IVE", "IWM", "UWM")
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


def refresh_completed_bar(symbol, close_ts):
    """Use the same Yahoo raw-close/adjusted-close convention as the nightly loader."""
    os.environ["MARKET_DATA_PRICE_PROVIDER"] = "yfinance"
    from data_retrieval import fetch_equity_prices, save_equity_prices

    session_date = close_ts.tz_convert("America/New_York").date().isoformat()
    frame = fetch_equity_prices(symbol, start_date=session_date, end_date=session_date,
                                require_adj_close=True)
    if frame.empty:
        logging.warning("%s: Yahoo returned no completed %s bar", symbol, session_date)
        return False
    matches = frame[pd.to_datetime(frame["ts"], utc=True) == close_ts.tz_convert("UTC")]
    if len(matches) != 1:
        logging.warning("%s: Yahoo returned %d bars, none uniquely matches %s",
                        symbol, len(frame), close_ts)
        return False
    save_equity_prices(matches)
    if not stored_completed_bar(symbol, close_ts):
        logging.error("%s: %s close did not persist", symbol, session_date)
        return False
    row = matches.iloc[0]
    logging.info("Verified %s %s: raw close=%s adjusted close=%s",
                 symbol, session_date, row["close"], row["adj_close"])
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
    args = parser.parse_args()
    if args.attempts < 1 or args.delay_seconds < 0:
        parser.error("attempts must be >= 1 and delay-seconds must be >= 0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    verify_required(attempts=args.attempts, delay_seconds=args.delay_seconds)


if __name__ == "__main__":
    main()
