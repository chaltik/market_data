"""Collect a filtered Yahoo options surface for the stored DJIA universe."""

import argparse
import json
import math
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import execute_values
import pandas_market_calendars as market_calendars
import yfinance as yf

from db_config import config

ET = ZoneInfo("America/New_York")
LOCK_KEY = 400061107  # prevents overlapping options sweeps on one coredata DB
CONTRACT_RE = re.compile(r"^[A-Z.]+\d{6}[CP]\d{8}$")


def finite_number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_contract(record, symbol, expiry, side, captured_at, underlying_price,
                       minimum_oi=100, maximum_spread=0.25):
    """Return a compact quote row, or None for an illiquid/bad contract."""
    contract = str(record.get("contractSymbol") or "")
    date_code = expiry.strftime("%y%m%d")
    if not CONTRACT_RE.fullmatch(contract) or not contract.startswith(symbol + date_code + side):
        return None
    strike = finite_number(record.get("strike"))
    bid = finite_number(record.get("bid"))
    ask = finite_number(record.get("ask"))
    oi = finite_number(record.get("openInterest"))
    if strike is None or strike <= 0 or bid is None or ask is None or bid <= 0 or ask < bid:
        return None
    midpoint = (bid + ask) / 2
    if midpoint <= 0 or (ask - bid) / midpoint > maximum_spread:
        return None
    if oi is None or oi < minimum_oi:
        return None
    last_trade = record.get("lastTradeDate")
    if hasattr(last_trade, "to_pydatetime"):
        last_trade = last_trade.to_pydatetime()
    if not isinstance(last_trade, datetime) or str(last_trade) == "NaT":
        last_trade = None
    elif last_trade.tzinfo is None:
        last_trade = last_trade.replace(tzinfo=timezone.utc)
    volume = finite_number(record.get("volume"))
    iv = finite_number(record.get("impliedVolatility"))
    last = finite_number(record.get("lastPrice"))
    itm = record.get("inTheMoney")
    if isinstance(itm, float) and math.isnan(itm):
        itm = None
    return (captured_at, symbol, expiry, contract, side, strike, bid, ask,
            last, iv, int(oi), None if volume is None else int(volume),
            last_trade, None if itm is None else bool(itm), underlying_price)


def rate_limited(error):
    message = str(error).lower()
    return "429" in message or "rate limit" in message or "too many requests" in message


def within_market_window(now_et):
    """Allow hourly polls from 10:00 ET through the NYSE close, including early closes."""
    schedule = market_calendars.get_calendar("NYSE").schedule(
        start_date=now_et.date(), end_date=now_et.date())
    if schedule.empty:
        return False
    session = schedule.iloc[0]
    now_utc = now_et.astimezone(timezone.utc)
    first_poll = session["market_open"].to_pydatetime() + timedelta(minutes=30)
    last_poll = session["market_close"].to_pydatetime() + timedelta(minutes=5)
    return first_poll <= now_utc <= last_poll


def fetch_universe(cur, requested):
    cur.execute("SELECT max(as_of_date) FROM market_indices.seed_snapshots s JOIN market_indices.indices i USING (index_id) WHERE i.code='DJIA'")
    universe_asof = cur.fetchone()[0]
    cur.execute("SELECT symbol FROM market_indices.constituents_asof('DJIA', CURRENT_DATE) ORDER BY symbol")
    all_symbols = [row[0] for row in cur.fetchall()]
    if len(all_symbols) != 30 or len(set(all_symbols)) != 30:
        raise ValueError(f"DJIA universe must have 30 distinct members; got {len(all_symbols)}")
    if requested:
        unknown = set(requested) - set(all_symbols)
        if unknown:
            raise ValueError(f"Not current DJIA constituents: {sorted(unknown)}")
        return universe_asof, sorted(set(requested))
    return universe_asof, all_symbols


def table_bytes(cur):
    cur.execute("SELECT pg_total_relation_size('options_data.quote_snapshots')")
    return cur.fetchone()[0]


def start_run(cur, asof, symbols, minimum_oi, maximum_spread, max_expirations):
    before = table_bytes(cur)
    cur.execute("""
        INSERT INTO options_data.collection_runs
          (status,universe_code,universe_asof,symbols_requested,filters,table_bytes_before)
        VALUES ('running','DJIA',%s,%s,%s::jsonb,%s) RETURNING run_id
    """, (asof, len(symbols), json.dumps({"minimum_open_interest": minimum_oi,
                                         "maximum_relative_spread": maximum_spread,
                                         "positive_bid": True,
                                         "max_expirations": max_expirations or "all"}), before))
    return cur.fetchone()[0]


def save_quotes(cur, run_id, rows):
    if not rows:
        return
    execute_values(cur, """
        INSERT INTO options_data.quote_snapshots
          (run_id,captured_at,underlying_symbol,expiration_date,contract_symbol,
           option_type,strike,bid,ask,last_price,implied_volatility,open_interest,
           volume,last_trade_at,in_the_money,underlying_price)
        VALUES %s ON CONFLICT (run_id,contract_symbol) DO NOTHING
    """, [(run_id, *row) for row in rows], page_size=500)


def run(args):
    now_et = datetime.now(ET)
    if args.start_date and now_et.date() < args.start_date:
        print("Options pilot has not started; no requests made")
        return 0
    if args.end_date and now_et.date() > args.end_date:
        print("Options pilot ended; no requests made")
        return 0
    if args.market_hours and not within_market_window(now_et):
        print("NYSE closed or outside pilot market-hours window; no requests made")
        return 0
    params = config(filename=args.db_config) if args.db_config else config()
    conn = psycopg2.connect(**params)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,))
            if not cur.fetchone()[0]:
                raise RuntimeError("Another DJIA options collection is already running")
            asof, symbols = fetch_universe(cur, args.symbols)
            run_id = None if args.dry_run else start_run(cur, asof, symbols, args.minimum_oi,
                                                         args.maximum_spread, args.max_expirations)
        conn.commit()
        stats = {}
        totals = {"symbols_completed": 0, "expirations_seen": 0,
                  "expirations_completed": 0, "contracts_seen": 0,
                  "contracts_retained": 0}
        errors = []
        hit_limit = False
        for symbol in symbols:
            item = {"expirations_seen": 0, "expirations_completed": 0,
                    "contracts_seen": 0, "contracts_retained": 0, "errors": []}
            stats[symbol] = item
            ticker = yf.Ticker(symbol)
            try:
                expirations = tuple(ticker.options)
                if not expirations:
                    raise ValueError("Yahoo returned no expirations")
            except Exception as error:
                item["errors"].append(f"expirations: {type(error).__name__}: {error}")
                errors.append(symbol)
                if rate_limited(error):
                    hit_limit = True
                    break
                continue
            if args.max_expirations:
                expirations = expirations[:args.max_expirations]
            item["expirations_seen"] = len(expirations)
            totals["expirations_seen"] += len(expirations)
            for expiry_text in expirations:
                try:
                    expiry = date.fromisoformat(expiry_text)
                    chain = ticker.option_chain(expiry_text)
                    if chain.calls is None or chain.puts is None:
                        raise ValueError("Yahoo returned an empty option side")
                    fetched_at = datetime.now(timezone.utc)
                    underlying = chain.underlying or {}
                    underlying_price = finite_number(underlying.get("regularMarketPrice"))
                    rows = []
                    seen = 0
                    for side, frame in (("C", chain.calls), ("P", chain.puts)):
                        for record in frame.to_dict("records"):
                            seen += 1
                            quote = normalize_contract(record, symbol, expiry, side, fetched_at,
                                                       underlying_price, args.minimum_oi, args.maximum_spread)
                            if quote is not None:
                                rows.append(quote)
                    if seen == 0:
                        raise ValueError("Yahoo returned zero contracts")
                    if run_id is not None:
                        with conn.cursor() as cur:
                            save_quotes(cur, run_id, rows)
                        conn.commit()
                    item["expirations_completed"] += 1
                    item["contracts_seen"] += seen
                    item["contracts_retained"] += len(rows)
                    totals["expirations_completed"] += 1
                    totals["contracts_seen"] += seen
                    totals["contracts_retained"] += len(rows)
                except Exception as error:
                    conn.rollback()
                    if isinstance(error, psycopg2.Error):
                        if run_id is not None:
                            with conn.cursor() as cur:
                                cur.execute("""
                                    UPDATE options_data.collection_runs
                                    SET finished_at=now(), status='failed', error_text=%s
                                    WHERE run_id=%s
                                """, (f"database error at {symbol}/{expiry_text}: {error}"[:4000], run_id))
                            conn.commit()
                        raise
                    item["errors"].append(f"{expiry_text}: {type(error).__name__}: {error}")
                    errors.append(f"{symbol}/{expiry_text}")
                    if rate_limited(error):
                        hit_limit = True
                        break
                time.sleep(args.request_delay)
            if not item["errors"]:
                totals["symbols_completed"] += 1
            print(f"{symbol}: {item['contracts_retained']}/{item['contracts_seen']} retained; "
                  f"{item['expirations_completed']}/{item['expirations_seen']} expirations")
            if run_id is not None:
                with conn.cursor() as cur:
                    cur.execute("UPDATE options_data.collection_runs SET symbol_stats=%s::jsonb WHERE run_id=%s",
                                (json.dumps(stats), run_id))
                conn.commit()
            if hit_limit:
                break
        status = "rate_limited" if hit_limit else "partial" if errors else "complete"
        if run_id is not None:
            with conn.cursor() as cur:
                after = table_bytes(cur)
                cur.execute("""
                    UPDATE options_data.collection_runs SET finished_at=now(),status=%s,
                      symbols_completed=%s,expirations_seen=%s,expirations_completed=%s,
                      contracts_seen=%s,contracts_retained=%s,symbol_stats=%s::jsonb,
                      error_text=%s,table_bytes_after=%s WHERE run_id=%s
                """, (status, totals["symbols_completed"], totals["expirations_seen"],
                      totals["expirations_completed"], totals["contracts_seen"],
                      totals["contracts_retained"], json.dumps(stats),
                      "; ".join(errors)[:4000] if errors else None, after, run_id))
            conn.commit()
        print(f"run={run_id or 'dry-run'} status={status} {totals} errors={errors[:5]}")
        return 0 if status == "complete" else 1
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--market-hours", action="store_true")
    parser.add_argument("--symbols", nargs="*", help="Subset of current DJIA names for a sample")
    parser.add_argument("--max-expirations", type=int, default=0, help="0 means every Yahoo expiration")
    parser.add_argument("--minimum-oi", type=int, default=100)
    parser.add_argument("--maximum-spread", type=float, default=0.25)
    parser.add_argument("--request-delay", type=float, default=0.3)
    parser.add_argument("--dry-run", action="store_true", help="Fetch and count but do not store quotes")
    parser.add_argument("--db-config", help="Path to database.ini when outside the repository root")
    args = parser.parse_args()
    if args.minimum_oi < 0 or not 0 <= args.maximum_spread <= 2 or args.request_delay < 0 or args.max_expirations < 0:
        parser.error("filter and delay arguments must be non-negative; spread must be <= 2")
    if args.start_date and args.end_date and args.end_date < args.start_date:
        parser.error("end date is before start date")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
