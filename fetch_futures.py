"""
fetch_futures.py

Fetch continuous futures daily OHLCV data from Yahoo Finance and store in
price_data.futures / price_data.futures_daily.

Yahoo Finance uses the "{symbol}=F" convention for continuous (front-month rolled)
futures contracts.

Usage
-----
    python fetch_futures.py                  # full refresh of all available symbols
    python fetch_futures.py --symbols CL NG  # specific symbols only
    python fetch_futures.py --incremental    # only fetch since last stored date
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import yfinance as yf

sys.path.insert(0, '/Users/captain/projects/market_data')
sys.path.insert(0, '/Users/captain/projects/strader')
from data_loader import config


# ── Contract specs ────────────────────────────────────────────────────────────
# symbol → (name, sector, contract_size, exchange)
FUTURES_SPECS: dict[str, tuple[str, str, float, str]] = {
    # Energy
    "CL":  ("Crude Oil WTI",           "Energy",       1000,    "NYMEX"),
    "BZ":  ("Brent Crude",             "Energy",       1000,    "NYMEX"),
    "NG":  ("Natural Gas",             "Energy",       10000,   "NYMEX"),
    "RB":  ("RBOB Gasoline",           "Energy",       42000,   "NYMEX"),
    "HO":  ("NY Harbor ULSD",          "Energy",       42000,   "NYMEX"),
    # Micro Energy (1/10 contract size)
    "MCL": ("Micro Crude Oil WTI",     "Energy",       100,     "NYMEX"),
    # Metals
    "GC":  ("Gold",                    "Metals",       100,     "COMEX"),
    "SI":  ("Silver",                  "Metals",       5000,    "COMEX"),
    "HG":  ("Copper",                  "Metals",       25000,   "COMEX"),
    "PA":  ("Palladium",               "Metals",       100,     "NYMEX"),
    "PL":  ("Platinum",                "Metals",       50,      "NYMEX"),
    # Micro Metals
    "MGC": ("Micro Gold",              "Metals",       10,      "COMEX"),
    # Agriculture — NOTE: Yahoo quotes grains/softs in cents, so contract_size
    # = physical_size × 0.01 to convert cents → dollars.
    # Grains: 5000 bu × $0.01/cent = 50
    "ZC":  ("Corn",                    "Agriculture",  50,      "CBOT"),
    "ZW":  ("Wheat",                   "Agriculture",  50,      "CBOT"),
    "ZS":  ("Soybeans",                "Agriculture",  50,      "CBOT"),
    "ZM":  ("Soybean Meal",            "Agriculture",  100,     "CBOT"),  # $/short ton on Yahoo
    "ZL":  ("Soybean Oil",             "Agriculture",  600,     "CBOT"),  # 60000 lb × $0.01
    # Softs (all cents/lb on Yahoo)
    "SB":  ("Sugar #11",               "Agriculture",  1120,    "ICE"),   # 112000 lb × $0.01
    "KC":  ("Coffee",                  "Agriculture",  375,     "ICE"),   # 37500 lb × $0.01
    "CC":  ("Cocoa",                   "Agriculture",  10,      "ICE"),   # $/metric ton on Yahoo
    "CT":  ("Cotton #2",               "Agriculture",  500,     "ICE"),   # 50000 lb × $0.01
    "OJ":  ("Orange Juice",            "Agriculture",  150,     "ICE"),   # 15000 lb × $0.01
    # Livestock (cents/lb on Yahoo)
    "LE":  ("Live Cattle",             "Agriculture",  400,     "CME"),   # 40000 lb × $0.01
    "HE":  ("Lean Hogs",               "Agriculture",  400,     "CME"),   # 40000 lb × $0.01
    "GF":  ("Feeder Cattle",           "Agriculture",  500,     "CME"),   # 50000 lb × $0.01
    "LBS": ("Lumber",                  "Agriculture",  110,     "CME"),   # 110000 bd-ft / 1000 (price per mbf)
    # Equity Index (standard)
    "ES":  ("E-Mini S&P 500",          "Equity Index", 50,      "CME"),
    "NQ":  ("E-Mini Nasdaq-100",       "Equity Index", 20,      "CME"),
    "YM":  ("Dow Mini",                "Equity Index", 5,       "CBOT"),
    "RTY": ("E-Mini Russell 2000",     "Equity Index", 50,      "CME"),
    "NKD": ("Nikkei 225 Dollar",       "Equity Index", 5,       "CME"),
    # Micro Equity Index ($5 or $2 multiplier)
    "MES": ("Micro E-Mini S&P 500",    "Equity Index", 5,       "CME"),
    "M2K": ("Micro E-Mini Russ 2000",  "Equity Index", 5,       "CME"),
    "MNQ": ("Micro E-Mini Nasdaq-100", "Equity Index", 2,       "CME"),
    # Interest Rates — price is % of par; contract_size = face_value / 100
    # ZN/ZF: $100k face → size=1000; ZT: $200k face → size=2000
    # ZB/TN/UB: $100k face → size=1000
    "ZN":  ("10-Year T-Note",          "Interest Rate",1000,    "CBOT"),
    "ZF":  ("5-Year T-Note",           "Interest Rate",1000,    "CBOT"),
    "ZT":  ("2-Year T-Note",           "Interest Rate",2000,    "CBOT"),
    "ZB":  ("30-Year T-Bond",          "Interest Rate",1000,    "CBOT"),
    "TN":  ("Ultra 10-Year T-Note",    "Interest Rate",1000,    "CBOT"),
    "UB":  ("Ultra T-Bond",            "Interest Rate",1000,    "CBOT"),
    # Cryptocurrency
    "BTC": ("Bitcoin",                 "Cryptocurrency",5,      "CME"),
    # FX Futures (standard size) — Yahoo ticker: "6E=F", "6B=F" etc.
    "6E":  ("Euro FX",                 "Currency",      125000, "CME"),
    "6B":  ("British Pound",           "Currency",      62500,  "CME"),
    "6A":  ("Australian Dollar",       "Currency",      100000, "CME"),
    "6C":  ("Canadian Dollar",         "Currency",      100000, "CME"),
    "6J":  ("Japanese Yen",            "Currency",      12500000,"CME"),
    "6S":  ("Swiss Franc",             "Currency",      125000, "CME"),
    "6N":  ("New Zealand Dollar",      "Currency",      100000, "CME"),
    # Micro FX Futures (1/10 contract size) — Yahoo ticker: "M6E=F" etc.
    "M6E": ("Micro Euro FX",           "Currency",      12500,  "CME"),
    "M6B": ("Micro British Pound",     "Currency",      6250,   "CME"),
    "M6A": ("Micro Australian Dollar", "Currency",      10000,  "CME"),
    "MCD": ("Micro Canadian Dollar",   "Currency",      10000,  "CME"),
    "MJY": ("Micro Japanese Yen",      "Currency",      1250000,"CME"),
    "MSF": ("Micro Swiss Franc",       "Currency",      12500,  "CME"),
}

# FX spot pairs: not available as futures on Yahoo, but "EURUSD=X" format works.
# We store them in the same tables using the base currency pair as symbol
# and yf_symbol = "EURUSD=X" etc.  Contract size = 1 (spot rate, no multiplier).
FX_SPOT_SPECS: dict[str, tuple[str, str, str]] = {
    "EURUSD": ("Euro / US Dollar",          "Currency", "EURUSD=X"),
    "GBPUSD": ("British Pound / US Dollar", "Currency", "GBPUSD=X"),
    "JPYUSD": ("Japanese Yen / US Dollar",  "Currency", "JPYUSD=X"),
    "AUDUSD": ("Australian Dollar / USD",   "Currency", "AUDUSD=X"),
    "CADUSD": ("Canadian Dollar / USD",     "Currency", "CADUSD=X"),
    "CHFUSD": ("Swiss Franc / US Dollar",   "Currency", "CHFUSD=X"),
    "NZDUSD": ("New Zealand Dollar / USD",  "Currency", "NZDUSD=X"),
    "USDDX":  ("US Dollar Index",           "Currency", "DX-Y.NYB"),  # ICE DXY
}

# Yahoo Finance availability confirmed (checked 2026-04-28)
# All FUTURES_SPECS symbols confirmed available via =F (standard FX: 6E=F etc.)
# Micro FX: M6E=F, M6B=F, M6A=F, MCD=F, MJY=F, MSF=F
YF_AVAILABLE = {s for s in FUTURES_SPECS}


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_conn(cfg: dict):
    keys = ('host', 'port', 'database', 'user', 'password')
    return psycopg2.connect(**{k: v for k, v in cfg.items() if k in keys})


def upsert_specs(symbols: list[str], cfg: dict):
    """Insert / update contract specs into price_data.futures."""
    conn = get_conn(cfg)
    cur = conn.cursor()
    for sym in symbols:
        if sym in FUTURES_SPECS:
            name, sector, contract_size, exchange = FUTURES_SPECS[sym]
            # Standard FX futures on Yahoo use numeric-prefix format: 6E=F not E6=F
            yf_sym = f"{sym}=F"
        elif sym in FX_SPOT_SPECS:
            name, sector, yf_sym = FX_SPOT_SPECS[sym]
            contract_size, exchange = 1, 'FOREX'
        else:
            continue
        cur.execute("""
            INSERT INTO price_data.futures
                (symbol, name, sector, contract_size, yf_symbol, exchange, currency)
            VALUES (%s, %s, %s, %s, %s, %s, 'USD')
            ON CONFLICT (symbol) DO UPDATE SET
                name          = EXCLUDED.name,
                sector        = EXCLUDED.sector,
                contract_size = EXCLUDED.contract_size,
                yf_symbol     = EXCLUDED.yf_symbol,
                exchange      = EXCLUDED.exchange
        """, (sym, name, sector, contract_size, yf_sym, exchange))
    conn.commit()
    conn.close()


def get_last_dates(symbols: list[str], cfg: dict) -> dict[str, date]:
    """Return last stored date per symbol."""
    conn = get_conn(cfg)
    cur = conn.cursor()
    cur.execute("""
        SELECT symbol, MAX(date) FROM price_data.futures_daily
        WHERE symbol = ANY(%s)
        GROUP BY symbol
    """, (symbols,))
    result = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return result


def upsert_daily(rows: list[dict], cfg: dict):
    """Bulk upsert OHLCV rows into price_data.futures_daily."""
    if not rows:
        return
    conn = get_conn(cfg)
    cur = conn.cursor()
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO price_data.futures_daily
            (symbol, date, open, high, low, close, adj_close, volume)
        VALUES %s
        ON CONFLICT (symbol, date) DO UPDATE SET
            open      = EXCLUDED.open,
            high      = EXCLUDED.high,
            low       = EXCLUDED.low,
            close     = EXCLUDED.close,
            adj_close = EXCLUDED.adj_close,
            volume    = EXCLUDED.volume
        """,
        [(r['symbol'], r['date'], r['open'], r['high'], r['low'],
          r['close'], r['adj_close'], r['volume']) for r in rows],
        page_size=1000,
    )
    conn.commit()
    conn.close()


# ── Fetch from Yahoo ──────────────────────────────────────────────────────────

def fetch_symbol(
    symbol: str,
    start: str | None = None,
    verbose: bool = True,
    yf_ticker_override: str | None = None,
) -> pd.DataFrame:
    """
    Download continuous futures (or FX spot) data from Yahoo Finance.
    Returns DataFrame with columns: open, high, low, close, adj_close, volume.
    Index: date (date objects).
    """
    yf_sym = yf_ticker_override or f"{symbol}=F"
    kwargs = dict(auto_adjust=False, progress=False)
    if start:
        kwargs['start'] = start
    else:
        kwargs['period'] = 'max'

    try:
        df = yf.download(yf_sym, **kwargs)
    except Exception as e:
        if verbose:
            print(f"  {symbol}: download error — {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if present (yfinance ≥0.2)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    col_map = {'adj close': 'adj_close', 'open': 'open', 'high': 'high',
               'low': 'low', 'close': 'close', 'volume': 'volume'}
    df = df.rename(columns=col_map)
    needed = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    df = df[needed].dropna(subset=['close'])
    df.index = pd.to_datetime(df.index).normalize().date
    df = df[~df.index.duplicated(keep='last')]
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run_fetch(
    symbols: list[str] | None = None,
    incremental: bool = True,
    include_fx: bool = True,
    verbose: bool = True,
    sleep: float = 0.3,
):
    cfg = config()
    if symbols is None:
        symbols = sorted(YF_AVAILABLE)
        if include_fx:
            symbols += sorted(FX_SPOT_SPECS)

    # Ensure all specs are in DB
    upsert_specs(symbols, cfg)
    if verbose:
        print(f"Fetching {len(symbols)} symbols  (incremental={incremental})")

    last_dates = get_last_dates(symbols, cfg) if incremental else {}

    total_rows = 0
    for sym in symbols:
        # Determine yahoo ticker
        if sym in FUTURES_SPECS:
            yf_ticker = f"{sym}=F"
        elif sym in FX_SPOT_SPECS:
            yf_ticker = FX_SPOT_SPECS[sym][2]
        else:
            continue

        last = last_dates.get(sym)
        if incremental and last:
            start = (last + timedelta(days=1)).isoformat()
        else:
            start = None

        df = fetch_symbol(sym, start=start, verbose=verbose, yf_ticker_override=yf_ticker)
        if df.empty:
            if verbose:
                print(f"  {sym}: no data")
            continue

        rows = [
            {
                'symbol':    sym,
                'date':      idx,
                'open':      None if pd.isna(row['open'])      else float(row['open']),
                'high':      None if pd.isna(row['high'])      else float(row['high']),
                'low':       None if pd.isna(row['low'])       else float(row['low']),
                'close':     None if pd.isna(row['close'])     else float(row['close']),
                'adj_close': None if pd.isna(row['adj_close']) else float(row['adj_close']),
                'volume':    None if pd.isna(row['volume'])    else int(row['volume']),
            }
            for idx, row in df.iterrows()
        ]
        upsert_daily(rows, cfg)
        total_rows += len(rows)

        if verbose:
            print(f"  {sym:6s}  {df.index[0]}→{df.index[-1]}  {len(rows):5d} rows stored")

        time.sleep(sleep)

    if verbose:
        print(f"\nDone. {total_rows:,} rows upserted across {len(symbols)} symbols.")


def get_futures_prices(
    symbols: list[str],
    start_date: str = '2000-01-01',
) -> pd.DataFrame:
    """
    Load futures daily OHLCV from DB.
    Returns wide DataFrame with MultiIndex columns (field, symbol).
    Mirrors get_equity_prices() interface for drop-in compatibility.
    """
    cfg = config()
    from db_utils import run_sql
    syms_str = "', '".join(symbols)
    df = run_sql(f"""
        SELECT symbol, date, open, high, low, close, adj_close, volume
        FROM price_data.futures_daily
        WHERE symbol IN ('{syms_str}')
          AND date >= '{start_date}'
        ORDER BY date, symbol
    """, cfg)
    if df.empty:
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date', 'symbol'])
    return df.unstack('symbol')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='*', help='Specific symbols to fetch')
    parser.add_argument('--full', action='store_true', help='Full refresh (ignore last date)')
    parser.add_argument('--list', action='store_true', help='List available symbols and exit')
    args = parser.parse_args()

    if args.list:
        print(f"{'Symbol':8s}  {'Sector':15s}  {'Name'}")
        for sym, (name, sector, size, exch) in sorted(FUTURES_SPECS.items(), key=lambda x: x[1][1]):
            avail = '✓' if sym in YF_AVAILABLE else '✗'
            print(f"  {avail} {sym:6s}  {sector:15s}  {name}")
        sys.exit(0)

    run_fetch(
        symbols=args.symbols if args.symbols else None,
        incremental=not args.full,
        verbose=True,
    )
