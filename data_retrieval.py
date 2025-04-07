import os
import yaml
import click
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tiingo import TiingoClient
import yfinance as yf
from db_utils import get_connection, run_sql
from db_config import config

# Load API key from .env
load_dotenv()
TIINGO_API_KEY = os.environ['TIINGO_API_KEY']

# Initialize Tiingo Client
ti_client = TiingoClient({'session':True, 'api_key': TIINGO_API_KEY})

### ✅ Helper Functions ###
def get_latest_date(symbol, table):
    """Fetches the latest available date for a symbol in the database."""
    query = f"SELECT MAX(date) FROM price_data.{table} WHERE symbol = %s"
    result = run_sql(query, config(), (symbol,))
    return result.iloc[0, 0] if not result.empty else None

def check_metadata_exists(symbol, table):
    """Checks if metadata exists for a given symbol."""
    query = f"SELECT COUNT(*) FROM price_data.{table} WHERE symbol = %s"
    result = run_sql(query, config(), (symbol,))
    return result.iloc[0, 0] > 0

def save_equity_metadata(symbol):
    """Fetches and stores metadata for a given equity."""
    metadata = ti_client.get_ticker_metadata(symbol)
    if metadata:
        with get_connection(config()) as (conn, cur):
            cur.execute("""
                INSERT INTO price_data.equities_us (symbol, name, asset_class, exchange, currency, sector, industry)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO NOTHING;
            """, (symbol, metadata.get("name"), "stock", metadata.get("exchangeCode"),
                metadata.get("currency"), metadata.get("sector"), metadata.get("industry")))
            conn.commit()

def save_crypto_metadata(symbol):
    """Stores metadata for a given crypto asset (manual entry)."""
    with get_connection(config()) as (conn, cur):
        cur.execute("""
            INSERT INTO price_data.crypto (symbol, name, exchange)
            VALUES (%s, %s, %s)
            ON CONFLICT (symbol) DO NOTHING;
        """, (symbol, symbol, "Yahoo Finance"))
        conn.commit()


### ✅ Fetching Data ###
def fetch_equity_prices(symbol):
    """Fetches historical or incremental price data for an equity."""
    latest_date = get_latest_date(symbol, "equities_us_daily")
    start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d") if latest_date else "1980-01-01"

    data = ti_client.get_dataframe(symbol, frequency="daily", startDate=start_date)
    data["symbol"] = symbol
    return data.reset_index().rename(columns={"index": "date"})

def fetch_crypto_prices(symbol):
    """Fetches historical or incremental price data for a crypto asset using Yahoo Finance."""
    latest_date = get_latest_date(symbol, "crypto_daily")
    start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d") if latest_date else "2010-01-01"

    # Fetch data
    data = yf.download(symbol, start=start_date)

    # Fix MultiIndex Columns: KEEP Level 0 (Price Type), DROP Level 1 (Ticker)
    data.columns = data.columns.get_level_values(0)  

    # Ensure the column names are correctly mapped
    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    # Add the symbol column
    data["symbol"] = symbol

    # Reset index and rename Date column
    data = data.reset_index().rename(columns={"Date": "date"})

    return data

### ✅ Storing Data ###
def save_equity_prices(df):
    """Saves equity price data to the database."""
    with get_connection(config()) as (conn, cur):
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO price_data.equities_us_daily (symbol, date, open, high, low, close, volume, adj_close)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO UPDATE SET
                open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                close = EXCLUDED.close, volume = EXCLUDED.volume, adj_close = EXCLUDED.adj_close;
            """, (row['symbol'], row['date'], row['open'], row['high'], row['low'], row['close'], row.get('volume', None), row.get('adjClose', None)))
        conn.commit()

def save_crypto_prices(df):
    """Saves crypto price data to the database."""
    with get_connection(config()) as (conn, cur):
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO price_data.crypto_daily (symbol, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO UPDATE SET
                open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                close = EXCLUDED.close, volume = EXCLUDED.volume;
            """, (
                row['symbol'], 
                row['date'].to_pydatetime(),  # Convert Pandas Timestamp to Python datetime
                float(row['open']),  # Ensure numeric fields are Python floats
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row.get('volume', 0)) if pd.notna(row.get('volume')) else None  # Handle NaNs safely
            ))
        conn.commit()


### ✅ Main Processing ###
@click.command()
@click.option('--assets_file', default="macro_assets.yaml", help="Path to assets JSON file.")
def main(assets_file):
    """Processes equities & crypto, using assets from a JSON file."""
    print(f"🔹 Loading assets from {assets_file}...")

    # Load YAML instead of JSON
    with open(assets_file, "r") as f:
        assets = yaml.safe_load(f)
    if 'equities' in assets:
        equity_symbols = assets["equities"]
    else:
        equity_symbols = []
    if 'crypto' in assets:
        crypto_symbols = assets["crypto"]
    else:
        crypto_symbols = []

    print("🔹 Processing Equities...")
    for symbol in equity_symbols:
        if not check_metadata_exists(symbol, "equities_us"):
            print(f"Fetching metadata for {symbol}...")
            save_equity_metadata(symbol)

        print(f"Fetching prices for {symbol}...")
        df = fetch_equity_prices(symbol)
        if not df.empty:
            save_equity_prices(df)
        print(f"Stored {symbol} equity data.")

    print("🔹 Processing Crypto...")
    for symbol in crypto_symbols:
        if not check_metadata_exists(symbol, "crypto"):
            print(f"Fetching metadata for {symbol}...")
            save_crypto_metadata(symbol)

        print(f"Fetching prices for {symbol}...")
        df = fetch_crypto_prices(symbol)
        if not df.empty:
            save_crypto_prices(df)
        print(f"Stored {symbol} crypto data.")

    print("✅ Data retrieval and storage complete.")

if __name__ == "__main__":
    main()
