import os
import yaml
import click
import pandas as pd
import logging
from datetime import timedelta
from dotenv import load_dotenv
from tiingo import TiingoClient
import yfinance as yf
from fredapi import Fred
from db_utils import get_connection, run_sql
from db_config import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API key from .env
load_dotenv()
TIINGO_API_KEY = os.environ['TIINGO_API_KEY']

# Initialize Tiingo Client
ti_client = TiingoClient({'session': True, 'api_key': TIINGO_API_KEY})

### ✅ Helper Functions ###
def get_latest_date_for_symbol(symbol, table):
    query = f"SELECT MAX(date) FROM price_data.{table} WHERE symbol = %s"
    result = run_sql(query, config(), (symbol,))
    return result.iloc[0, 0] if not result.empty else None

def get_latest_date(table: str, date_col: str = "date"):
    query = f"SELECT MAX({date_col}) FROM {table};"
    result = run_sql(query, config())
    return result.iloc[0,0] if not result.empty else None

def check_metadata_exists(symbol, table):
    query = f"SELECT COUNT(*) FROM price_data.{table} WHERE symbol = %s"
    result = run_sql(query, config(), (symbol,))
    return result.iloc[0, 0] > 0

def save_equity_metadata(symbol):
    metadata = ti_client.get_ticker_metadata(symbol)
    if metadata:
        with get_connection(config()) as (conn, cur):
            cur.execute("""
                INSERT INTO price_data.equities_us (symbol, name, asset_class, exchange, currency, sector, industry)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO NOTHING;
            """, (
                symbol,
                metadata.get("name"),
                "stock",
                metadata.get("exchangeCode"),
                metadata.get("currency"),
                metadata.get("sector"),
                metadata.get("industry")
            ))
            conn.commit()

def save_crypto_metadata(symbol):
    with get_connection(config()) as (conn, cur):
        cur.execute("""
            INSERT INTO price_data.crypto (symbol, name, exchange)
            VALUES (%s, %s, %s)
            ON CONFLICT (symbol) DO NOTHING;
        """, (symbol, symbol, "Yahoo Finance"))
        conn.commit()

### ✅ Fetching Data ###
def fetch_equity_prices(symbol):
    latest_date = get_latest_date_for_symbol(symbol, "equities_us_daily")
    start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d") if latest_date else "1980-01-01"
    data = ti_client.get_dataframe(symbol, frequency="daily", startDate=start_date)
    data["symbol"] = symbol
    return data.reset_index().rename(columns={"index": "date"})

def fetch_crypto_prices(symbol):
    latest_date = get_latest_date_for_symbol(symbol, "crypto_daily")
    start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d") if latest_date else "2010-01-01"
    data = yf.download(symbol, start=start_date)
    data.columns = data.columns.get_level_values(0)
    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    data["symbol"] = symbol
    data = data.reset_index().rename(columns={"Date": "date"})
    return data

def fetch_vix_from_FRED():
    fred_api = Fred(api_key=os.environ['FRED_API_KEY'])
    return fred_api.get_series("VIXCLS").dropna()

def download_treasury_yield_curve(yyyy, n_latest=7):
    csv_url_yyyy = f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{yyyy}/all?type=daily_treasury_yield_curve&field_tdr_date_value={yyyy}&page&_format=csv'
    df = pd.read_csv(csv_url_yyyy, nrows=n_latest, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df

### ✅ Storing Data ###
def save_equity_prices(df):
    with get_connection(config()) as (conn, cur):
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO price_data.equities_us_daily (symbol, date, open, high, low, close, volume, adj_close)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO UPDATE SET
                open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                close = EXCLUDED.close, volume = EXCLUDED.volume, adj_close = EXCLUDED.adj_close;
            """, (row['symbol'], row['date'], row['open'], row['high'], row['low'], row['close'], row.get('volume'), row.get('adjClose')))
        conn.commit()

def save_crypto_prices(df):
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
                row['date'].to_pydatetime(),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row.get('volume', 0)) if pd.notna(row.get('volume')) else None
            ))
        conn.commit()

def save_treasury_cmt(df):
    df = df.copy().reset_index()
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df_long = df.melt(id_vars="Date", var_name="Tenor", value_name="Rate")
    df_long = df_long.dropna(subset=["Rate"])
    with get_connection(config()) as (conn, cur):
        for _, row in df_long.iterrows():
            cur.execute("""
                INSERT INTO interest_rates.treasury_cmt (date, tenor, rate)
                VALUES (%s, %s, %s)
                ON CONFLICT (date, tenor) DO UPDATE SET rate = EXCLUDED.rate;
            """, (row["Date"].date(), row["Tenor"], round(float(row["Rate"]), 2)))
        conn.commit()
    logging.info(f"Saved {len(df_long)} UST entries")

def save_index_to_db(series: pd.Series, table: str):
    if series.empty:
        return
    df = series.reset_index()
    df.columns = ["date", "value"]
    df = df.dropna(subset=["date", "value"])
    with get_connection(config()) as (conn, cur):
        for _, row in df.iterrows():
            cur.execute(f"""
                INSERT INTO {table} (date, value)
                VALUES (%s, %s)
                ON CONFLICT (date) DO UPDATE SET value = EXCLUDED.value;
            """, (row["date"].to_pydatetime(), float(row["value"])))
        conn.commit()

### ✅ Main ###
@click.command()
@click.option('--assets_file', default="macro_assets.yaml", help="Path to assets YAML file.")
def main(assets_file):
    logging.info(f"🔹 Loading assets from {assets_file}...")
    with open(assets_file, "r") as f:
        assets = yaml.safe_load(f)
    equity_symbols = assets.get("equities", [])
    crypto_symbols = assets.get("crypto", [])

    logging.info("🔹 Processing Equities...")
    for symbol in equity_symbols:
        if not check_metadata_exists(symbol, "equities_us"):
            logging.info(f"Fetching metadata for {symbol}...")
            save_equity_metadata(symbol)
        logging.info(f"Fetching prices for {symbol}...")
        df = fetch_equity_prices(symbol)
        if not df.empty:
            save_equity_prices(df)
            logging.info(f"Stored {symbol} equity data.")

    logging.info("🔹 Processing Crypto...")
    for symbol in crypto_symbols:
        if not check_metadata_exists(symbol, "crypto"):
            logging.info(f"Fetching metadata for {symbol}...")
            save_crypto_metadata(symbol)
        logging.info(f"Fetching prices for {symbol}...")
        df = fetch_crypto_prices(symbol)
        if not df.empty:
            save_crypto_prices(df)
            logging.info(f"Stored {symbol} crypto data.")

    logging.info("🔹 Updating VIX...")
    vix_latest = pd.to_datetime(get_latest_date("market_indices.vix"))
    vix_series = fetch_vix_from_FRED()
    if vix_latest:
        vix_series = vix_series[vix_series.index > vix_latest]
    save_index_to_db(vix_series, "market_indices.vix")
    logging.info(f"Stored {len(vix_series)} new VIX entries.")

    logging.info("🔹 Updating Treasury CMT...")
    cmt_latest = get_latest_date("interest_rates.treasury_cmt", date_col="date")
    logging.info(f'Latest cmt data stored is for {cmt_latest}')
    today = pd.Timestamp.today().date()
    years = list(range((cmt_latest.year if cmt_latest else 2000), today.year + 1))
    for y in years:
        logging.info(f'getting UST history for {y}')
        df = download_treasury_yield_curve(y, n_latest=999)
        logging.info(f'{df.shape[0]} rows downloaded')
        df = df.rename(columns=lambda x: x.strip())
        # df = df.rename(columns={df.columns[0]: "Date"})
        # df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        # df = df.dropna(subset=["Date"])
        if cmt_latest:
            df = df[df.index > pd.to_datetime(cmt_latest)]
            logging.info(f'{df.shape[0]} rows needs to be saved')
        save_treasury_cmt(df)

    logging.info("✅ Data retrieval and storage complete.")

if __name__ == "__main__":
    main()
