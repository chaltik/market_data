import os
import yaml
import click
import pandas as pd
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import logging
from datetime import timedelta
from dotenv import load_dotenv
from tiingo import TiingoClient
import requests
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

TIINGO_CRYPTO_URL='https://api.tiingo.com/tiingo/crypto/prices'

utc=pytz.UTC

### ✅ Helper Functions ###
def get_latest_date_for_symbol(symbol, table):
    query = f"SELECT MAX(date) FROM price_data.{table} WHERE symbol = %s"
    result = run_sql(query, config(), (symbol,))
    return result.iloc[0, 0] if not result.empty else None

def get_latest_date(table: str, date_col: str = "date"):
    query = f"SELECT MAX({date_col}) FROM {table};"
    result = run_sql(query, config())
    return result.iloc[0,0] if not result.empty else None

def check_metadata_exists(symbol, table, schema='price_data'):
    query = f"SELECT COUNT(*) FROM {schema}.{table} WHERE symbol = %s"
    result = run_sql(query, config(), (symbol,))
    return result.iloc[0, 0] > 0

def check_eco_metadata_exists(series_name):
    query = f"SELECT COUNT(*) FROM eco.release_content WHERE series_name = %s"
    result = run_sql(query, config(), (series_name,))
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
        """, (symbol, symbol, "TIINGO"))
        conn.commit()

def save_eco_metadata(
    release_id: int,
    release_name: str,
    series_name: str,
    bls_series_id: str | None = None,
    fred_series_id: str | None = None,
    source: str | None = None,
    country_code: str | None = None,
):
    """
    Insert one row into eco.release_content.
    If (release_name, series_name) already exists, do nothing.
    """
    with get_connection(config()) as (conn, cur):
        cur.execute(
            """
            INSERT INTO eco.release_content
                (release_id, release_name, series_name, bls_series_id, fred_series_id, source, country_code)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (release_name, series_name) DO NOTHING;
            """,
            (
                release_id,
                release_name,
                series_name,
                bls_series_id,
                fred_series_id,
                source,
                country_code,
            ),
        )
        conn.commit()

### ✅ Fetching Data ###
def fetch_equity_prices(symbol,start_date=None):
    latest_date = get_latest_date_for_symbol(symbol, "equities_us_daily")
    if latest_date is None:
        return pd.DataFrame()
    if start_date is None:
        start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d") if latest_date else "1980-01-01"
    try:
        data = ti_client.get_dataframe(symbol, frequency="daily", startDate=start_date)
        data["symbol"] = symbol
        return data.reset_index().rename(columns={"index": "date"})
    except Exception as e:
        logging.info(f'No rows received from tiingo after start_date = {start_date}')
        return pd.DataFrame()


def fetch_crypto_prices(symbol):
    """
    Fetch crypto price data from Tiingo for a given symbol, starting from the latest date in the DB.
    Returns a DataFrame with columns: date, symbol, open, high, low, close, volume
    """
    latest_date = get_latest_date_for_symbol(symbol, "crypto_daily")
    start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d") if latest_date else "2010-01-01"
    tk = os.environ['TIINGO_API_KEY']
    url = (
        f"{TIINGO_CRYPTO_URL}?"
        f"tickers={symbol}&startDate={start_date}&resampleFreq=1day&token={tk}"
    )

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if not data:
        return pd.DataFrame()

    records = []
    for entry in data:
        ticker = entry['ticker']
        for price in entry['priceData']:
            records.append({
                'date': pd.to_datetime(price['date']).tz_convert(utc),
                'symbol': ticker.upper(),
                'open': price['open'],
                'high': price['high'],
                'low': price['low'],
                'close': price['close'],
                'volume': price.get('volume', None)
            })

    df = pd.DataFrame(records)
    df = df.sort_values('date').reset_index(drop=True)
    return df


# https://api.tiingo.com/tiingo/crypto/prices?tickers=atomusd&startDate=2019-01-02&resampleFreq=1day
# def fetch_crypto_prices(symbol,start_date_str,freq="1Day"):
#     request_string="https://api.tiingo.com/tiingo/crypto/prices?tickers="+ticker+"&resampleFreq="+freq+"&token="+tk
#     request_string=request_string+"&startDate="+start_date_str
#     print(request_string)
#     data=pd.read_json(request_string).set_index('ticker')
#     tkrs=ticker.split(",")
#     data_out_list=[]
#     for t in data.index:
#         ds=pd.DataFrame(data.loc[t]['priceData'])
#         ds.loc[:,'ticker']=t
#         #ds.loc[:,'date']=pd.to_datetime(ds['date'])
#         if start_date_str is not None:
#             start_date=datetime.strptime(start_date_str,'%Y-%m-%d')
#         else:
#             start_date=ds.date.min()
#         start_date=start_date.replace(tzinfo=utc)
#         #while ds.date.min()>start_date:
#         #    end_date=ds.date.min()
#         #    request_string_ext=request_string+"&endDate="+end_date.strftime('%Y-%m-%d')
#         #    data1=pd.read_json(request_string_ext)
#         #    ds1=pd.DataFrame(data1['priceData'][i])
#         #    ds1.loc[:,'date']=pd.to_datetime(ds1['date'])
#         #    ds=pd.concat([ds,ds1],axis=0)
#         #    ds.drop_duplicates(inplace=True)
#         data_out_list.append(ds)
#     if len(data_out_list)>1:
#         return pd.concat(data_out_list,axis=0)
#     else:
#         return data_out_list[0]
            

# def get_tiingo_equity_multiple(tickers,start_date='1980-01-01',end_date=None):
#     return pd.concat([get_tiingo_equity(ticker,start_date=start_date,end_date=end_date) for ticker in tickers],axis=0)

# def get_tiingo_crypto_multiple(tickers,start_date_str='1980-01-01'):
#     return pd.concat([get_tiingo_crypto(ticker,start_date_str=start_date_str) for ticker in tickers],axis=0)

# def fetch_crypto_prices(symbol):
#     latest_date = get_latest_date_for_symbol(symbol, "crypto_daily")
#     start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d") if latest_date else "2010-01-01"
#     data = yf.download(symbol, start=start_date)
#     data.columns = data.columns.get_level_values(0)
#     data = data.rename(columns={
#         "Open": "open",
#         "High": "high",
#         "Low": "low",
#         "Close": "close",
#         "Volume": "volume"
#     })
#     data["symbol"] = symbol
#     data = data.reset_index().rename(columns={"Date": "date"})
#     return data

def fetch_vix_from_FRED():
    fred_api = Fred(api_key=os.environ['FRED_API_KEY'])
    return fred_api.get_series("VIXCLS").dropna()

def download_treasury_yield_curve(yyyy, n_latest=7):
    csv_url_yyyy = f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{yyyy}/all?type=daily_treasury_yield_curve&field_tdr_date_value={yyyy}&page&_format=csv'
    df = pd.read_csv(csv_url_yyyy, nrows=n_latest, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df

GSCPI_URL = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"
GSCPI_RELEASE_NAME = "Global Supply Chain Pressure Index"
GSCPI_SERIES_NAME = "GSCPI"

US_BD = CustomBusinessDay(calendar=USFederalHolidayCalendar())

def download_gscpi():
    """
    Download the GSCPI data from the FRBNY website.
    Returns a DataFrame with columns: reference_date, value.
    """
    df = pd.read_excel(GSCPI_URL, sheet_name='GSCPI Monthly Data', header=None, skiprows=5, usecols=[0,1], names=['reference_date', 'gscpi'])
    df['reference_date'] = pd.to_datetime(df['reference_date'])
    df = df.dropna()
    return df


def compute_release_date(reference_date):
    """
    Given a reference date, return the release date (4th business day of next month).
    """
    first_of_next_month = reference_date + pd.offsets.MonthBegin(1)
    fourth_bday = first_of_next_month + 3 * US_BD  # 0-indexed so +3 gives the 4th business day
    return fourth_bday


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
            """, (row['symbol'], row['date'].date(), row['open'], row['high'], row['low'], row['close'], row.get('volume'), row.get('adjClose')))
        conn.commit()


def save_crypto_prices(df):
    with get_connection(config()) as (conn, cur):
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO price_data.crypto_daily (
                    symbol, date, open, high, low, close, volume, volume_notional, trades_done
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    volume_notional = EXCLUDED.volume_notional,
                    trades_done = EXCLUDED.trades_done;
            """, (
                row['symbol'],
                row['date'].to_pydatetime(),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row.get('volume', 0)) if pd.notna(row.get('volume')) else None,
                float(row.get('volumeNotional', 0)) if pd.notna(row.get('volumeNotional')) else None,
                int(row.get('tradesDone', 0)) if pd.notna(row.get('tradesDone')) else None
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

def save_gscpi_to_db(df):
    """
    Save GSCPI data to eco.release_schedule and eco.release_data tables.
    """
    df = df.copy()
    df['release_name'] = GSCPI_RELEASE_NAME
    df['series_name'] = GSCPI_SERIES_NAME
    df['release_date'] = df['reference_date'].apply(compute_release_date)
    df['release_id'] = df['release_date'].apply(lambda d: int(d.strftime('%Y%m%d')))

    with get_connection(config()) as (conn, cur):
        for _, row in df.iterrows():
            # Save into release_schedule
            cur.execute("""
                INSERT INTO eco.release_schedule (release_id, release_name, release_date, reference_date)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (release_name, reference_date) DO UPDATE 
                SET release_date = EXCLUDED.release_date;
            """, (
                row['release_id'],
                row['release_name'],
                row['release_date'],
                row['reference_date']
            ))

            # Save into release_data
            cur.execute("""
                INSERT INTO eco.release_data (release_id, release_name, series_name, release_date, reference_date, value)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (release_name, series_name, reference_date) DO UPDATE
                SET value = EXCLUDED.value;
            """, (
                row['release_id'],
                row['release_name'],
                row['series_name'],
                row['release_date'],
                row['reference_date'],
                row['gscpi']
            ))

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
        print(df.shape)
        logging.info(f"Obtained {df.shape[0]} records")
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

    logging.info("🔹 Updating GSCPI...")
    gscpi_series = download_gscpi()
    if not gscpi_series.empty:
        if not check_eco_metadata_exists("GCSPI"):
            logging.info("Inserting metadata for GSCPI")
            save_eco_metadata(
            release_id=1,
            release_name=GSCPI_RELEASE_NAME,
            series_name=GSCPI_SERIES_NAME,
            source='FRBNY',
            country_code='USA',
            )
        save_gscpi_to_db(gscpi_series)
        logging.info(f"Stored {len(gscpi_series)} GSCPI entries.")
    else:
        logging.info("No new GSCPI data to store.")
        
    logging.info("✅ Data retrieval and storage complete.")

if __name__ == "__main__":
    main()
