import os
import yaml
import click
import pandas as pd
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import pandas_market_calendars as mcal
import logging
from datetime import timedelta, datetime as dtm
from dotenv import load_dotenv
from tiingo import TiingoClient
from tiingo.restclient import RestClientError
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
def utcnow():
    return dtm.now(utc)


### ✅ Helper Functions ###
def get_latest_ts_for_symbol(symbol, table, ts_col='ts'):
    query = f"SELECT MAX({ts_col}) FROM price_data.{table} WHERE symbol = %s"
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

def _is_404(err: Exception) -> bool:
    s = str(err)
    return "404" in s or "Not found" in s or "status code: 404" in s

def _safe_yf_name_exchange(symbol: str) -> tuple[str|None, str|None]:
    try:
        t = yf.Ticker(symbol)
        # prefer lightweight .fast_info when available
        fi = getattr(t, "fast_info", None)
        exch = getattr(fi, "exchange", None) if fi else None
        # .info is heavier; only if needed
        info = getattr(t, "info", None) or {}
        name = info.get("shortName") or info.get("longName")
        if not exch:
            exch = info.get("exchange")
        return name, exch
    except Exception:
        return None, None

def save_equity_metadata(symbol: str):
    """
    Upsert metadata; never raise on 404. Return a dict with what we saved.
    """
    try:
        meta = ti_client.get_ticker_metadata(symbol)  # Tiingo
        if not meta:
            raise ValueError("empty metadata")
        # Normalize a minimal payload for your DB
        payload = {
            "symbol": symbol,
            "name": meta.get("name"),
            "exchange": meta.get("exchangeCode") or meta.get("exchange"),
            "currency": meta.get("quoteCurrency") or meta.get("priceCurrency"),
            "start_date": meta.get("startDate"),
            "end_date": meta.get("endDate"),
            "provider": "tiingo",
            "status": "ok",
            "updated_at": utcnow(),
        }
        _upsert_equity_metadata_row(payload)  # your existing upsert
        return payload

    except (RestClientError, requests.HTTPError) as e:
        if _is_404(e):
            logging.warning("Tiingo metadata 404 for %s; marking as tiingo_not_found and continuing.", symbol)
            # Optional: try to get a friendly name/exchange from yfinance
            name, exch = _safe_yf_name_exchange(symbol)
            payload = {
                "symbol": symbol,
                "name": name,
                "exchange": exch,
                "currency": None,
                "start_date": None,
                "end_date": None,
                "provider": "yfinance_fallback_meta",
                "status": "tiingo_not_found",
                "updated_at": utcnow(),
            }
            _upsert_equity_metadata_row(payload)
            return payload
        else:
            # Non-404: network hiccup or 5xx. Don't crash the batch; log and skip.
            logging.error("Tiingo metadata error for %s: %s", symbol, e)
            payload = {
                "symbol": symbol,
                "name": None,
                "exchange": None,
                "currency": None,
                "start_date": None,
                "end_date": None,
                "provider": "unknown",
                "status": "metadata_error",
                "updated_at": utcnow(),
            }
            _upsert_equity_metadata_row(payload)
            return payload

    except Exception as e:
        logging.error("Metadata fetch unexpected error for %s: %s", symbol, e)
        payload = {
            "symbol": symbol,
            "name": None,
            "exchange": None,
            "currency": None,
            "start_date": None,
            "end_date": None,
            "provider": "unknown",
            "status": "metadata_exception",
            "updated_at": utcnow(),
        }
        _upsert_equity_metadata_row(payload)
        return payload
    
def _upsert_equity_metadata_row(metadata):
    if metadata:
        with get_connection(config()) as (conn, cur):
            cur.execute("""
                INSERT INTO price_data.equities_us (symbol, name, asset_class, exchange, currency, sector, industry)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO NOTHING;
            """, (
                metadata.get("symbol"),
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
def fetch_equity_prices(symbol, start_date=None, end_date=None):
    """
    Return DataFrame with columns:
      ts (America/New_York close), open, high, low, close, volume, adj_close, symbol

    Path A (preferred): Tiingo daily.
      - tiingo 'date' is UTC midnight of trading date -> map to NYSE market_close.
      - Use 'adjClose' if present; else fall back to 'close'.

    Path B (fallback): yfinance history().
      - Index is usually America/New_York at 00:00 wall time for each trading date.
      - We join to NYSE schedule to set ts to that day’s 16:00 ET market close.
      - Use 'Adj Close' if present; else fall back to 'Close'.
    """

    latest_ts = get_latest_ts_for_symbol(symbol, "equities_us_daily", ts_col="ts")
    if start_date is None:
        if latest_ts is not None:
            last_ny_date = pd.to_datetime(latest_ts).tz_convert("America/New_York").date()
            # ensure we fetch at least one row even if already up to date
            start_date = pd.Timestamp(last_ny_date).strftime("%Y-%m-%d")
        else:
            start_date = "1980-01-01"

    nyse = mcal.get_calendar("NYSE")

    # ---------------------------
    # Path A: Tiingo first
    # ---------------------------
    try:
        tiingo_kwargs = {"frequency": "daily", "startDate": start_date}
        if end_date is not None:
            tiingo_kwargs["endDate"] = end_date
        ti = ti_client.get_dataframe(symbol, **tiingo_kwargs)
        if ti is None or len(ti) == 0:
            raise ValueError("Tiingo returned no rows")

        df = ti.reset_index().rename(columns={"index": "date"})
        # tiingo 'date' should be tz-aware (UTC). Normalize to UTC midnight (date key).
        dts = pd.to_datetime(df["date"], utc=True)
        df["trade_date_utc"] = dts.dt.tz_convert("UTC").dt.normalize()

        # Build calendar window
        min_utc = dts.min().tz_convert("UTC").date() - timedelta(days=5)
        max_utc = dts.max().tz_convert("UTC").date() + timedelta(days=5)
        sched = nyse.schedule(start_date=min_utc, end_date=max_utc)

        # Join on trading date in UTC; pick market close
        sched_df = pd.DataFrame({"mc_utc": sched["market_close"]})
        sched_df["trade_date_utc"] = sched_df["mc_utc"].dt.normalize()

        df = df.merge(sched_df[["trade_date_utc", "mc_utc"]], on="trade_date_utc", how="inner")
        df["ts"] = df["mc_utc"].dt.tz_convert("America/New_York")

        # Map columns
        # Tiingo typical columns: open, high, low, close, volume, adjClose (camelCase)
        out = pd.DataFrame({
            "ts": df["ts"],
            "open": pd.to_numeric(df.get("open"), errors="coerce"),
            "high": pd.to_numeric(df.get("high"), errors="coerce"),
            "low": pd.to_numeric(df.get("low"), errors="coerce"),
            "close": pd.to_numeric(df.get("close"), errors="coerce"),
            "volume": pd.to_numeric(df.get("volume"), errors="coerce"),
        })
        # adjClose may be present; if not, fall back to close
        if "adjClose" in df.columns:
            out["adj_close"] = pd.to_numeric(df["adjClose"], errors="coerce")
        elif "adj_close" in df.columns:
            out["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
        else:
            out["adj_close"] = out["close"]

        out["symbol"] = symbol
        out = out.dropna(subset=["ts", "close"])  # keep only valid rows
        return out

    except Exception as e:
        logging.warning("Tiingo failed for %s (%s); falling back to yfinance", symbol, e)

    # ---------------------------
    # Path B: yfinance fallback
    # ---------------------------
    import yfinance as yf

    history_kwargs = {
        "start": start_date,
        "auto_adjust": False,  # ensure 'Adj Close' stays separate from 'Close'
        "actions": True,
    }
    if end_date is not None:
        # yfinance treats `end` as exclusive; the importer treats it as inclusive.
        history_kwargs["end"] = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    px = yf.Ticker(symbol).history(**history_kwargs)

    if px.empty:
        # nothing we can do
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "adj_close", "symbol"])

    # yfinance index tz: usually America/New_York at 00:00 for the trading date
    if px.index.tz is None:
        px.index = px.index.tz_localize("America/New_York")

    # Build NYSE schedule for padded range (in NY)
    min_ny = px.index.min().tz_convert("America/New_York").date() - timedelta(days=5)
    max_ny = px.index.max().tz_convert("America/New_York").date() + timedelta(days=5)
    sched = nyse.schedule(start_date=min_ny, end_date=max_ny)

    sched_df = pd.DataFrame({"mc_utc": sched["market_close"]})
    # Normalize to NY midnight to create a join key in local wall time
    sched_df["trade_date_ny"] = sched_df["mc_utc"].dt.tz_convert("America/New_York").dt.normalize()

    df = px.copy()
    df["trade_date_ny"] = pd.to_datetime(df.index).tz_convert("America/New_York").normalize()
    df = df.merge(sched_df[["trade_date_ny", "mc_utc"]], on="trade_date_ny", how="inner")
    df["ts"] = df["mc_utc"].dt.tz_convert("America/New_York")

    # Map columns from yfinance:
    # Open, High, Low, Close, Volume, (optional) Adj Close, Dividends, Stock Splits, Capital Gains
    def _pick(name):
        # tolerate different cases/spaces
        for c in df.columns:
            if c.replace(" ", "").lower() == name.replace(" ", "").lower():
                return c
        return None

    col_open  = _pick("Open")
    col_high  = _pick("High")
    col_low   = _pick("Low")
    col_close = _pick("Close")
    col_vol   = _pick("Volume")
    col_adj   = _pick("Adj Close")  # may be None

    out = pd.DataFrame({
        "ts": df["ts"],
        "open": pd.to_numeric(df[col_open], errors="coerce") if col_open else pd.NA,
        "high": pd.to_numeric(df[col_high], errors="coerce") if col_high else pd.NA,
        "low": pd.to_numeric(df[col_low], errors="coerce") if col_low else pd.NA,
        "close": pd.to_numeric(df[col_close], errors="coerce") if col_close else pd.NA,
        "volume": pd.to_numeric(df[col_vol], errors="coerce") if col_vol else pd.NA,
        "adj_close": pd.to_numeric(df[col_adj], errors="coerce") if col_adj else pd.to_numeric(df[col_close], errors="coerce"),
    })

    out["symbol"] = symbol
    out = out.dropna(subset=["ts", "close"])
    return out


def fetch_crypto_prices(symbol):
    """
    Fetch crypto price data from Tiingo for a given symbol, starting from the latest date in the DB.
    Returns a DataFrame with columns: date, symbol, open, high, low, close, volume
    """
    latest_date = get_latest_ts_for_symbol(symbol, "crypto_daily", ts_col="date")
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

def save_equity_prices(df, conn_params=None):
    with get_connection(config()) as (conn,cur):
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO price_data.equities_us_daily
                    (symbol, ts, open, high, low, close, volume, adj_close)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, ts) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                    close = EXCLUDED.close, volume = EXCLUDED.volume, adj_close = EXCLUDED.adj_close;
            """, (
                row["symbol"],
                pd.to_datetime(row["ts"]).to_pydatetime(),  # tz-aware America/New_York
                row.get("open"), row.get("high"), row.get("low"),
                row.get("close"), row.get("volume"), row.get("adj_close"),
            ))
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



# ----------------------------
# Economic data from FRED/ALFRED
# ----------------------------

_FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

def _fetch_alfred_observations(
    fred_series_id: str,
    observation_start: str | None = None,
    observation_end: str | None = None,
) -> pd.DataFrame:
    """
    Pull *all vintages* for a FRED series via the ALFRED mechanism:
    GET /fred/series/observations with realtime_start=1776-07-04, realtime_end=9999-12-31.

    Returns a DataFrame with columns:
      realtime_start, realtime_end, date, value
    """
    api_key = os.environ["FRED_API_KEY"]

    params = {
        "series_id": fred_series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": "1776-07-04",
        "realtime_end": "9999-12-31",
        "limit": 100000,
        "offset": 0,
        "sort_order": "asc",
    }
    if observation_start:
        params["observation_start"] = observation_start
    if observation_end:
        params["observation_end"] = observation_end

    # PUBLICATION LAG assumed for series that carry NO vintage history.
    #
    # FRED rejects the all-vintages request (HTTP 400) for its own CALCULATED
    # daily series -- DFII5, DFII10, T5YIE, T10YIE all fail, while WALCL and
    # DTWEXBGS succeed. The distinction is revision: a series that is never
    # revised has no vintages to return, so ALFRED has nothing to serve.
    #
    # That is a statement about availability, and it has to be modelled rather
    # than ignored. These are Treasury/market rates: the curve is published
    # around 15:30 ET and FRED posts it the same evening, so the value for day
    # t is readable by an observer on day t+1 but NOT during day t's session.
    # Setting release_date = reference_date would claim same-day knowledge of a
    # number published after the close.
    #
    # One business day is therefore the conservative and correct convention,
    # and it is an ASSUMPTION -- flagged in the returned frame as
    # `vintage_source='observations'` so downstream code can tell a measured
    # release date from an imputed one.
    NO_VINTAGE_LAG_BDAYS = 1

    obs = []
    used_vintages = True
    while True:
        r = requests.get(_FRED_OBS_URL, params=params, timeout=60)
        if r.status_code == 400 and used_vintages:
            # No vintage history for this series -- fall back to plain
            # observations, which is the only thing FRED will serve for it.
            used_vintages = False
            params = {k: v for k, v in params.items()
                      if k not in ("realtime_start", "realtime_end")}
            params["offset"] = 0
            obs = []
            continue
        r.raise_for_status()
        payload = r.json()
        batch = payload.get("observations", []) or []
        if not batch:
            break
        obs.extend(batch)

        count = int(payload.get("count", len(obs)))
        params["offset"] = int(params["offset"]) + int(params["limit"])
        if len(obs) >= count:
            break

    if not obs:
        return pd.DataFrame(columns=["realtime_start", "realtime_end", "date",
                                      "value", "vintage_source"])

    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if used_vintages:
        df["realtime_start"] = pd.to_datetime(df["realtime_start"], errors="coerce").dt.date
        df["realtime_end"] = pd.to_datetime(df["realtime_end"], errors="coerce").dt.date
        df["vintage_source"] = "alfred"
    else:
        # No vintages: impute availability as one business day after the
        # observation, and say so in the frame. realtime_end is open-ended
        # because a never-revised value is never superseded.
        rs = df["date"] + pd.tseries.offsets.BDay(NO_VINTAGE_LAG_BDAYS)
        df["realtime_start"] = rs.dt.date
        df["realtime_end"] = pd.Timestamp("9999-12-31").date()
        df["vintage_source"] = "observations"

    df["date"] = df["date"].dt.date
    # Some values are "." (missing) in FRED JSON
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def fetch_eco_from_fred(
    fred_series_id: str,
    observation_start: str | None = None,
    observation_end: str | None = None,
    series_name: str | None = None,
    release_name: str | None = None,
) -> pd.DataFrame:
    """
    Fetch an economic series from FRED with *correct* first-release dates computed from ALFRED vintages.

    Output (one row per reference_date / observation date):
      - fred_series_id
      - series_name
      - release_name
      - release_id         (YYYYMMDD of release_date)
      - release_date       (first vintage realtime_start for that reference_date)
      - reference_date     (the observation date)
      - value              (latest available vintage value for that reference_date)

    Notes:
      - 'release_name' is your own labeling. FRED doesn't provide it.
      - We treat the *first appearance* of an observation (min realtime_start) as the release date.
      - We treat the *latest* value as the one with realtime_end == 9999-12-31 (when present), else max realtime_end.
    """
    series_name = series_name or fred_series_id
    release_name = release_name or series_name

    vint = _fetch_alfred_observations(
        fred_series_id=fred_series_id,
        observation_start=observation_start,
        observation_end=observation_end,
    )
    if vint.empty:
        return pd.DataFrame(
            columns=[
                "fred_series_id",
                "series_name",
                "release_name",
                "release_id",
                "release_date",
                "reference_date",
                "value",
            ]
        )

    first_release = (
        vint.groupby("date", as_index=False)["realtime_start"]
        .min()
        .rename(columns={"date": "reference_date", "realtime_start": "release_date"})
    )

    # Latest value per reference_date
    latest_end_sentinel = dtm.strptime("9999-12-31", "%Y-%m-%d").date()
    is_latest = (vint["realtime_end"] == latest_end_sentinel)
    if is_latest.any():
        latest = vint[is_latest].copy()
        latest = latest.sort_values(["date", "realtime_start"]).drop_duplicates("date", keep="last")
    else:
        latest = vint.sort_values(["date", "realtime_end", "realtime_start"]).drop_duplicates("date", keep="last")

    latest = latest[["date", "value"]].rename(columns={"date": "reference_date"})

    out = first_release.merge(latest, on="reference_date", how="left")
    out["fred_series_id"] = fred_series_id
    out["series_name"] = series_name
    out["release_name"] = release_name

    out["release_date"] = pd.to_datetime(out["release_date"])
    out["release_id"] = out["release_date"].dt.strftime("%Y%m%d").astype(int)

    out["reference_date"] = pd.to_datetime(out["reference_date"])
    out = out.dropna(subset=["release_date", "reference_date"])

    return out[
        [
            "fred_series_id",
            "series_name",
            "release_name",
            "release_id",
            "release_date",
            "reference_date",
            "value",
        ]
    ].sort_values(["reference_date"])


def save_eco_from_fred_to_db(
    df: pd.DataFrame,
    source: str = "FRED/ALFRED",
    country_code: str = "USA",
):
    """
    Save economic release schedule + release data into:
      - eco.release_schedule
      - eco.release_data

    Also ensures eco.release_content has a row for (release_name, series_name).
    """
    if df is None or df.empty:
        return

    required = {"release_id", "release_name", "series_name", "release_date", "reference_date", "value", "fred_series_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"save_eco_from_fred_to_db: missing required columns: {sorted(missing)}")

    # Ensure metadata exists (release_content)
    meta_rows = (
        df[["release_name", "series_name", "fred_series_id"]]
        .drop_duplicates()
        .to_dict("records")
    )
    for m in meta_rows:
        if not check_eco_metadata_exists(m["series_name"]):
            rid = int(df["release_id"].min())
            save_eco_metadata(
                release_id=rid,
                release_name=m["release_name"],
                series_name=m["series_name"],
                fred_series_id=m["fred_series_id"],
                source=source,
                country_code=country_code,
            )

    with get_connection(config()) as (conn, cur):
        for _, row in df.iterrows():
            cur.execute(
                """
                INSERT INTO eco.release_schedule (release_id, release_name, release_date, reference_date)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (release_name, reference_date) DO UPDATE
                SET release_date = EXCLUDED.release_date,
                    release_id   = EXCLUDED.release_id;
                """,
                (
                    int(row["release_id"]),
                    row["release_name"],
                    pd.to_datetime(row["release_date"]).to_pydatetime(),
                    pd.to_datetime(row["reference_date"]).to_pydatetime(),
                ),
            )

            cur.execute(
                """
                INSERT INTO eco.release_data (release_id, release_name, series_name, release_date, reference_date, value)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (release_name, series_name, reference_date) DO UPDATE
                SET value       = EXCLUDED.value,
                    release_date = EXCLUDED.release_date,
                    release_id   = EXCLUDED.release_id;
                """,
                (
                    int(row["release_id"]),
                    row["release_name"],
                    row["series_name"],
                    pd.to_datetime(row["release_date"]).to_pydatetime(),
                    pd.to_datetime(row["reference_date"]).to_pydatetime(),
                    None if pd.isna(row["value"]) else float(row["value"]),
                ),
            )

        conn.commit()


### ✅ Main ###
@click.command()
@click.option('--assets_file', default="macro_assets.yaml", help="Path to assets YAML file.")
@click.option('--eco-only', is_flag=True, default=False,
              help="Skip equities/crypto/VIX/CMT/GSCPI; refresh only the FRED "
                   "economic series. Use with --eco-series to target a few.")
@click.option('--assets-only', is_flag=True, default=False,
              help="Refresh only the assets in --assets_file; skip the FRED "
                   "economic series entirely. The eco block issues an ALFRED "
                   "vintage request per series and dominates the runtime, so "
                   "pulling a handful of new tickers does not need it.")
# Explicit parameter name: `eco_series` is ALREADY a local inside main() -- the
# list of (id, name, release) tuples built from the YAML -- so letting click
# derive that name from --eco-series shadowed the CLI value with a list, and the
# filter below died on list.split(). Named apart so the two cannot collide.
@click.option('--eco-series', 'eco_series_filter', default=None,
              help="Comma-separated fred_series_id list to refresh, filtered "
                   "from macro_series.yaml. Default: every series in the file.")
def main(assets_file, eco_only, assets_only, eco_series_filter):
    """Refresh market and macro data into coredata.

    The full run touches every asset in macro_assets.yaml plus all ~84 FRED
    series, which is many minutes of ALFRED vintage requests. When only a
    handful of series need refreshing -- a newly added one, or a series whose
    vintage coverage turned out to be short -- --eco-only --eco-series avoids
    re-pulling everything else to get at them. The YAML stays the single
    definition of what exists; these flags only narrow what runs.
    """
    if eco_only and assets_only:
        raise click.UsageError("--eco-only and --assets-only are mutually "
                               "exclusive; passing both would skip everything.")

    logging.info(f"🔹 Loading assets from {assets_file}...")
    with open(assets_file, "r") as f:
        assets = yaml.load(f, Loader=yaml.BaseLoader)  # all scalars as strings
    equity_symbols = list(map(str, assets.get("equities", [])))
    
    crypto_symbols = assets.get("crypto", [])
    
    if eco_only:
        logging.info("🔹 --eco-only: skipping equities, crypto, VIX, CMT, GSCPI.")
    for symbol in ([] if eco_only else equity_symbols):
        logging.info(f'Processing {symbol}')
        logging.info(f'Do we have {symbol} metadata?')
        if not check_metadata_exists(symbol, "equities_us"):
            logging.info(f"Fetching metadata for {symbol}...")
            save_equity_metadata(symbol)
        logging.info(f"Fetching prices for {symbol}...")
        df = fetch_equity_prices(symbol)
        if not df.empty:
            save_equity_prices(df)
            logging.info(f"Stored {symbol} equity data.")

    for symbol in ([] if eco_only else crypto_symbols):
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

    # VIX / Treasury CMT / GSCPI are outside an asset-only refresh. Both
    # flags must skip the work, not just its log lines, so a targeted price
    # update cannot fail on an unrelated data source.
    if not eco_only and not assets_only:
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
            if not check_eco_metadata_exists("GSCPI"):
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
        
    
    # As with --eco-only above: the flag must skip the WORK, not just the log
    # line. Returning here is the whole point of --assets-only -- the loop
    # below issues one ALFRED vintage request per series across ~84 series and
    # is what makes a full run take many minutes.
    if assets_only:
        logging.info("🔹 --assets-only: skipping FRED economic series.")
        logging.info("✅ Data retrieval and storage complete.")
        return

    logging.info("🔹 Updating Economic Releases from FRED/ALFRED (with true release dates)...")
    macro_series_file = os.path.join(os.path.dirname(__file__), "macro_series.yaml")
    with open(macro_series_file) as _f:
        _macro_cfg = yaml.safe_load(_f)
    eco_series = [
        (s["fred_series_id"], s["series_name"], s["release_name"])
        for group in _macro_cfg.values()
        for s in group
    ]
    # Deduplicate: same (fred_series_id, series_name) may appear in multiple groups
    seen = set()
    eco_series_deduped = []
    for item in eco_series:
        key = (item[0], item[1])
        if key not in seen:
            seen.add(key)
            eco_series_deduped.append(item)

    if eco_series_filter:
        wanted = {x.strip() for x in eco_series_filter.split(",") if x.strip()}
        before = len(eco_series_deduped)
        eco_series_deduped = [t for t in eco_series_deduped if t[0] in wanted]
        missing = wanted - {t[0] for t in eco_series_deduped}
        logging.info(f"--eco-series: {len(eco_series_deduped)} of {before} selected")
        if missing:
            logging.warning(f"not found in macro_series.yaml: {sorted(missing)}")

    for fred_series_id, series_name, release_name in eco_series_deduped:
        try:
            logging.info(f"Fetching {series_name} ({fred_series_id}) vintages via ALFRED...")
            df_eco = fetch_eco_from_fred(
                fred_series_id,
                observation_start="1990-01-01",
                series_name=series_name,
                release_name=release_name,
            )
            if df_eco.empty:
                logging.info(f"No data returned for {fred_series_id}.")
                continue
            save_eco_from_fred_to_db(df_eco, source="FRED/ALFRED", country_code="USA")
            logging.info(f"Upserted {len(df_eco)} rows for {series_name}.")
        except Exception as e:
            logging.error(f"Failed updating {fred_series_id}: {e}")

logging.info("✅ Data retrieval and storage complete.")

if __name__ == "__main__":
    main()
