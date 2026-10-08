-- Create schema for price data
CREATE SCHEMA IF NOT EXISTS price_data;

-- Table for equities & ETFs (renamed from "equities" to "equities_daily")
CREATE TABLE IF NOT EXISTS price_data.equities_us (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    asset_class TEXT CHECK (asset_class IN ('stock', 'etf')),
    exchange TEXT,
    currency TEXT,
    sector TEXT NULL,
    industry TEXT NULL
);

-- Daily equity/ETF prices at NYSE close (stored in UTC)
CREATE TABLE IF NOT EXISTS price_data.equities_us_daily (
    symbol       TEXT NOT NULL REFERENCES price_data.equities_us(symbol),
    ts           TIMESTAMPTZ NOT NULL,   -- NYSE market close, converted to UTC
    open         NUMERIC,
    high         NUMERIC,
    low          NUMERIC,
    close        NUMERIC,
    volume       BIGINT,
    adj_close    NUMERIC,
    PRIMARY KEY (symbol, ts)
);


-- Helpful for range scans
CREATE INDEX IF NOT EXISTS equities_us_daily_ts_idx ON price_data.equities_us_daily (ts);

-- One-time study universe and prices. These tables are intentionally separate
-- from the regularly maintained equities tables so study selections remain
-- fixed and are never picked up by the daily updater.
CREATE TABLE IF NOT EXISTS price_data.study_equities (
    study_id                  TEXT NOT NULL,
    symbol                    TEXT NOT NULL,
    company_name              TEXT NOT NULL,
    security_type             TEXT,
    security_price            NUMERIC,
    beta_3yr_annualized       NUMERIC,
    market_capitalization     TEXT,
    source_file               TEXT,
    imported_at               TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (study_id, symbol)
);

CREATE TABLE IF NOT EXISTS price_data.study_equities_daily (
    study_id    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    open        NUMERIC,
    high        NUMERIC,
    low         NUMERIC,
    close       NUMERIC,
    volume      BIGINT,
    adj_close   NUMERIC,
    PRIMARY KEY (study_id, symbol, ts),
    FOREIGN KEY (study_id, symbol)
        REFERENCES price_data.study_equities (study_id, symbol)
);

CREATE INDEX IF NOT EXISTS study_equities_daily_ts_idx
    ON price_data.study_equities_daily (study_id, ts);


-- Table for crypto prices (BTCUSD, ETHUSD, etc.)
CREATE TABLE IF NOT EXISTS price_data.crypto (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    exchange TEXT
);

-- Table for storing daily crypto prices
CREATE TABLE IF NOT EXISTS price_data.crypto_daily (
    symbol TEXT REFERENCES price_data.crypto(symbol),
    date DATE NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC NULL,
    volume_notional NUMERIC NULL,
    trades_done INTEGER NULL,
    PRIMARY KEY (symbol, date)
);
