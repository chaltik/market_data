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

-- Table for storing daily equity & ETF prices
CREATE TABLE IF NOT EXISTS price_data.equities_us_daily (
    symbol TEXT REFERENCES price_data.equities_us(symbol),
    date DATE NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT NULL,
    adj_close NUMERIC NULL,
    PRIMARY KEY (symbol, date)
);

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
