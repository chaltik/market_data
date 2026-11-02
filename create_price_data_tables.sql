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

-- Provenance and country/region mapping for the international ETF research panel.
-- Prices remain in the established equities_us_daily table.
CREATE TABLE IF NOT EXISTS price_data.international_etf_metadata (
    symbol                  TEXT PRIMARY KEY REFERENCES price_data.equities_us(symbol),
    country_code            CHAR(2),
    country_name            TEXT,
    region                  TEXT NOT NULL,
    universe_role           TEXT NOT NULL,
    trading_currency        CHAR(3) NOT NULL DEFAULT 'USD',
    provider                TEXT NOT NULL,
    source                  TEXT NOT NULL,
    inception_date          DATE,
    last_observation_date   DATE,
    delist_date             DATE,
    price_adjustment_field  TEXT NOT NULL DEFAULT 'adj_close',
    provenance              TEXT NOT NULL,
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS price_data.international_etf_cohort_membership (
    cohort_id       TEXT NOT NULL,
    symbol          TEXT NOT NULL REFERENCES price_data.international_etf_metadata(symbol),
    cohort_role     TEXT NOT NULL,
    selected_at     DATE NOT NULL,
    frozen          BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (cohort_id, symbol)
);

CREATE TABLE IF NOT EXISTS price_data.fx_currency_metadata (
    currency_code       CHAR(3) PRIMARY KEY,
    source_series_id    TEXT NOT NULL,
    source_quote        TEXT NOT NULL,
    normalized_quote    TEXT NOT NULL DEFAULT 'local_currency_per_usd',
    provider            TEXT NOT NULL,
    source              TEXT NOT NULL,
    earliest_date       DATE,
    latest_date         DATE,
    provenance          TEXT NOT NULL,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS price_data.fx_daily (
    currency_code       CHAR(3) NOT NULL REFERENCES price_data.fx_currency_metadata(currency_code),
    date                DATE NOT NULL,
    local_per_usd       NUMERIC NOT NULL,
    source_value        NUMERIC NOT NULL,
    source_series_id    TEXT NOT NULL,
    source_quote        TEXT NOT NULL,
    provider            TEXT NOT NULL,
    PRIMARY KEY (currency_code, date),
    CONSTRAINT fx_daily_positive_local_per_usd CHECK (local_per_usd > 0),
    CONSTRAINT fx_daily_positive_source_value CHECK (source_value > 0)
);

-- Keep existing installations safe as well as fresh ones.  A source missing
-- sentinel such as H.10's -9999 must never enter the normalized FX table.
DO $$
BEGIN
    BEGIN
        ALTER TABLE price_data.fx_daily
            ADD CONSTRAINT fx_daily_positive_local_per_usd CHECK (local_per_usd > 0);
    EXCEPTION WHEN duplicate_object THEN
        NULL;
    END;
    BEGIN
        ALTER TABLE price_data.fx_daily
            ADD CONSTRAINT fx_daily_positive_source_value CHECK (source_value > 0);
    EXCEPTION WHEN duplicate_object THEN
        NULL;
    END;
END $$;

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
