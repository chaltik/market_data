-- Create schema
CREATE SCHEMA IF NOT EXISTS interest_rates;

DROP TABLE IF EXISTS interest_rates.treasury_cmt;

CREATE TABLE interest_rates.treasury_cmt (
    date DATE NOT NULL,
    tenor TEXT NOT NULL,   -- e.g. '1 Mo', '2 Yr', etc.
    rate NUMERIC(5,2),     -- can be NULL for missing data
    PRIMARY KEY (date, tenor)
);

COMMENT ON TABLE interest_rates.treasury_cmt IS 'Normalized format of daily US Treasury CMT rates.';

CREATE SCHEMA IF NOT EXISTS market_indices;

DROP TABLE IF EXISTS market_indices.vix;

CREATE TABLE market_indices.vix (
    date DATE PRIMARY KEY,
    value NUMERIC NOT NULL
);
CREATE INDEX vix_date_idx ON market_indices.vix ("date");
