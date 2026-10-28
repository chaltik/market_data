-- Daily volatility-index observations. Run once before loading.
-- Consolidates the legacy per-index tables into one table. This migration is
-- separate from create_rates_tables.sql, which contains legacy DROP statements.
BEGIN;

CREATE SCHEMA IF NOT EXISTS market_indices;

CREATE TABLE IF NOT EXISTS market_indices.volatility_indices (
    date DATE NOT NULL,
    index_symbol TEXT NOT NULL,
    value NUMERIC NOT NULL,
    source TEXT NOT NULL,
    source_series_id TEXT NOT NULL,
    PRIMARY KEY (date, index_symbol)
);

-- Preserve previously loaded observations when upgrading from per-index tables.
DO $$
BEGIN
    IF to_regclass('market_indices.vix') IS NOT NULL THEN
        EXECUTE $copy$
            INSERT INTO market_indices.volatility_indices
                (date, index_symbol, value, source, source_series_id)
            SELECT date, 'VIX', value, 'FRED', 'VIXCLS'
            FROM market_indices.vix
            ON CONFLICT (date, index_symbol) DO NOTHING
        $copy$;
    END IF;
    IF to_regclass('market_indices.move') IS NOT NULL THEN
        EXECUTE $copy$
            INSERT INTO market_indices.volatility_indices
                (date, index_symbol, value, source, source_series_id)
            SELECT date, 'MOVE', value, 'Yahoo Finance', '^MOVE'
            FROM market_indices.move
            ON CONFLICT (date, index_symbol) DO NOTHING
        $copy$;
    END IF;
    IF to_regclass('market_indices.gvz') IS NOT NULL THEN
        EXECUTE $copy$
            INSERT INTO market_indices.volatility_indices
                (date, index_symbol, value, source, source_series_id)
            SELECT date, 'GVZ', value, 'FRED', 'GVZCLS'
            FROM market_indices.gvz
            ON CONFLICT (date, index_symbol) DO NOTHING
        $copy$;
    END IF;
    IF to_regclass('market_indices.ovx') IS NOT NULL THEN
        EXECUTE $copy$
            INSERT INTO market_indices.volatility_indices
                (date, index_symbol, value, source, source_series_id)
            SELECT date, 'OVX', value, 'FRED', 'OVXCLS'
            FROM market_indices.ovx
            ON CONFLICT (date, index_symbol) DO NOTHING
        $copy$;
    END IF;
END $$;

-- Keep the former VIX table name available to existing read queries.
-- On the first migration it is a table; on later runs it is this view.
DROP VIEW IF EXISTS market_indices.vix;
DROP TABLE IF EXISTS market_indices.vix;
DROP TABLE IF EXISTS market_indices.move;
DROP TABLE IF EXISTS market_indices.gvz;
DROP TABLE IF EXISTS market_indices.ovx;

CREATE VIEW market_indices.vix AS
    SELECT date, value
    FROM market_indices.volatility_indices
    WHERE index_symbol = 'VIX';

CREATE INDEX IF NOT EXISTS volatility_indices_symbol_date_idx
    ON market_indices.volatility_indices (index_symbol, date);

COMMENT ON TABLE market_indices.volatility_indices IS
    'Daily closes for volatility indices; one row per date and index_symbol, with source identifiers.';
COMMENT ON VIEW market_indices.vix IS
    'Read-compatible VIX-only view over market_indices.volatility_indices.';

COMMIT;
