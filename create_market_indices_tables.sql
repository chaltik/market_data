-- =========================================================
-- market_indices schema (NDX-ready) — with sector columns
-- and UNIQUE (index_id, symbol, start_date, end_date)
-- =========================================================
BEGIN;

-- Drop in dependency order (safe if tables already exist)
DROP VIEW IF EXISTS market_indices.v_index_constituents_pti CASCADE;
DROP TABLE IF EXISTS market_indices.index_constituents CASCADE;
DROP TABLE IF EXISTS market_indices.index_changes_raw CASCADE;
DROP TABLE IF EXISTS market_indices.seed_snapshots CASCADE;
DROP TABLE IF EXISTS market_indices.sources CASCADE;
DROP TABLE IF EXISTS market_indices.indices CASCADE;

-- Schema
CREATE SCHEMA IF NOT EXISTS market_indices;

-- Enums
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname='change_action') THEN
    CREATE TYPE market_indices.change_action AS ENUM ('ADD','REMOVE','RENAME','TRANSFER','OTHER');
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname='effective_session') THEN
    CREATE TYPE market_indices.effective_session AS ENUM ('PRIOR_TO_OPEN','AFTER_CLOSE','INTRADAY','UNKNOWN');
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname='source_type') THEN
    CREATE TYPE market_indices.source_type AS ENUM ('WIKIPEDIA','PRESS_RELEASE','ETF','MANUAL');
  END IF;
END $$;

-- Core tables
CREATE TABLE market_indices.indices (
  index_id        BIGSERIAL PRIMARY KEY,
  code            TEXT UNIQUE NOT NULL,      -- e.g., 'NDX'
  name            TEXT NOT NULL,
  provider        TEXT,
  methodology_url TEXT,
  notes           TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE market_indices.sources (
  source_id      BIGSERIAL PRIMARY KEY,
  source_type    market_indices.source_type NOT NULL,
  title          TEXT,
  url            TEXT NOT NULL,
  published_date DATE,
  retrieved_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  raw_text       TEXT,
  checksum       TEXT,
  CONSTRAINT uq_sources_url UNIQUE (url)
);

-- Atomic events (adds/removes etc.)
CREATE TABLE market_indices.index_changes_raw (
  change_id          BIGSERIAL PRIMARY KEY,
  index_id           BIGINT NOT NULL REFERENCES market_indices.indices(index_id) ON DELETE CASCADE,
  action             market_indices.change_action NOT NULL,
  company_name       TEXT,
  symbol             TEXT,
  sector             TEXT, -- NEW
  effective_date     DATE NOT NULL,
  effective_session  market_indices.effective_session NOT NULL DEFAULT 'UNKNOWN',
  announcement_date  DATE,
  source_id          BIGINT REFERENCES market_indices.sources(source_id) ON DELETE SET NULL,
  notes              TEXT,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  event_hash         TEXT NOT NULL,
  CONSTRAINT uq_event UNIQUE (event_hash)
);
CREATE INDEX idx_changes_idx_date       ON market_indices.index_changes_raw(index_id, effective_date);
CREATE INDEX idx_changes_symbol         ON market_indices.index_changes_raw(symbol);

-- Point-in-time membership
CREATE TABLE market_indices.index_constituents (
  constituent_id    BIGSERIAL PRIMARY KEY,
  index_id          BIGINT NOT NULL REFERENCES market_indices.indices(index_id) ON DELETE CASCADE,
  symbol            TEXT NOT NULL,
  company_name      TEXT,
  sector            TEXT, -- NEW
  start_date        DATE NOT NULL,
  end_date          DATE,                -- NULL => currently active
  added_change_id   BIGINT REFERENCES market_indices.index_changes_raw(change_id) ON DELETE SET NULL,
  removed_change_id BIGINT REFERENCES market_indices.index_changes_raw(change_id) ON DELETE SET NULL,
  last_verified_asof DATE,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  -- Updated uniqueness: allow multiple unknown-start intervals with different end_dates
  CONSTRAINT uq_constituent_period UNIQUE (index_id, symbol, start_date, end_date)
);
CREATE INDEX idx_constituents_active    ON market_indices.index_constituents(index_id) WHERE end_date IS NULL;
CREATE INDEX idx_constituents_period    ON market_indices.index_constituents(index_id, start_date, end_date);
CREATE INDEX idx_constituents_symbol    ON market_indices.index_constituents(index_id, symbol);

-- Seed snapshots (current list from ETF/website)
CREATE TABLE market_indices.seed_snapshots (
  seed_id      BIGSERIAL PRIMARY KEY,
  index_id     BIGINT NOT NULL REFERENCES market_indices.indices(index_id) ON DELETE CASCADE,
  as_of_date   DATE NOT NULL,
  symbol       TEXT NOT NULL,
  company_name TEXT,
  sector       TEXT, -- NEW
  weight       NUMERIC,
  source_id    BIGINT REFERENCES market_indices.sources(source_id) ON DELETE SET NULL,
  notes        TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_seed UNIQUE (index_id, as_of_date, symbol)
);
CREATE INDEX idx_seed_idx_date          ON market_indices.seed_snapshots(index_id, as_of_date);

-- Helper view (simple pass-through; filter by date in queries)
CREATE OR REPLACE VIEW market_indices.v_index_constituents_pti AS
SELECT index_id, symbol, company_name, sector, start_date, end_date
FROM market_indices.index_constituents;

-- Bootstrap NDX row
INSERT INTO market_indices.indices(code, name, provider, methodology_url)
VALUES ('NDX','Nasdaq-100','Nasdaq','https://indexes.nasdaq.com/docs/Methodology_NDX.pdf')
ON CONFLICT (code) DO NOTHING;

COMMIT;

-- Constituents for an index code as-of a date (defaults to today)
CREATE OR REPLACE FUNCTION market_indices.constituents_asof(
  p_code  text,
  p_asof  date DEFAULT CURRENT_DATE
)
RETURNS TABLE (
  symbol        text,
  sector        text,
  company_name  text
)
LANGUAGE sql
STABLE
AS $$
  SELECT v.symbol, v.sector, v.company_name
  FROM market_indices.v_index_constituents_pti v
  JOIN market_indices.indices i ON i.index_id = v.index_id
  WHERE i.code = p_code
    AND p_asof >= v.start_date
    AND (v.end_date IS NULL OR p_asof < v.end_date)
  ORDER BY v.symbol;
$$;
