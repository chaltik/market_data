CREATE SCHEMA IF NOT EXISTS eco;

CREATE TABLE IF NOT EXISTS eco.release_content (
    release_id BIGINT NOT NULL,
    release_name TEXT NOT NULL,
    series_name TEXT NOT NULL,
    bls_series_id VARCHAR NULL,
    fred_series_id VARCHAR NULL,
    source TEXT,
    country_code VARCHAR(3),
    PRIMARY KEY (release_name, series_name)
);

CREATE TABLE IF NOT EXISTS eco.release_schedule (
    release_id BIGINT NOT NULL,
    release_name TEXT NOT NULL,
    release_date DATE NOT NULL,
    reference_date DATE NOT NULL,
    release_time TIME,
    release_timezone TEXT,
    release_date_source TEXT,
    release_date_verified BOOLEAN NOT NULL DEFAULT FALSE,
    first_fetched_at TIMESTAMPTZ,
    PRIMARY KEY (release_name, reference_date)
);

-- Existing databases may predate these columns. DDL belongs here, not in
-- individual data loaders. These statements are safe to repeat.
ALTER TABLE eco.release_schedule ADD COLUMN IF NOT EXISTS release_time TIME;
ALTER TABLE eco.release_schedule ADD COLUMN IF NOT EXISTS release_timezone TEXT;
ALTER TABLE eco.release_schedule ADD COLUMN IF NOT EXISTS release_date_source TEXT;
ALTER TABLE eco.release_schedule ADD COLUMN IF NOT EXISTS release_date_verified BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE eco.release_schedule ADD COLUMN IF NOT EXISTS first_fetched_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS eco.release_data (
    release_id BIGINT NOT NULL,
    release_name TEXT NOT NULL,
    series_name TEXT NOT NULL,
    release_date DATE NOT NULL,
    reference_date DATE NOT NULL,
    value NUMERIC,
    PRIMARY KEY (release_name, series_name, reference_date),
    FOREIGN KEY (release_name, series_name) REFERENCES eco.release_content(release_name, series_name),
    FOREIGN KEY (release_name, reference_date) REFERENCES eco.release_schedule(release_name, reference_date)
);

-- Point-in-time macro releases and immutable retrieval snapshots.
CREATE TABLE IF NOT EXISTS eco.release_vintages (
    source TEXT NOT NULL,
    source_series_id TEXT NOT NULL,
    series_name TEXT NOT NULL,
    release_name TEXT NOT NULL,
    reference_date DATE NOT NULL,
    vintage_start DATE NOT NULL,
    vintage_end DATE,
    value NUMERIC,
    observed_at TIMESTAMPTZ,
    observed_at_confidence TEXT NOT NULL DEFAULT 'date_only',
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (source, source_series_id, reference_date, vintage_start)
);

CREATE INDEX IF NOT EXISTS release_vintages_asof_idx
    ON eco.release_vintages (source_series_id, reference_date, vintage_start, vintage_end);

CREATE TABLE IF NOT EXISTS eco.release_observation_snapshots (
    source TEXT NOT NULL,
    source_series_id TEXT NOT NULL,
    series_name TEXT NOT NULL,
    release_name TEXT NOT NULL,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    value NUMERIC,
    release_date_confidence TEXT NOT NULL,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (source, source_series_id, reference_date, release_date, retrieved_at)
);

CREATE TABLE IF NOT EXISTS eco.spf_raw_snapshots (
    forecast_variable TEXT NOT NULL,
    response_statistic TEXT NOT NULL,
    source_url TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload_gzip BYTEA NOT NULL,
    CONSTRAINT spf_raw_snapshots_response_statistic_check
        CHECK (response_statistic IN ('mean', 'median')),
    PRIMARY KEY (forecast_variable, response_statistic, payload_sha256)
);

CREATE TABLE IF NOT EXISTS eco.spf_observations (
    forecast_variable TEXT NOT NULL,
    response_statistic TEXT NOT NULL,
    survey_year SMALLINT NOT NULL,
    survey_quarter SMALLINT NOT NULL,
    reference_date DATE NOT NULL,
    release_date DATE,
    value NUMERIC,
    unit TEXT NOT NULL DEFAULT 'annualized percentage points',
    release_date_confidence TEXT NOT NULL,
    source_url TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT spf_observations_response_statistic_check
        CHECK (response_statistic IN ('mean', 'median')),
    CONSTRAINT spf_observations_survey_quarter_check
        CHECK (survey_quarter >= 1 AND survey_quarter <= 4),
    PRIMARY KEY (forecast_variable, response_statistic, survey_year,
                 survey_quarter, payload_sha256)
);

CREATE INDEX IF NOT EXISTS spf_observations_asof_idx
    ON eco.spf_observations (forecast_variable, response_statistic,
                             reference_date, release_date);

CREATE TABLE IF NOT EXISTS eco.eia_wpsr_raw_snapshots (
    release_date DATE NOT NULL,
    reference_date DATE NOT NULL,
    table_number SMALLINT NOT NULL,
    source_url TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload_gzip BYTEA NOT NULL,
    PRIMARY KEY (release_date, table_number, payload_sha256)
);

CREATE TABLE IF NOT EXISTS eco.eia_wpsr_observations (
    release_date DATE NOT NULL,
    reference_date DATE NOT NULL,
    table_number SMALLINT NOT NULL,
    table_title TEXT NOT NULL,
    series_key TEXT NOT NULL,
    dimension_1 TEXT NOT NULL,
    dimension_2 TEXT NOT NULL DEFAULT '',
    measure_variant TEXT NOT NULL DEFAULT 'weekly',
    value NUMERIC,
    raw_value TEXT,
    source_url TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    release_timestamp TIMESTAMPTZ,
    release_timestamp_confidence TEXT NOT NULL DEFAULT 'date_only_unverified_time',
    PRIMARY KEY (release_date, table_number, reference_date,
                 series_key, measure_variant, payload_sha256)
);

CREATE INDEX IF NOT EXISTS eia_wpsr_asof_idx
    ON eco.eia_wpsr_observations (series_key, reference_date, release_date);

