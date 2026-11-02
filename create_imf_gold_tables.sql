-- IMF monthly official-reserve gold-volume observations. This is not a
-- release-vintage table: IMF SDMX does not expose actual publication dates.
CREATE SCHEMA IF NOT EXISTS eco;

CREATE TABLE IF NOT EXISTS eco.imf_gold_observations (
    dataset_code TEXT NOT NULL,
    country_code TEXT NOT NULL,
    data_kind TEXT NOT NULL,
    indicator_code TEXT NOT NULL,
    indicator_name TEXT NOT NULL,
    sector_code TEXT NOT NULL DEFAULT '',
    accounting_entry_code TEXT NOT NULL DEFAULT '',
    frequency TEXT NOT NULL,
    reference_period TEXT NOT NULL,
    reference_date DATE NOT NULL,
    source_unit_code TEXT NOT NULL,
    normalized_unit TEXT NOT NULL,
    source_scale SMALLINT NOT NULL,
    raw_value NUMERIC NOT NULL,
    value NUMERIC NOT NULL,
    source_url TEXT NOT NULL,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (dataset_code, country_code, indicator_code, sector_code,
                 accounting_entry_code, frequency, reference_period,
                 source_unit_code),
    CHECK (dataset_code IN ('IRFCL', 'BOP')),
    CHECK (data_kind IN ('holding', 'net_transaction')),
    CHECK (frequency IN ('M', 'Q', 'A'))
);

CREATE INDEX IF NOT EXISTS imf_gold_observations_country_date_idx
    ON eco.imf_gold_observations (country_code, reference_date);

CREATE INDEX IF NOT EXISTS imf_gold_observations_indicator_date_idx
    ON eco.imf_gold_observations (indicator_code, reference_date);

COMMENT ON TABLE eco.imf_gold_observations IS
    'Monthly IMF IRFCL reserve-assets gold volume for sector S1XS1311 (Monetary Authorities and Central Government excluding Social Security). This is not a vintage table; IMF SDMX does not provide actual observation release dates.';

COMMENT ON COLUMN eco.imf_gold_observations.value IS
    'Value normalized to normalized_unit by dividing the raw SDMX value by 10^source_scale; raw_value is retained for auditability.';

COMMENT ON COLUMN eco.imf_gold_observations.sector_code IS
    'IMF IRFCL reporting sector; this backfill contains S1XS1311 (Monetary Authorities and Central Government excluding Social Security).';

COMMENT ON COLUMN eco.imf_gold_observations.data_kind IS
    'holding identifies an official-reserve stock observation.';

COMMENT ON COLUMN eco.imf_gold_observations.reference_date IS
    'Period end: month end for M, quarter end for Q, and December 31 for A.';
