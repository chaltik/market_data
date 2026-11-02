-- Run after create_eco_release_data_pit_cache.sql. Keep EIA gas storage
-- selected directly so it is visible before the next PIT cache refresh.
CREATE OR REPLACE VIEW eco.v_release_data AS
SELECT
    cache.release_id,
    cache.release_name,
    cache.series_name,
    cache.source,
    cache.source_series_id,
    cache.release_date,
    cache.reference_date,
    cache.vintage_end,
    cache.value,
    cache.release_date_confidence,
    cache.source_metadata
FROM eco._release_data_pit_cache AS cache
WHERE NOT (
    cache.release_name = 'EIA Weekly Natural Gas Storage Report'
    AND cache.series_name = 'Working gas in underground storage, Lower 48'
)

UNION ALL

SELECT
    to_char(COALESCE(spf.release_date, spf.reference_date)::timestamptz,
            'YYYYMMDD')::bigint AS release_id,
    'Philadelphia Fed Survey of Professional Forecasters'::text AS release_name,
    (spf.forecast_variable || '_' || spf.response_statistic)::text AS series_name,
    'Philadelphia Fed SPF'::text AS source,
    (spf.forecast_variable || '_' || spf.response_statistic)::text AS source_series_id,
    spf.release_date,
    spf.reference_date,
    NULL::date AS vintage_end,
    spf.value,
    spf.release_date_confidence,
    jsonb_build_object(
        'forecast_variable', spf.forecast_variable,
        'response_statistic', spf.response_statistic,
        'survey_year', spf.survey_year,
        'survey_quarter', spf.survey_quarter,
        'unit', spf.unit,
        'source_url', spf.source_url,
        'payload_sha256', spf.payload_sha256,
        'retrieved_at', spf.retrieved_at
    ) AS source_metadata
FROM eco.spf_observations AS spf

UNION ALL

SELECT
    data.release_id,
    data.release_name,
    data.series_name,
    COALESCE(content.source, 'EIA') AS source,
    'NW2_EPG0_SWO_R48_BCF'::text AS source_series_id,
    data.release_date,
    data.reference_date,
    NULL::date AS vintage_end,
    data.value,
    CASE
        WHEN schedule.release_date_verified THEN 'verified_date_only'
        WHEN schedule.release_date_source IS NOT NULL THEN 'inferred_conservative_lag'
        ELSE 'unknown'
    END AS release_date_confidence,
    jsonb_strip_nulls(jsonb_build_object(
        'source_series_id', 'NW2_EPG0_SWO_R48_BCF',
        'frequency', 'weekly',
        'unit', 'Bcf',
        'reference_date_semantics', 'storage week ending Friday',
        'release_date_source', schedule.release_date_source,
        'release_date_verified', schedule.release_date_verified,
        'availability_lag', '14 calendar days + 1 US federal business day',
        'availability_date_semantics',
            'Conservative research availability assumption; not the actual EIA publication date',
        'standard_eia_release', 'Usually Thursday at 10:30 a.m. America/New_York',
        'source_url', 'https://www.eia.gov/dnav/ng/hist_xls/NW2_EPG0_SWO_R48_BCFw.xls',
        'retrieved_at', schedule.first_fetched_at
    )) AS source_metadata
FROM eco.release_data AS data
JOIN eco.release_content AS content
  ON content.release_name = data.release_name
 AND content.series_name = data.series_name
JOIN eco.release_schedule AS schedule
  ON schedule.release_name = data.release_name
 AND schedule.reference_date = data.reference_date
WHERE data.release_name = 'EIA Weekly Natural Gas Storage Report'
  AND data.series_name = 'Working gas in underground storage, Lower 48';

COMMENT ON VIEW eco.v_release_data IS
    'Unified release observations with source identifiers, vintages where available, and release-date confidence/provenance. EIA gas storage uses a conservative inferred research availability date, not the actual historical publication date.';
