-- Run after create_eco_data_tables.sql. This is the indexed PIT source for
-- eco.v_release_data. The definition mirrors the materialized view in coredata.
CREATE MATERIALIZED VIEW IF NOT EXISTS eco._release_data_pit_cache AS
SELECT md5(jsonb_build_array(
           release_id, release_name, series_name, source, source_series_id,
           release_date, reference_date, vintage_end, value,
           release_date_confidence, source_metadata)::text) AS row_key,
       release_id, release_name, series_name, source, source_series_id,
       release_date, reference_date, vintage_end, value,
       release_date_confidence, source_metadata
FROM (
    SELECT to_char(v.vintage_start::timestamptz, 'YYYYMMDD')::bigint AS release_id,
           v.release_name, v.series_name, v.source, v.source_series_id,
           v.vintage_start AS release_date, v.reference_date, v.vintage_end,
           v.value, v.observed_at_confidence AS release_date_confidence,
           v.source_metadata
    FROM eco.release_vintages v

    UNION ALL

    SELECT d.release_id, d.release_name, d.series_name,
           COALESCE(c.source, 'unknown') AS source,
           COALESCE(c.fred_series_id, c.bls_series_id, d.series_name)
               AS source_series_id,
           d.release_date, d.reference_date, NULL::date AS vintage_end,
           d.value,
           CASE WHEN s.release_date_verified THEN 'verified_date_only'
                WHEN s.release_date_source IS NOT NULL THEN 'inferred_date'
                ELSE 'unknown' END AS release_date_confidence,
           jsonb_strip_nulls(jsonb_build_object(
               'release_date_source', s.release_date_source,
               'release_date_verified', s.release_date_verified,
               'first_fetched_at', s.first_fetched_at)) AS source_metadata
    FROM eco.release_data d
    LEFT JOIN eco.release_content c
      ON c.release_name = d.release_name AND c.series_name = d.series_name
    LEFT JOIN eco.release_schedule s
      ON s.release_name = d.release_name AND s.reference_date = d.reference_date
    WHERE NOT EXISTS (
        SELECT 1 FROM eco.release_vintages v
        WHERE v.series_name = d.series_name
          AND v.reference_date = d.reference_date)

    UNION ALL

    SELECT to_char(e.release_date::timestamptz, 'YYYYMMDD')::bigint AS release_id,
           'EIA Weekly Petroleum Status Report'::text AS release_name,
           e.series_key AS series_name, 'EIA WPSR'::text AS source,
           e.series_key AS source_series_id, e.release_date, e.reference_date,
           NULL::date AS vintage_end, e.value,
           e.release_timestamp_confidence AS release_date_confidence,
           jsonb_strip_nulls(jsonb_build_object(
               'table_number', e.table_number,
               'table_title', e.table_title,
               'dimension_1', e.dimension_1,
               'dimension_2', e.dimension_2,
               'measure_variant', e.measure_variant,
               'raw_value', e.raw_value,
               'source_url', e.source_url,
               'payload_sha256', e.payload_sha256,
               'retrieved_at', e.retrieved_at)) AS source_metadata
    FROM eco.eia_wpsr_observations e
) r
WITH NO DATA;

CREATE INDEX IF NOT EXISTS release_data_pit_cache_lookup_idx
    ON eco._release_data_pit_cache
       (source, source_series_id, reference_date, release_date DESC);

CREATE UNIQUE INDEX IF NOT EXISTS release_data_pit_cache_row_key_idx
    ON eco._release_data_pit_cache (row_key);

-- On a fresh database, populate once without CONCURRENTLY before using the
-- concurrent refresh job: REFRESH MATERIALIZED VIEW eco._release_data_pit_cache;
