-- Compare a transaction-local eco_schema_probe, built from the versioned
-- create scripts, with the existing eco schema. Run before ROLLBACK.
DO $$
DECLARE
    differences BIGINT;
BEGIN
    WITH items AS (
        SELECT n.nspname AS schema_name, c.relname AS relation_name,
               c.relkind, a.attname AS column_name,
               format_type(a.atttypid, a.atttypmod) AS column_type,
               a.attnotnull AS not_null,
               COALESCE(pg_get_expr(d.adbin, d.adrelid), '') AS default_expr
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid
        LEFT JOIN pg_attrdef d ON d.adrelid = c.oid AND d.adnum = a.attnum
        WHERE n.nspname IN ('eco', 'eco_schema_probe')
          AND c.relkind IN ('r', 'm', 'v')
          AND a.attnum > 0 AND NOT a.attisdropped
    ), live AS (
        SELECT relation_name, relkind, column_name, column_type,
               not_null, default_expr FROM items WHERE schema_name = 'eco'
    ), probe AS (
        SELECT relation_name, relkind, column_name, column_type,
               not_null, default_expr FROM items
        WHERE schema_name = 'eco_schema_probe'
    )
    SELECT count(*) INTO differences FROM (
        (SELECT * FROM live EXCEPT SELECT * FROM probe)
        UNION ALL
        (SELECT * FROM probe EXCEPT SELECT * FROM live)
    ) diff;
    IF differences <> 0 THEN
        RAISE EXCEPTION 'eco columns/objects differ in % places', differences;
    END IF;

    WITH items AS (
        SELECT n.nspname AS schema_name, t.relname AS table_name,
               c.conname,
               replace(pg_get_constraintdef(c.oid),
                       'eco_schema_probe.', 'eco.') AS definition
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname IN ('eco', 'eco_schema_probe')
    ), live AS (
        SELECT table_name, conname, definition FROM items
        WHERE schema_name = 'eco'
    ), probe AS (
        SELECT table_name, conname, definition FROM items
        WHERE schema_name = 'eco_schema_probe'
    )
    SELECT count(*) INTO differences FROM (
        (SELECT * FROM live EXCEPT SELECT * FROM probe)
        UNION ALL
        (SELECT * FROM probe EXCEPT SELECT * FROM live)
    ) diff;
    IF differences <> 0 THEN
        RAISE EXCEPTION 'eco constraints differ in % places', differences;
    END IF;

    WITH items AS (
        SELECT schemaname, tablename, indexname,
               replace(indexdef, 'eco_schema_probe.', 'eco.') AS definition
        FROM pg_indexes
        WHERE schemaname IN ('eco', 'eco_schema_probe')
    ), live AS (
        SELECT tablename, indexname, definition FROM items
        WHERE schemaname = 'eco'
    ), probe AS (
        SELECT tablename, indexname, definition FROM items
        WHERE schemaname = 'eco_schema_probe'
    )
    SELECT count(*) INTO differences FROM (
        (SELECT * FROM live EXCEPT SELECT * FROM probe)
        UNION ALL
        (SELECT * FROM probe EXCEPT SELECT * FROM live)
    ) diff;
    IF differences <> 0 THEN
        RAISE EXCEPTION 'eco indexes differ in % places', differences;
    END IF;

    WITH items AS (
        SELECT n.nspname AS schema_name, c.relname AS relation_name,
               replace(pg_get_viewdef(c.oid, true),
                       'eco_schema_probe.', 'eco.') AS definition
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname IN ('eco', 'eco_schema_probe')
          AND c.relkind IN ('m', 'v')
    ), live AS (
        SELECT relation_name, definition FROM items WHERE schema_name = 'eco'
    ), probe AS (
        SELECT relation_name, definition FROM items
        WHERE schema_name = 'eco_schema_probe'
    )
    SELECT count(*) INTO differences FROM (
        (SELECT * FROM live EXCEPT SELECT * FROM probe)
        UNION ALL
        (SELECT * FROM probe EXCEPT SELECT * FROM live)
    ) diff;
    IF differences <> 0 THEN
        RAISE EXCEPTION 'eco views differ in % places', differences;
    END IF;

    RAISE NOTICE 'eco schema parity verified: relations, columns, constraints, indexes and views match';
END $$;
