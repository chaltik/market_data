#!/usr/bin/env bash
set -euo pipefail

# Use PGHOST, PGPORT, PGUSER, PGDATABASE and PGPASSWORD/PGPASSFILE (or a
# PGSERVICE) for the target coredata database. All probe DDL is rolled back.
cd "$(dirname "$0")"
rewrite() {
  sed -e 's/eco\./eco_schema_probe./g' \
      -e 's/EXISTS eco;/EXISTS eco_schema_probe;/' "$1"
}

psql -X -v ON_ERROR_STOP=1 \
  -c 'BEGIN' \
  -f <(rewrite create_eco_data_tables.sql) \
  -f <(rewrite create_imf_gold_tables.sql) \
  -f <(rewrite create_eco_release_data_pit_cache.sql) \
  -f <(rewrite create_eco_release_data_unified_view.sql) \
  -f tests/verify_eco_schema_parity.sql \
  -c 'ROLLBACK'
