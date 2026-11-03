
# **Market Data Pipeline (`market_data`)**
🚀 A standalone project for retrieving, storing, and updating **historical & daily market data** for **equities, ETFs, and cryptocurrencies** using **Tiingo API**.

## **📌 Features**
- ✅ **Retrieves historical & daily market data** for equities, ETFs, and cryptocurrencies.
- ✅ **Automatically detects new tickers** and fetches their metadata.
- ✅ **Supports incremental updates** (only fetches new data).
- ✅ **Stores data in PostgreSQL** with structured tables.
- ✅ **Uses YAML for easy asset management.**
- ✅ **Command-line loaders for scheduled or manual updates.**

---
## **📂 Project Structure**
```
market_data/
│── data_retrieval.py   # Main script to fetch and store market data
│── macro_assets.example.yaml # Example asset configuration
│── macro_assets.yaml   # Local download selection (ignored by Git)
│── .env                # API key (ignored by Git)
│── database.ini        # Database connection config (ignored by Git)
│── README.md           # This documentation
│── db_utils.py         # Database interaction functions
│── db_config.py        # Database configuration
│── requirements.txt    # Required Python packages
```

---

## **🔧 Setup Instructions**
### **1️⃣ Install Dependencies**
```
pip install -r requirements.txt
```

Create your local asset selection from the example:

```bash
cp macro_assets.example.yaml macro_assets.yaml
```

Download selection YAMLs and `.python-version` are local files ignored by Git.
Use `main` for ongoing development; feature branches are unnecessary unless requested.

Provide Required Config Files
Ensure the following files exist in the project root:

✅ .env (API keys)
Create a file named .env and add the keys for the sources you use:
```
TIINGO_API_KEY=my_tiingo_api_key
FRED_API_KEY=my_fred_api_key
```
✅ database.ini (PostgreSQL Connection)
Create a file named database.ini with:
```
[coredata]
host=your_db_host
database=your_db_name
user=your_db_user
password=your_db_password
port=your_db_port
```
🚀 Running the Script
Use the following command to run data_retrieval.py:

`python data_retrieval.py --assets_file=my_assets.yaml --assets-only`

Or specify a custom asset file:

`python data_retrieval.py --assets_file=my_assets.yaml --assets-only`

Callers can schedule these commands in their own projects using their own
asset lists, logging, and deployment environment.

### **New!**

## **Index Constituents: Methodology & CLI (Nasdaq-100)**

This project includes a scruffy industrial-grade pipeline to **reconstruct and maintain historical index membership** (starting with **NDX / Nasdaq-100**) in PostgreSQL schema **`market_indices`**.

## What we store
- **`market_indices.index_changes_raw`** — atomic ADD/REMOVE events (from Wikipedia), with effective dates + source IDs; deduped by content hash.
- **`market_indices.index_constituents`** — point-in-time membership intervals per symbol `[start_date, end_date)`, with optional `company_name`, `sector` and links to `added_change_id`/`removed_change_id`.
- **`market_indices.seed_snapshots`** — periodic ground-truth snapshots (parsed from **Invesco QQQ CSV**: **Holding Ticker**, Name, Sector).
- **`market_indices.sources` / `market_indices.indices`** — provenance and index registry.

All three main tables include **`sector`**. Constituents have a unique constraint on `(index_id, symbol, start_date, end_date)` so multiple historical intervals per symbol are allowed.

## Data sources & logic
1) **Seed (ground truth) from QQQ CSV**
   - Fetched via **`pandas.read_csv(URL)`**.
   - Robust column picking: prefer **Holding Ticker** (never Fund Ticker).
   - Safety floor: **abort** updates if parsed rows `< 80`.
   - Each run inserts a new snapshot row per symbol into `seed_snapshots`.

2) **Historical events from Wikipedia**
   - `build_ndx_history.py` parses the **“Yearly changes”** tables on the **Nasdaq-100 Wikipedia page**.
   - Each row becomes a change event (`ADD` / `REMOVE`) with effective date; persisted to `index_changes_raw`.

3) **Interval construction**
   - Intervals are computed by walking events in chronological order:
     - **ADD** → starts (or resets) an active interval (no synthetic pre-history).
     - **REMOVE** → closes the active interval at its effective date.
   - Open intervals (`end_date IS NULL`) represent current membership.
   - Latest seed info (name/sector) is propagated into `index_constituents`.

4) **Daily updates (maintenance)**
   - `update_ndx_history.py` diffs **today’s QQQ** vs **previous snapshot**:
     - **Removed** → close open interval at `prev_asof + 1 day` (conservative).
     - **Added** → open a new interval at the same boundary.
     - **Continued** → update `company_name/sector` if missing; set `last_verified_asof`.
   - Safety: abort if snapshot looks suspicious (e.g., row count too small).

5) **Querying membership as of a date**
   - Official (event-driven) membership:
     ```sql
     SELECT * FROM market_indices.constituents_asof('NDX', DATE '2025-09-08');
     ```
   - Seed-only (exact ETF snapshot on/before the date):
     ```sql
     SELECT * FROM market_indices.constituents_asof_seed('NDX', DATE '2025-09-08');
     ```

> We intentionally **don’t fetch Nasdaq PRs** right now (rate limits/timeouts). Wikipedia + QQQ snapshots yield robust, auditable results.

---

## 🛠 CLI (Constituents)

### One-time (or ad-hoc) build from Wikipedia + QQQ
```bash
PYTHONPATH=. python -m nasdaq_100_history.build_ndx_history
```
- Ingests Wikipedia changes → `index_changes_raw`
- Seeds from QQQ → `seed_snapshots`
- Computes intervals → `index_constituents`

### Daily updater from QQQ (safe diff)
```bash
PYTHONPATH=. python -m nasdaq_100_history.update_ndx_history
```
Optional args:
```bash
# Index code (default: NDX)
PYTHONPATH=. python -m nasdaq_100_history.update_ndx_history --code NDX

# Backfill or force a specific snapshot date
PYTHONPATH=. python -m nasdaq_100_history.update_ndx_history --asof 2025-09-08
```

The index loader can be called directly or included in a caller-owned schedule.

### Convenience daily update script ###

`nasdaq_100_history/ndx_daily_update.sh` performs the Nasdaq-100 portion alone:
it updates the index constituents from QQQ holdings, generates a YAML file for
the current members, and runs the incremental price pull.

## Approximate S&P 500 large-cap universe

`sp500_history/` maintains `SP500`, an approximately 500-name US large-cap universe.
It is intentionally not a claim of exact S&P 500 membership or weights.

- Historical backfill: the public point-in-time `date,tickers` snapshots in
  [thuningxu/sp500nq100](https://github.com/thuningxu/sp500nq100), beginning 1996-01-02.
  Every supplied snapshot is retained in `market_indices.seed_snapshots`; transitions
  become point-in-time intervals, so removed names and ticker changes remain available.
- Daily maintenance: State Street's official [SPY daily holdings](https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy)
  XLSX. The parser accepts the workbook's metadata rows and validates a 450–550-name universe.

Run the initial backfill, then the daily update:

```bash
PYTHONPATH=. python -m sp500_history.build_sp500_history
PYTHONPATH=. python -m sp500_history.update_sp500_history
PYTHONPATH=. python -m sp500_history.gen_sp500_yaml --mode asof --asof 2026-10-16 --output /tmp/sp500.yaml
```

`sp500_history/sp500_daily_update.sh` applies the update, creates an as-of YAML,
and runs the price retrieval job. `market_data_daily_update.sh POLICY.yaml`
refreshes stock-index constituents and general ETFs in separate yfinance passes
into `price_data.equities_us_daily`, then strictly checks the caller's required
ETF list through Tiingo. `required_etf_repair.sh POLICY.yaml` retries missing
completed closes. Set `IB_ETF_FALLBACK=1` to permit a read-only IB attempt if
Tiingo still fails; that optional path requires `ib_async` and a reachable IB
Gateway or TWS. Missing IB exits with a clear error and leaves the audit failed.
Each missing ticker and verified source is logged. Both commands require the caller's policy path; this repository does
not own any application's ETF list or schedule. Set `MARKET_DATA_PYTHON` to
choose the interpreter, or let the scripts use `python` on the caller's path.

The S&P-only script can still be run independently with
`./sp500_history/sp500_daily_update.sh`.

Query historical membership with the shared helper:

```sql
SELECT * FROM market_indices.constituents_asof('SP500', DATE '2005-01-03');
```

### 🔍 SQL Helpers

The helper function `market_indices.constituents_asof(code text, p_asof date)` is defined in `create_market_indices_tables.sql`. Below are usage examples.

Get constituents for a specific date

```sql
SELECT symbol, sector, company_name
FROM market_indices.constituents_asof('NDX', DATE '2025-09-08')
ORDER BY symbol;
```

Get latest constituents

```sql
SELECT symbol, sector, company_name
FROM market_indices.constituents_asof('NDX')
ORDER BY symbol;
```

### `eco` schema setup and migrations

The schema definitions are versioned SQL, not embedded in data loaders. Apply
them in this order to a fresh `coredata` database:

```bash
psql -v ON_ERROR_STOP=1 -d coredata -f create_eco_data_tables.sql
psql -v ON_ERROR_STOP=1 -d coredata -f create_imf_gold_tables.sql
psql -v ON_ERROR_STOP=1 -d coredata -f create_eco_release_data_pit_cache.sql
psql -v ON_ERROR_STOP=1 -d coredata -f create_eco_release_data_unified_view.sql
psql -v ON_ERROR_STOP=1 -d coredata -c 'REFRESH MATERIALIZED VIEW eco._release_data_pit_cache'
```

Use your configured PostgreSQL connection options if local peer authentication
is not available. The first two table scripts and cache indexes are rerunnable;
the view script replaces its definition without dropping underlying data. On an
existing database, `create_eco_data_tables.sql` adds older missing
`release_schedule` columns. Run the scripts before running any release loader.
After the initial cache population, subsequent updates can use
`REFRESH MATERIALIZED VIEW CONCURRENTLY eco._release_data_pit_cache`.
Run `verify_eco_schema.sh` with the same PostgreSQL connection settings to
rebuild all definitions in a rollback-only probe and compare tables, columns,
constraints, indexes, and views against the live `eco` schema.

### Load a few FRED economic releases on a fresh machine

Install the Python dependencies from `requirements.txt`, create `database.ini`
with a `[coredata]` section, and put **only** `FRED_API_KEY` in `.env`. The
`--eco-only` path does not need Tiingo, an asset YAML, or the optional
`macro_series.yaml`. Create the eco tables with the SQL commands above, then:

```bash
python data_retrieval.py --eco-only --eco-series GDP,CPIAUCSL
```

Replace those IDs with the FRED tickers you want. Each direct ID becomes the
`fred_series_id`, `series_name`, and `release_name` in
`eco.release_content`; its observations are upserted into
`eco.release_schedule` and `eco.release_data`. Re-running the command updates
the same keys. A failed or empty series makes the command exit unsuccessfully;
check the log and row counts rather than assuming a zero exit means data arrived.
For custom descriptive names, supply `--eco-series-file PATH` pointing to a
YAML catalog with groups of `fred_series_id`, `series_name`, and `release_name`
objects. Unknown selected IDs in that catalog are errors. The dedicated
`hy_oas.py` loader, not this generic command, owns `BAMLH0A0HYM2`.

```sql
SELECT c.fred_series_id, count(*) AS observations,
       min(d.reference_date) AS first_reference,
       max(d.reference_date) AS last_reference
FROM eco.release_data d
JOIN eco.release_content c USING (release_name, series_name)
WHERE c.fred_series_id IN ('GDP', 'CPIAUCSL')
GROUP BY c.fred_series_id;
```

The generic loader stores the latest available value for each reference date
alongside the first ALFRED vintage date (or an explicitly assumed lag for
non-vintage series). It does **not** preserve every historical vintage in
`eco.release_data`; do not interpret those latest values as original release
prints for point-in-time backtests.

### FRED high-yield OAS

`hy_oas.py` stores FRED series `BAMLH0A0HYM2` in the existing
`eco.release_content`, `eco.release_schedule`, and `eco.release_data` tables.
Run the historical import once, then call `hy_oas_daily_update.sh` on the
schedule chosen by the consuming project. The script uses the same `.env` and
`database.ini` as the other loaders.

```bash
PYENV_VERSION=market_data pyenv exec python hy_oas.py backfill \
  --archive /path/to/hy_oas_fred_archive.csv
PYENV_VERSION=market_data pyenv exec python hy_oas.py refresh
```

The archive lacks publication timestamps. For archive rows, `release_date`
means the next NYSE session after `reference_date` and
`release_date_source='inferred_next_nyse_session_from_archive'`. For FRED
rows with ALFRED vintages, availability is the next NYSE session after the
first vintage date (and no earlier than the next session after the observation).
If vintages are unavailable, it is no earlier than the next NYSE session after
the actual fetch time. Recent vintages are also bounded by the first actual
fetch: a value fetched after 11:30 ET is usable at the next session.
`first_fetched_at` records that time. These dates are
conservative estimates, not verified release timestamps. Missing values are
omitted, and the first published nonmissing value for each observation date
is retained on reruns to avoid backtest leakage from later revisions.

An 11:30 ET strategy can use the following query with `release_date <=` the
decision's trade date. Requiring `release_date <` trade date adds another
session of lag to the archive's already inferred one-session lag.

```sql
SELECT d.reference_date AS observation_date, d.release_date AS available_on,
       d.value, s.release_date_source, s.first_fetched_at
FROM eco.release_data d
JOIN eco.release_content c USING (release_name, series_name)
JOIN eco.release_schedule s
  ON s.release_name = d.release_name
 AND s.reference_date = d.reference_date
WHERE c.fred_series_id = 'BAMLH0A0HYM2'
  AND d.value IS NOT NULL
  AND d.release_date <= %(trade_date)s
ORDER BY d.reference_date DESC;
```

