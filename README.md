
# **Market Data Pipeline (`market_data`)**
🚀 A standalone project for retrieving, storing, and updating **historical & daily market data** for **equities, ETFs, and cryptocurrencies** using **Tiingo API**.

## **📌 Features**
- ✅ **Retrieves historical & daily market data** for equities, ETFs, and cryptocurrencies.
- ✅ **Automatically detects new tickers** and fetches their metadata.
- ✅ **Supports incremental updates** (only fetches new data).
- ✅ **Stores data in PostgreSQL** with structured tables.
- ✅ **Uses YAML for easy asset management.**
- ✅ **Automated with `cron` for daily updates.**

---
## **📂 Project Structure**
```
market_data/
│── data_retrieval.py   # Main script to fetch and store market data
│── macro_assets.yaml   # List of tracked equities & crypto (configurable)
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

Provide Required Config Files
Ensure the following files exist in the project root:

✅ .env (API Key)
Create a file named .env and add:
```
TIINGO_API_KEY=my_tiingo_api_key
```
✅ database.ini (PostgreSQL Connection)
Create a file named database.ini with:
```
[postgresql]
host=your_db_host
database=your_db_name
user=your_db_user
password=your_db_password
port=your_db_port
```
🚀 Running the Script
Use the following command to run data_retrieval.py:

`eval $(cat .env) PYTHONPATH=. python data_retrieval.py`

Or specify a custom asset file:

`eval $(cat .env) PYTHONPATH=. python data_retrieval.py --assets_file=my_assets.yaml`

⚙ Automating with Cron Job
To schedule daily updates, add this line to crontab -e:

`0 0 * * * cd /path/to/market_data && eval $(cat .env) PYTHONPATH=. python data_retrieval.py`

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

**Cron (daily Nasdaq-100 maintenance):**
```cron
5 0 * * * cd /path/to/market_data && PYTHONPATH=. python -m nasdaq_100_history.update_ndx_history >> logs/ndx_update.log 2>&1
```

The repository's scheduled market-data loader is `commodity_etf_daily.cron`. Its
18:00 Pacific job now refreshes the Nasdaq-100 universe from QQQ, generates the
current constituent list, pulls Nasdaq-100 prices, and then refreshes the
commodity/ETF universe. It uses the pyenv `market_data` environment and runs
both price loads with `--assets-only`.

### Convenience daily update script ###

`nasdaq_100_history/ndx_daily_update.sh` performs the Nasdaq-100 portion alone:
it updates the index constituents from QQQ holdings, generates a YAML file for
the current members, and runs the incremental price pull.

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

### FRED high-yield OAS

`hy_oas.py` stores FRED series `BAMLH0A0HYM2` in the existing
`eco.release_content`, `eco.release_schedule`, and `eco.release_data` tables.
Run the historical import once, then append the line in `hy_oas_daily.cron`
to the existing user crontab for a 07:00 Pacific daily refresh. The script
uses the same `.env`, `database.ini`, and pyenv `market_data` environment as
the other loaders.

```bash
PYENV_VERSION=market_data pyenv exec python hy_oas.py backfill \
  --archive /home/captain/projects/strader/etf_factors/data/hy_oas_fred_archive.csv
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

