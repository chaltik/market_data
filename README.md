
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

**Cron (daily):**
```cron
5 0 * * * cd /path/to/market_data && PYTHONPATH=. python -m nasdaq_100_history.update_ndx_history >> logs/ndx_update.log 2>&1
```

### Convenience daily update script ###

`nasdaq_100_history/ndx_daily_update.sh` will first update the index constituents from QQQ holdings. It then generates a YAML file containing all relevant symbols and runs the data retrieval operation for them.

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

