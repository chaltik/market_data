# Dow Jones membership

`DJIA` uses the existing `market_indices` tables. No new table or schema
migration is required. Do **not** rerun `create_market_indices_tables.sql` on a
populated database: that bootstrap script drops the index tables first.

## One-off history import

From the repository root, with the project's Python environment and
`database.ini` configured:

```bash
python -m djia_history.load_djia --history --daily --dry-run
python -m djia_history.load_djia --history --daily
```

The first dated baseline is May 6, 1991. Each subsequent dated Wikipedia
component table is an effective-date snapshot; the loader requires exactly 30
distinct names and an explicit ticker mapping in `company_tickers.yaml`.
It rejects unfamiliar names and cross-checks the latest history against DIA.
All database inserts are one transaction. A repeat import validates the
existing snapshots and leaves them unchanged.

The Wikipedia tables are *sparse change-date snapshots*, not daily ETF
holdings. Membership between dates is inferred from adjacent snapshots. Some
predecessor/defunct tickers (for example `WX`, `BS`, `XON`, and `KFT`) may not
be available from a present-day price vendor; membership coverage is not
historical price coverage. Verify event dates and legacy identities against
S&P Dow Jones announcements before using this reconstruction for production
backtests.

## Forward update

`market_data_daily_update.sh` runs `python -m djia_history.load_djia --daily`
before the shared price retrieval. The loader reads the **published as-of
date** in State Street's DIA workbook, excludes its cash row, requires 30
stocks, rejects files more than one weekday behind (counting holidays
conservatively), and appends new dated snapshots. The generated
DJIA asset YAML is merged with NDX, SP500, and the ETF lists through a set, so
each overlapping ticker is fetched only once in that run.

The latest stored universe can be checked with:

```sql
SELECT symbol FROM market_indices.constituents_asof('DJIA', CURRENT_DATE);
SELECT as_of_date, COUNT(*) FROM market_indices.seed_snapshots s
JOIN market_indices.indices i USING (index_id)
WHERE i.code = 'DJIA' GROUP BY as_of_date ORDER BY as_of_date DESC;
```

Sources: [Wikipedia historical components](https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average),
[State Street DIA daily holdings](https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-dia.xlsx),
[S&P Dow Jones Indices DJIA](https://www.spglobal.com/spdji/en/indices/equity/dow-jones-industrial-average/).
