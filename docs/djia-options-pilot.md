# DJIA options pilot

The collector records Yahoo Finance option quotes for the **current 30 DJIA
constituents** in `coredata.options_data.quote_snapshots`. It captures every
available expiration, both calls and puts, once per scheduled hour. This is a
forward-only quote archive, **not** historical option data. The timestamp
`captured_at` records when each expiration chain was fetched;
`last_trade_at` is Yahoo's trade timestamp, not a quote timestamp.

The one-week pilot runs November 4–10, 2026 on the five trading days at
10:00–16:00 Eastern, inclusive (seven polls per day). The installed cron line
is documented in `options_pilot.cron`. After November 10, the collector's
date guard makes the entry inert; remove it from crontab when reviewing the
pilot. The independent market-hours guard prevents weekend requests.

Only a contract meeting **all** of these thresholds is stored:

- positive bid and ask at least bid;
- bid–ask spread no more than 25% of their midpoint;
- open interest at least 100 contracts.

These are configurable CLI defaults. `collection_runs` retains the filter
settings and, for every symbol, expiration and contract counts before/after
filtering. Partial responses and rate-limit errors are marked explicitly;
the collector stops requesting more chains on a 429. It never labels a
partial surface as complete. Yahoo may serve delayed or stale quotes, and
the feed supplies no reliable bid/ask timestamp.

Create the tables on a new database with `create_options_data_tables.sql`.
It is non-destructive and may be rerun. For a read-only sample:

```bash
python collect_djia_options.py --symbols AAPL --max-expirations 1 --dry-run
```

Measure the pilot after a week:

```sql
SELECT count(*) AS quotes,
       pg_size_pretty(pg_total_relation_size('options_data.quote_snapshots')) AS table_with_indexes
FROM options_data.quote_snapshots;

SELECT date_trunc('hour', started_at) AS hour_in_session_timezone,
       status, symbols_completed, symbols_requested,
       expirations_completed, expirations_seen,
       contracts_retained, contracts_seen,
       table_bytes_after - table_bytes_before AS bytes_added
FROM options_data.collection_runs ORDER BY started_at;
```

The second query's timestamps are stored in UTC and rendered in the database
session timezone; convert to `America/New_York` if desired. Compare rows per
hour, failed expirations, filtered share, and actual table/index growth before
choosing a twice-daily schedule. Sources: [yfinance option chain API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html)
and [upstream field implementation](https://github.com/ranaroussi/yfinance/blob/main/yfinance/ticker.py).
