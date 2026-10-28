# Daily volatility indices

The daily loader stores VIX, MOVE, GVZ, and OVX together in
`market_indices.volatility_indices`. The primary key is `(date, index_symbol)`;
`source` and `source_series_id` preserve how each value was retrieved.

| `index_symbol` | Source | Source identifier |
| --- | --- | --- |
| `VIX` | FRED | `VIXCLS` |
| `MOVE` | Yahoo Finance | `^MOVE` |
| `GVZ` | FRED | `GVZCLS` |
| `OVX` | FRED | `OVXCLS` |

Run `create_volatility_index_tables.sql` once to create the consolidated table.
It copies any rows in the legacy per-index tables, replaces the old `vix` table
with a compatibility view, and removes the legacy MOVE/GVZ/OVX tables. The
scheduled daily loader backfills available history on first run and upserts
subsequent values by date and index symbol. Existing VIX read queries can
continue to use `market_indices.vix`; write new data to the consolidated table.

For one series, query it directly:

```sql
SELECT date, value
FROM market_indices.volatility_indices
WHERE index_symbol = 'VIX'
ORDER BY date;
```
