# Monthly U.S. weather data

Run `python weather/backfill_monthly_us_weather.py` for a dry run. Add `--apply`
to save an immutable source snapshot and its parsed observations to
`coredata.weather` (the script creates the schema using
`create_weather_tables.sql`). Re-run the loader after NOAA updates its current
files; each changed source hash creates a new snapshot bundle rather than
overwriting prior retrieved values.

The current loader acquires NOAA nClimDiv statewide mean temperature, HDD, and
CDD files from 1973 onward, plus the NOAA decennial climate-division population
file and GHCN-Daily station `USW00013743` for an explicitly flagged DC-metro
proxy. Alaska and Hawaii are excluded. The four-region crosswalk is the U.S.
Census Northeast, Midwest, South, and West mapping; DC is in the South. NOAA
does not publish DC as a state row in these nClimDiv statewide files, so the
Washington National Airport station (in Arlington, Virginia) is used only as a
proxy and is flagged as such.

## Interpretation and point-in-time limits

- NOAA nClimDiv statewide mean temperatures are spatially area-weighted
  averages. Regional means in this project combine those state means with
  fixed 2010 Census state-population weights; this is a retrospective,
  population-weighted summary of state area means, not a grid-level
  population-weighted temperature.
- HDD/CDD are NOAA source metrics based on daily temperatures; they are not
  calculated by thresholding the monthly mean. The NOAA documentation says the
  operational degree-day series can differ from degree days in official
  publications, and its population weights use the 2010 Census. The source
  values are retained unchanged.
- `weather.source_snapshots` keeps the original downloaded files, SHA-256,
  NOAA processing date when encoded in the filename, and retrieval timestamp.
  `release_date` stays NULL and confidence is `unknown`: these current files do
  not provide historical publication dates or vintages. NOAA updates monthly
  and may revise the latest two calendar years. The current snapshot is
  therefore useful for descriptive analysis, but is not a genuinely
  point-in-time historical dataset. Do not expose same-month realized weather
  at the beginning of that month.
- `weather.monthly_observations` retains the source-state rows and has one
  derived row for each Census region/month. Check `population_coverage_ratio`,
  `expected_member_count`, and `observed_member_count` before using a region.
  The South row has `is_proxy = true` because it includes the DC station proxy.

## Sources

- [NOAA nClimDiv metadata](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00005)
- [NOAA current nClimDiv statewide files and documentation](https://www.ncei.noaa.gov/monitoring-content/data/us/climdiv/monthly/current/)
- [NOAA GHCN-Daily station archive](https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USW00013743.dly)
- [EIA degree-day definition and population-weighted historical benchmark](https://www.eia.gov/energyexplained/units-and-calculators/degree-days.php)

EIA’s monthly population-weighted degree-day benchmark starts in 1973 but is
published for nine Census divisions rather than the four Census regions. It is
useful for validation; it is not silently substituted for the NOAA source
series in this loader.
