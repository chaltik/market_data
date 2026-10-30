#!/usr/bin/env python3
"""Backfill NOAA nClimDiv monthly state weather and Census-region rollups.

The source is a current/revised NOAA snapshot, not a historical vintage archive.
No publication date is inferred. Run with --apply to write immutable source
files, a snapshot bundle, state observations, and four derived region series.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from psycopg2.extras import Json, execute_values

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from db_config import config
from db_utils import get_connection


BASE_URL = "https://www.ncei.noaa.gov/monitoring-content/data/us/climdiv/monthly/current/"
POP_URL = BASE_URL + "climdiv_pop_1900-2020.dat"
DC_GHCN_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USW00013743.dly"
START_YEAR = 1973
NOAA_FILE_RE = re.compile(r"climdiv-(tmpcst|hddcst|cddcst)-v([\d.]+)-(\d{8})$")

# NOAA's 3-digit nClimDiv statewide identifiers.
STATE_NOAA_CODES = {
    "AL": 1, "AZ": 2, "AR": 3, "CA": 4, "CO": 5, "CT": 6,
    "DE": 7, "FL": 8, "GA": 9, "ID": 10, "IL": 11, "IN": 12,
    "IA": 13, "KS": 14, "KY": 15, "LA": 16, "ME": 17, "MD": 18,
    "MA": 19, "MI": 20, "MN": 21, "MS": 22, "MO": 23, "MT": 24,
    "NE": 25, "NV": 26, "NH": 27, "NJ": 28, "NM": 29, "NY": 30,
    "NC": 31, "ND": 32, "OH": 33, "OK": 34, "OR": 35, "PA": 36,
    "RI": 37, "SC": 38, "SD": 39, "TN": 40, "TX": 41, "UT": 42,
    "VT": 43, "VA": 44, "WA": 45, "WV": 46, "WI": 47, "WY": 48,
}
STATE_NAMES = {
    "AL":"Alabama", "AZ":"Arizona", "AR":"Arkansas", "CA":"California",
    "CO":"Colorado", "CT":"Connecticut", "DE":"Delaware", "FL":"Florida",
    "GA":"Georgia", "ID":"Idaho", "IL":"Illinois", "IN":"Indiana",
    "IA":"Iowa", "KS":"Kansas", "KY":"Kentucky", "LA":"Louisiana",
    "ME":"Maine", "MD":"Maryland", "MA":"Massachusetts", "MI":"Michigan",
    "MN":"Minnesota", "MS":"Mississippi", "MO":"Missouri", "MT":"Montana",
    "NE":"Nebraska", "NV":"Nevada", "NH":"New Hampshire", "NJ":"New Jersey",
    "NM":"New Mexico", "NY":"New York", "NC":"North Carolina", "ND":"North Dakota",
    "OH":"Ohio", "OK":"Oklahoma", "OR":"Oregon", "PA":"Pennsylvania",
    "RI":"Rhode Island", "SC":"South Carolina", "SD":"South Dakota", "TN":"Tennessee",
    "TX":"Texas", "UT":"Utah", "VT":"Vermont", "VA":"Virginia",
    "WA":"Washington", "WV":"West Virginia", "WI":"Wisconsin", "WY":"Wyoming",
}
REGION_MEMBERS = {
    "Northeast": ("CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"),
    "Midwest": ("IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"),
    "South": ("DE", "MD", "DC", "VA", "WV", "NC", "SC", "GA", "FL", "KY", "TN", "MS", "AL", "OK", "TX", "AR", "LA"),
    "West": ("AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "WA", "OR", "CA"),
}
DC_POPULATION_2010 = 601_723


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "market-data-weather-loader/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} fetching {url}")
        return response.read()


def discover_source_files() -> tuple[date, dict[str, tuple[str, bytes]]]:
    parser = LinkParser()
    parser.feed(fetch(BASE_URL).decode("utf-8", "replace"))
    candidates: dict[date, dict[str, str]] = defaultdict(dict)
    for href in parser.hrefs:
        match = NOAA_FILE_RE.fullmatch(href.rsplit("/", 1)[-1])
        if match:
            metric, _version, processing_date = match.groups()
            candidates[date.fromisoformat(f"{processing_date[:4]}-{processing_date[4:6]}-{processing_date[6:8]}")][metric] = href
    complete = [d for d, files in candidates.items() if set(files) == {"tmpcst", "hddcst", "cddcst"}]
    if not complete:
        raise RuntimeError("Could not find a complete, same-vintage set of NOAA nClimDiv state files")
    processing_date = max(complete)
    sources = {}
    for metric, href in candidates[processing_date].items():
        filename = href.rsplit("/", 1)[-1]
        sources[metric] = (filename, fetch(BASE_URL + filename))
    return processing_date, sources


def parse_state_file(payload: bytes, element_code: str, start_year: int = START_YEAR) -> dict[tuple[str, date], float]:
    """Parse NOAA 94-character state records; only state code 001-048 and div 0."""
    expected_element = {"tmpcst": "02", "hddcst": "25", "cddcst": "26"}[element_code]
    code_to_state = {code: abbrev for abbrev, code in STATE_NOAA_CODES.items()}
    values: dict[tuple[str, date], float] = {}
    for raw_line in payload.decode("ascii", "replace").splitlines():
        line = raw_line.ljust(94)
        try:
            state_code = int(line[0:3])
            division_number = line[3]
            element = line[4:6]
            year = int(line[6:10])
        except (ValueError, IndexError):
            continue
        if division_number != "0" or element != expected_element or state_code not in code_to_state or year < start_year:
            continue
        for month in range(1, 13):
            field = line[10 + 7 * (month - 1):17 + 7 * (month - 1)].strip()
            if not field:
                continue
            try:
                value = float(field)
            except ValueError:
                continue
            if value in (-99.90, -9999.0, -9999.00):
                continue
            values[(code_to_state[state_code], date(year, month, 1))] = value
    return values


def parse_population(payload: bytes) -> dict[str, int]:
    """Sum NOAA's climate-division population file to state, taking 2010 census weights."""
    pop: dict[str, int] = defaultdict(int)
    rows = payload.decode("ascii", "replace").splitlines()
    for row in rows:
        fields = row.split()
        if len(fields) < 14 or fields[0] not in STATE_NOAA_CODES:
            continue
        # Tokens 2..14 are decennial years 1900..2020, so index 13 is 2010.
        try:
            pop[fields[0]] += int(fields[13])
        except ValueError:
            continue
    if len(pop) != 48:
        raise RuntimeError(f"Expected 2010 population weights for 48 contiguous states; parsed {len(pop)}")
    return dict(pop)


def parse_dc_station(payload: bytes, start_year: int = START_YEAR) -> dict[date, tuple[float, float, float]]:
    """Derive Washington National Airport monthly mean/HDD/CDD from quality-passed daily extremes."""
    daily_by_month: dict[tuple[date, str], dict[int, float]] = defaultdict(dict)
    for line in payload.decode("ascii", "replace").splitlines():
        if len(line) < 21 or line[:11] != "USW00013743":
            continue
        try:
            year, month = int(line[11:15]), int(line[15:17])
        except ValueError:
            continue
        if year < start_year:
            continue
        element = line[17:21]
        if element not in {"TMAX", "TMIN"}:
            continue
        for day in range(1, 32):
            pos = 21 + (day - 1) * 8
            chunk = line[pos:pos + 8]
            if len(chunk) < 8:
                continue
            try:
                raw_value = int(chunk[:5])
                actual_day = date(year, month, day)
            except ValueError:
                continue
            quality_flag = chunk[6:7]
            if raw_value == -9999 or quality_flag.strip():
                continue
            celsius = raw_value / 10.0
            fahrenheit = celsius * 9.0 / 5.0 + 32.0
            daily_by_month[(date(actual_day.year, actual_day.month, 1), element)][day] = fahrenheit

    out: dict[date, tuple[float, float, float]] = {}
    months = {month for month, _element in daily_by_month}
    for month in months:
        highs = daily_by_month.get((month, "TMAX"), {})
        lows = daily_by_month.get((month, "TMIN"), {})
        daily_mean = [(highs[d] + lows[d]) / 2 for d in highs.keys() & lows.keys()]
        if not daily_mean:
            continue
        out[month] = (
            sum(daily_mean) / len(daily_mean),
            sum(max(0.0, 65.0 - value) for value in daily_mean),
            sum(max(0.0, value - 65.0) for value in daily_mean),
        )
    return out


def weighted_mean(rows: list[tuple[float | None, int]]) -> tuple[float | None, float]:
    valid = [(value, weight) for value, weight in rows if value is not None]
    total_weight = sum(weight for _value, weight in valid)
    all_weight = sum(weight for _value, weight in rows)
    return ((sum(value * weight for value, weight in valid) / total_weight) if total_weight else None,
            (total_weight / all_weight) if all_weight else 0.0)


def parse_snapshot_date(filename: str) -> date | None:
    match = NOAA_FILE_RE.fullmatch(filename)
    if not match:
        return None
    date_text = match.group(3)
    return date.fromisoformat(f"{date_text[:4]}-{date_text[4:6]}-{date_text[6:8]}")


def prepare_data():
    processing_date, files = discover_source_files()
    population_bytes = fetch(POP_URL)
    dc_bytes = fetch(DC_GHCN_URL)
    parsed = {name: parse_state_file(payload, name) for name, (_filename, payload) in files.items()}
    populations = parse_population(population_bytes)
    dc_months = parse_dc_station(dc_bytes)

    all_months = sorted({month for metric_values in parsed.values() for _state, month in metric_values})
    if not all_months:
        raise RuntimeError("No NOAA state observations parsed")
    first_month = date(START_YEAR, 1, 1)
    last_month = max(all_months)
    months = [month for month in all_months if first_month <= month <= last_month]

    state_rows: dict[tuple[str, date], dict] = {}
    for month in months:
        for state in STATE_NOAA_CODES:
            state_rows[(state, month)] = {
                "state": state,
                "month": month,
                "tavg": parsed["tmpcst"].get((state, month)),
                "hdd": parsed["hddcst"].get((state, month)),
                "cdd": parsed["cddcst"].get((state, month)),
                "population": populations[state],
                "region": next(region for region, states in REGION_MEMBERS.items() if state in states),
                "proxy": False,
                "method": "NOAA nClimDiv statewide monthly values",
                "weighting": "NOAA state series; temperature is area-weighted within state; HDD/CDD use NOAA operational degree-day method",
            }
    for month, (tavg, hdd, cdd) in dc_months.items():
        state_rows[("DC", month)] = {
            "state": "DC", "month": month, "tavg": tavg, "hdd": hdd, "cdd": cdd,
            "population": 601_723, "region": "South", "proxy": True,
            "method": "GHCN-Daily USW00013743 Washington National Airport proxy; monthly mean of daily (TMAX+TMIN)/2; HDD/CDD summed from daily means at 65F",
            "weighting": "single-station DC metro proxy; 2010 Census population for regional rollup",
        }

    observation_rows = []
    for (state, month), item in sorted(state_rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        observation_rows.append((
            "state", state, STATE_NAMES.get(state, "District of Columbia"), item["region"], month,
            item["tavg"], item["hdd"], item["cdd"], item["method"], item["weighting"],
            2010, 1.0, 1, 1, item["proxy"],
        ))

    for region, members in REGION_MEMBERS.items():
        expected_weight = sum(populations.get(state, 601_723 if state == "DC" else 0) for state in members)
        for month in months:
            rows = [state_rows[(state, month)] for state in members if (state, month) in state_rows]
            tavg, tavg_coverage = weighted_mean([(row["tavg"], row["population"]) for row in rows])
            hdd, hdd_coverage = weighted_mean([(row["hdd"], row["population"]) for row in rows])
            cdd, cdd_coverage = weighted_mean([(row["cdd"], row["population"]) for row in rows])
            member_weight = sum(row["population"] for row in rows)
            member_coverage = member_weight / expected_weight if expected_weight else 0.0
            coverage = min(member_coverage, tavg_coverage, hdd_coverage, cdd_coverage)
            observation_rows.append((
                "census_region", region.upper(), region, region, month, tavg, hdd, cdd,
                "NOAA nClimDiv state values; DC airport-station proxy included in South",
                "2010 Census population weighted across state aggregates; NOAA state temperature inputs are area-weighted within state; fixed historical weights are retrospective",
                2010, coverage, len(members), len(rows), region == "South",
            ))

    return processing_date, files, population_bytes, dc_bytes, observation_rows


def store_snapshot(cur, *, name, filename, url, payload, processing_date=None, metadata=None, retrieved_at=None):
    retrieved_at = retrieved_at or datetime.now(timezone.utc)
    digest = hashlib.sha256(payload).hexdigest()
    cur.execute(
        """INSERT INTO weather.source_snapshots
           (source_name, source_file, source_url, source_processing_date, retrieved_at,
            sha256, raw_content, release_date, release_date_confidence, metadata)
           VALUES (%s,%s,%s,%s,%s,%s,%s,NULL,'unknown',%s)
           ON CONFLICT (source_name, source_file, sha256) DO UPDATE
             SET source_file=EXCLUDED.source_file
           RETURNING snapshot_id""",
        (name, filename, url, processing_date, retrieved_at, digest, payload, Json(metadata or {})),
    )
    return cur.fetchone()["snapshot_id"]


def persist(processing_date, files, population_bytes, dc_bytes, observations):
    params = config()
    params.update(host="127.0.0.1", port=5432, connect_timeout=10)
    sql = Path(__file__).resolve().parents[1] / "create_weather_tables.sql"
    with get_connection(params) as (_conn, cur):
        cur.execute(sql.read_text())
        retrieved_at = datetime.now(timezone.utc)
        ids = {}
        for metric, (filename, payload) in files.items():
            ids[metric] = store_snapshot(
                cur, name="NOAA_nClimDiv_state", filename=filename, url=BASE_URL + filename,
                payload=payload, processing_date=parse_snapshot_date(filename), retrieved_at=retrieved_at,
                metadata={"metric": metric, "release_date": None, "release_timing": "exact date unknown; product updates monthly and recent two calendar years can revise"},
            )
        pop_id = store_snapshot(
            cur, name="NOAA_nClimDiv_population_weights", filename="climdiv_pop_1900-2020.dat",
            url=POP_URL, payload=population_bytes, retrieved_at=retrieved_at,
            metadata={"weight_year": 2010, "weights": "sum of climate-division population values by state; decennial data", "publication_date": None},
        )
        dc_id = store_snapshot(
            cur, name="NOAA_GHCN_Daily_DC_proxy", filename="USW00013743.dly", url=DC_GHCN_URL,
            payload=dc_bytes, retrieved_at=retrieved_at,
            metadata={"station_id": "USW00013743", "station_name": "Washington National Airport", "proxy_note": "Airport station is in Arlington, VA; it is a DC-metro proxy, not a DC area average", "release_date": None},
        )
        cur.execute(
            """INSERT INTO weather.snapshot_bundles
               (retrieved_at,nclimdiv_processing_date,tavg_snapshot_id,hdd_snapshot_id,cdd_snapshot_id,
                population_snapshot_id,dc_station_snapshot_id,release_date,release_date_confidence,
                availability_note,metadata)
               VALUES (%s,%s,%s,%s,%s,%s,%s,NULL,'unknown',%s,%s)
               ON CONFLICT (tavg_snapshot_id,hdd_snapshot_id,cdd_snapshot_id,population_snapshot_id,dc_station_snapshot_id)
               DO NOTHING
               RETURNING bundle_id""",
            (retrieved_at, processing_date, ids["tmpcst"], ids["hddcst"], ids["cddcst"], pop_id, dc_id,
             "Current NOAA snapshot only; exact historical publication dates/vintages are not supplied by these files. NOAA nClimDiv updates monthly and recent values may be revised. Treat as descriptive, not point-in-time backtest data without a conservative lag and vintage caveat.",
             Json({"data_start_year": START_YEAR, "scope": "contiguous 48 states plus an explicitly marked DC airport-station proxy; Alaska and Hawaii excluded", "processing_date": processing_date.isoformat()})),
        )
        result = cur.fetchone()
        if result:
            bundle_id = result["bundle_id"]
        else:
            cur.execute("""SELECT bundle_id FROM weather.snapshot_bundles
                WHERE tavg_snapshot_id=%s AND hdd_snapshot_id=%s AND cdd_snapshot_id=%s
                  AND population_snapshot_id=%s AND dc_station_snapshot_id=%s""",
                (ids["tmpcst"], ids["hddcst"], ids["cddcst"], pop_id, dc_id))
            bundle_id = cur.fetchone()["bundle_id"]

        crosswalk = []
        for region, members in REGION_MEMBERS.items():
            for state in members:
                if state == "DC":
                    crosswalk.append(("DC", "District of Columbia", region, None, 601_723,
                                      "2010 U.S. Census population; DC station data are an airport proxy", True, True))
                else:
                    pop_payload = population_bytes.decode("ascii", "replace").splitlines()
                    state_population = sum(int(line.split()[13]) for line in pop_payload if line.split()[0] == state and len(line.split()) >= 14)
                    crosswalk.append((state, STATE_NAMES[state], region, STATE_NOAA_CODES[state], state_population,
                                      "NOAA nClimDiv climate-division population file, summed at 2010 Census year", True, False))
        execute_values(cur, """INSERT INTO weather.census_region_state_crosswalk
            (state_abbrev,state_name,census_region,noaa_state_code,population_2010,population_source,included_in_contiguous_us,is_dc_proxy)
            VALUES %s ON CONFLICT(state_abbrev) DO UPDATE SET state_name=EXCLUDED.state_name,
            census_region=EXCLUDED.census_region,noaa_state_code=EXCLUDED.noaa_state_code,
            population_2010=EXCLUDED.population_2010,population_source=EXCLUDED.population_source,
            included_in_contiguous_us=EXCLUDED.included_in_contiguous_us,is_dc_proxy=EXCLUDED.is_dc_proxy""", crosswalk)
        execute_values(cur, """INSERT INTO weather.monthly_observations
            (bundle_id,geography_type,geography_code,geography_name,census_region,reference_month,
             mean_temperature_f,hdd65_f_days,cdd65_f_days,source_method,weighting_method,weighting_year,
             population_coverage_ratio,expected_member_count,observed_member_count,is_proxy,availability_date,availability_confidence)
            VALUES %s ON CONFLICT(bundle_id,geography_type,geography_code,reference_month) DO UPDATE SET
             mean_temperature_f=EXCLUDED.mean_temperature_f,hdd65_f_days=EXCLUDED.hdd65_f_days,
             cdd65_f_days=EXCLUDED.cdd65_f_days,source_method=EXCLUDED.source_method,
             weighting_method=EXCLUDED.weighting_method,weighting_year=EXCLUDED.weighting_year,
             population_coverage_ratio=EXCLUDED.population_coverage_ratio,
             expected_member_count=EXCLUDED.expected_member_count,observed_member_count=EXCLUDED.observed_member_count,
             is_proxy=EXCLUDED.is_proxy""",
            [(bundle_id, *row, None, "unknown") for row in observations], page_size=2000)
        print(f"Stored weather bundle {bundle_id}: {len(observations):,} rows; NOAA processing date {processing_date}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="persist source snapshots and observations in coredata.weather")
    args = parser.parse_args()
    processing_date, files, population_bytes, dc_bytes, observations = prepare_data()
    start = min(row[4] for row in observations)
    end = max(row[4] for row in observations if row[0] == "census_region")
    print(f"NOAA processing date: {processing_date}")
    for metric, (filename, payload) in files.items():
        print(f"{metric}: {filename}, {len(payload):,} bytes, sha256={hashlib.sha256(payload).hexdigest()}")
    print(f"population weights: {len(population_bytes):,} bytes; DC station: {len(dc_bytes):,} bytes")
    print(f"prepared rows: {len(observations):,}, reference range {start:%Y-%m} through {end:%Y-%m}")
    if not args.apply:
        print("Dry run only. Pass --apply to store this immutable source vintage and observations.")
        return
    persist(processing_date, files, population_bytes, dc_bytes, observations)


if __name__ == "__main__":
    main()
