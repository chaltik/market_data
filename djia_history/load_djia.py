"""Backfill dated DJIA snapshots and maintain the forward DIA snapshot.

Historical Wikipedia tables give full membership, not weights. DIA is an
approximate forward proxy; its cash holding is deliberately excluded.
"""

import argparse
import hashlib
import io
import re
from datetime import date, datetime
from pathlib import Path

import psycopg2
import requests
import yaml
from bs4 import BeautifulSoup
from openpyxl import load_workbook

from db_config import config

CODE = "DJIA"
HISTORY_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average"
DIA_URL = "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-dia.xlsx"
MAPPING_PATH = Path(__file__).with_name("company_tickers.yaml")
USER_AGENT = "market-data-djia/1.0 (membership research)"
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,7}$")


def fetch(url):
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=40)
    response.raise_for_status()
    return response.content


def parse_history(html, mapping):
    soup = BeautifulSoup(html, "html.parser")
    snapshots = []
    for heading in soup.find_all("h2"):
        try:
            day = datetime.strptime(heading.get_text(" ", strip=True), "%B %d, %Y").date()
        except ValueError:
            continue
        if day < date(1991, 1, 1):
            continue
        table = heading.parent.find_next_sibling("table")
        if table is None or "wikitable" not in table.get("class", []):
            raise ValueError(f"Missing historical component table for {day}")
        names = []
        for row in table.find_all("tr"):
            if "Dropped from Average" in row.get_text(" ", strip=True):
                break
            for cell in row.find_all("td"):
                name = re.split(r"\s*[↑↓†]", cell.get_text(" ", strip=True), maxsplit=1)[0].strip()
                if name:
                    names.append(name)
        if len(names) != 30 or len(set(names)) != 30:
            raise ValueError(f"{day}: expected 30 distinct company names; got {len(names)}")
        unknown = sorted(set(names) - mapping.keys())
        if unknown:
            raise ValueError(f"{day}: unmapped companies: {unknown}")
        members = {mapping[name]: name for name in names}
        if len(members) != 30 or any(not SYMBOL_RE.fullmatch(symbol) for symbol in members):
            raise ValueError(f"{day}: duplicate or invalid mapped ticker")
        snapshots.append((day, members))
    snapshots.sort()
    if not snapshots or snapshots[0][0] != date(1991, 5, 6):
        raise ValueError("Historical source did not contain the expected 1991 baseline")
    return snapshots


def parse_dia(content):
    sheet = load_workbook(io.BytesIO(content), read_only=True, data_only=True).active
    rows = sheet.iter_rows(values_only=True)
    asof = None
    headers = None
    members = {}
    weights = {}
    for row in rows:
        if row[0] == "Holdings:":
            asof = datetime.strptime(str(row[1]), "As of %d-%b-%Y").date()
        if row[0] == "Name" and row[1] == "Ticker":
            headers = {str(value): i for i, value in enumerate(row) if value is not None}
            continue
        if headers is None:
            continue
        ticker = row[headers["Ticker"]]
        if not ticker or ticker == "-":
            continue
        symbol = str(ticker).strip().upper()
        if not SYMBOL_RE.fullmatch(symbol) or symbol in members:
            raise ValueError(f"DIA contains invalid/duplicate ticker {symbol!r}")
        members[symbol] = str(row[headers["Name"]]).strip()
        weights[symbol] = float(row[headers["Weight"]])
    if asof is None or len(members) != 30 or not 98 <= sum(weights.values()) <= 101:
        raise ValueError(f"DIA workbook is incomplete: date={asof}, stocks={len(members)}")
    if asof > date.today():
        raise ValueError(f"DIA workbook is future dated: {asof}")
    return asof, members, weights


def check_recent_agreement(snapshots, dia_members):
    expected = set(snapshots[-1][1])
    actual = set(dia_members)
    if expected != actual:
        raise ValueError(f"Latest historical snapshot differs from DIA: wiki_only={sorted(expected - actual)}, dia_only={sorted(actual - expected)}")


def weekday_sessions_since(asof, today):
    """Count elapsed weekdays; holidays are conservatively counted as sessions."""
    from datetime import timedelta
    day = asof
    count = 0
    while day < today:
        day += timedelta(days=1)
        if day.weekday() < 5:
            count += 1
    return count


def upsert_source(cur, kind, title, url, digest):
    cur.execute("""
        INSERT INTO market_indices.sources(source_type,title,url,checksum)
        VALUES (%s,%s,%s,%s)
        ON CONFLICT (url) DO UPDATE SET title=EXCLUDED.title, checksum=EXCLUDED.checksum
        RETURNING source_id
    """, (kind, title, url, digest))
    return cur.fetchone()[0]


def ensure_index(cur):
    cur.execute("""
        INSERT INTO market_indices.indices(code,name,provider,methodology_url,notes)
        VALUES (%s,%s,%s,%s,%s)
        ON CONFLICT (code) DO UPDATE SET name=EXCLUDED.name
        RETURNING index_id
    """, (CODE, "Dow Jones Industrial Average", "S&P Dow Jones Indices",
          "https://www.spglobal.com/spdji/en/indices/equity/dow-jones-industrial-average/",
          "Historical membership: dated Wikipedia snapshots from 1991-05-06; forward proxy: DIA holdings."))
    return cur.fetchone()[0]


def apply_snapshot(cur, index_id, asof, members, source_id, weights=None):
    """Append a complete snapshot and advance half-open membership intervals."""
    cur.execute("SELECT max(as_of_date) FROM market_indices.seed_snapshots WHERE index_id=%s", (index_id,))
    latest = cur.fetchone()[0]
    if latest and asof < latest:
        raise ValueError(f"Refusing out-of-order DJIA snapshot {asof}; latest is {latest}")
    if latest == asof:
        cur.execute("SELECT symbol FROM market_indices.seed_snapshots WHERE index_id=%s AND as_of_date=%s", (index_id, asof))
        old = {row[0] for row in cur.fetchall()}
        if old != set(members):
            raise ValueError(f"Refusing to rewrite conflicting DJIA snapshot {asof}")
        return False
    cur.execute("SELECT symbol FROM market_indices.index_constituents WHERE index_id=%s AND end_date IS NULL", (index_id,))
    active = {row[0] for row in cur.fetchall()}
    if latest and len(active) != 30:
        raise ValueError(f"DJIA active membership is damaged: {len(active)} rows")
    for symbol, name in sorted(members.items()):
        cur.execute("""
            INSERT INTO market_indices.seed_snapshots(index_id,as_of_date,symbol,company_name,weight,source_id)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (index_id, asof, symbol, name, None if weights is None else weights[symbol], source_id))
    for action, symbols in (("REMOVE", active - members.keys()), ("ADD", members.keys() - active)):
        for symbol in sorted(symbols):
            digest = hashlib.sha256(f"DJIA|{action}|{symbol}|{asof}".encode()).hexdigest()
            cur.execute("""
                INSERT INTO market_indices.index_changes_raw
                  (index_id,action,company_name,symbol,effective_date,effective_session,source_id,event_hash,notes)
                VALUES (%s,%s,%s,%s,%s,'PRIOR_TO_OPEN',%s,%s,%s)
                ON CONFLICT (event_hash) DO UPDATE SET event_hash=EXCLUDED.event_hash
                RETURNING change_id
            """, (index_id, action, members.get(symbol) if action == "ADD" else None,
                  symbol, asof, source_id, digest, "Inferred from adjacent full membership snapshots"))
            change_id = cur.fetchone()[0]
            if action == "REMOVE":
                cur.execute("""
                    UPDATE market_indices.index_constituents SET end_date=%s,removed_change_id=%s
                    WHERE index_id=%s AND symbol=%s AND end_date IS NULL
                """, (asof, change_id, index_id, symbol))
                if cur.rowcount != 1:
                    raise ValueError(f"Expected one active row for removed {symbol}")
            else:
                cur.execute("""
                    INSERT INTO market_indices.index_constituents
                      (index_id,symbol,company_name,start_date,added_change_id,last_verified_asof)
                    VALUES (%s,%s,%s,%s,%s,%s)
                """, (index_id, symbol, members[symbol], asof, change_id, asof))
    cur.execute("SELECT count(*) FROM market_indices.index_constituents WHERE index_id=%s AND end_date IS NULL", (index_id,))
    count = cur.fetchone()[0]
    if count != 30:
        raise ValueError(f"DJIA membership became {count}, expected 30")
    return True


def history_already_loaded(cur, index_id, snapshots):
    """Allow a repeat historical run only when every dated snapshot matches."""
    cur.execute("SELECT max(as_of_date) FROM market_indices.seed_snapshots WHERE index_id=%s", (index_id,))
    latest = cur.fetchone()[0]
    if latest is None:
        return False
    for asof, members in snapshots:
        cur.execute("SELECT symbol FROM market_indices.seed_snapshots WHERE index_id=%s AND as_of_date=%s", (index_id, asof))
        existing = {row[0] for row in cur.fetchall()}
        if existing != set(members):
            raise ValueError(f"Existing DJIA history differs or is incomplete at {asof}; manual reconciliation required")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="Import 1991-onward dated Wikipedia snapshots")
    parser.add_argument("--daily", action="store_true", help="Append the published DIA holdings snapshot")
    parser.add_argument("--dry-run", action="store_true", help="Validate sources only; do not connect to database")
    parser.add_argument("--db-config", help="Path to database.ini, if not in repository root")
    args = parser.parse_args()
    if not (args.history or args.daily):
        parser.error("choose --history and/or --daily")
    snapshots = []
    html = None
    if args.history:
        html = fetch(HISTORY_URL)
        mapping = yaml.safe_load(MAPPING_PATH.read_text())
        snapshots = parse_history(html, mapping)
        print(f"Historical source: {len(snapshots)} complete snapshots, {snapshots[0][0]} through {snapshots[-1][0]}")
    dia = None
    raw_dia = None
    if args.daily:
        raw_dia = fetch(DIA_URL)
        dia = parse_dia(raw_dia)
        if args.history:
            check_recent_agreement(snapshots, dia[1])
        lag = weekday_sessions_since(dia[0], date.today())
        if lag > 1:
            raise ValueError(f"DIA workbook is more than one weekday behind: {dia[0]}")
        print(f"DIA source: {dia[0]}, {len(dia[1])} stocks, {lag} weekday(s) behind")
    if args.dry_run:
        return
    params = config(filename=args.db_config) if args.db_config else config()
    with psycopg2.connect(**params) as conn:
        with conn.cursor() as cur:
            index_id = ensure_index(cur)
            if args.history:
                source_id = upsert_source(cur, "WIKIPEDIA", "Dow historical full component snapshots", HISTORY_URL, hashlib.sha256(html).hexdigest())
                if not history_already_loaded(cur, index_id, snapshots):
                    for asof, members in snapshots:
                        apply_snapshot(cur, index_id, asof, members, source_id)
            if args.daily:
                source_id = upsert_source(cur, "ETF", "State Street DIA daily holdings", DIA_URL, hashlib.sha256(raw_dia).hexdigest())
                apply_snapshot(cur, index_id, dia[0], dia[1], source_id, dia[2])
            cur.execute("SELECT count(*) FROM market_indices.index_constituents WHERE index_id=%s AND end_date IS NULL", (index_id,))
            print(f"DJIA active constituents: {cur.fetchone()[0]}")


if __name__ == "__main__":
    main()
