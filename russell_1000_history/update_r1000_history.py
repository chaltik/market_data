"""Append a dated IWB snapshot to the existing R1000 market-indices tables."""

import argparse
import csv
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path

import requests
import yaml

from db_config import config as conn_params
from db_utils import get_connection

INDEX_CODE = "R1000"
HOLDINGS_URL = "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/latest-holdings.csv"
SOURCE_TITLE = "iShares Russell 1000 ETF (IWB) holdings"
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,7}$")
MIN_ROWS, MAX_ROWS = 900, 1150
EXCLUSIONS_FILE = Path(__file__).with_name("constituent_exclusions.yaml")


@dataclass(frozen=True)
class Holding:
    symbol: str
    name: str
    sector: str
    weight: float


def load_excluded_symbols(path: Path = EXCLUSIONS_FILE) -> set[str]:
    """Symbols retained by the ETF source after their public listing ended."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    symbols = payload.get("excluded_symbols", {})
    if not isinstance(symbols, dict) or any(not SYMBOL_RE.fullmatch(str(symbol))
                                            for symbol in symbols):
        raise ValueError(f"Invalid symbol exclusion file: {path}")
    return set(symbols)


def normalize_symbol(raw: str) -> str | None:
    symbol = raw.strip().upper().replace(" ", ".")
    return symbol if SYMBOL_RE.fullmatch(symbol) else None


def parse_holdings(content: str) -> tuple[date, list[Holding], list[str]]:
    rows = list(csv.reader(StringIO(content.lstrip("\ufeff"))))
    metadata = {row[0]: row[1] for row in rows[:10] if len(row) >= 2}
    if "Fund Holdings as of" not in metadata:
        raise ValueError("IWB CSV has no holdings date")
    as_of = datetime.strptime(metadata["Fund Holdings as of"], "%b %d, %Y").date()
    required = {"Ticker", "Name", "Sector", "Asset Class"}
    header = next((n for n, row in enumerate(rows)
                   if required.issubset(row) and ("Weight (%)" in row or "Market Weight" in row)), None)
    if header is None:
        raise ValueError("IWB CSV has no recognized holdings header")
    total_value = sum(float(row.get("Market Value", "0").replace(",", ""))
                      for values in rows[header + 1:] if len(values) == len(rows[header])
                      for row in [dict(zip(rows[header], values))]
                      if row.get("Market Value", "").replace(",", "").replace(".", "", 1).replace("-", "", 1).isdigit())
    holdings: list[Holding] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for values in rows[header + 1:]:
        if len(values) != len(rows[header]):
            continue
        row = dict(zip(rows[header], values))
        if row.get("Asset Class") != "Equity":
            continue
        raw = (row.get("Ticker") or "").strip()
        symbol = normalize_symbol(raw)
        if symbol is None:
            skipped.append(raw)
            continue
        if symbol in seen:
            raise ValueError(f"Duplicate IWB ticker after normalization: {symbol}")
        seen.add(symbol)
        weight = row.get("Weight (%)") or row.get("Market Weight") or ""
        if weight.strip() in ("", "-"):
            if total_value <= 0 or row.get("Market Value", "").strip() in ("", "-"):
                raise ValueError(f"Missing IWB weight for {symbol}")
            weight = str(100 * float(row["Market Value"].replace(",", "")) / total_value)
        holdings.append(Holding(symbol, (row.get("Name") or "").strip(),
                                (row.get("Sector") or "").strip(),
                                float(weight.replace(",", ""))))
    if not MIN_ROWS <= len(holdings) <= MAX_ROWS:
        raise ValueError(f"Refusing IWB snapshot with {len(holdings)} valid equities")
    return as_of, holdings, skipped


def fetch_holdings() -> tuple[date, list[Holding], list[str]]:
    response = requests.get(HOLDINGS_URL, timeout=45, headers={"User-Agent": "market-data/1.0"})
    response.raise_for_status()
    as_of, holdings, skipped = parse_holdings(response.content.decode("utf-8-sig"))
    excluded = load_excluded_symbols()
    if excluded:
        retained = [holding for holding in holdings if holding.symbol not in excluded]
        removed = sorted({holding.symbol for holding in holdings} - {holding.symbol for holding in retained})
        if removed:
            logging.warning("Excluded retired symbols from IWB holdings: %s", removed)
        holdings = retained
    return as_of, holdings, skipped


def update(as_of: date, holdings: list[Holding], code: str = INDEX_CODE) -> tuple[int, int]:
    current = {h.symbol: h for h in holdings}
    with get_connection(conn_params()) as (conn, cur):
        cur.execute("SELECT index_id FROM market_indices.indices WHERE code = %s FOR UPDATE", (code,))
        index = cur.fetchone()
        if not index:
            raise LookupError(f"Index {code} does not exist in market_indices.indices")
        index_id = index["index_id"]
        cur.execute("SELECT MAX(as_of_date) AS latest FROM market_indices.seed_snapshots WHERE index_id = %s", (index_id,))
        latest = cur.fetchone()["latest"]
        if latest and as_of < latest:
            raise ValueError(f"IWB file date {as_of} predates latest {code} snapshot {latest}")
        if latest == as_of:
            cur.execute("SELECT symbol FROM market_indices.seed_snapshots WHERE index_id = %s AND as_of_date = %s", (index_id, as_of))
            existing = {r["symbol"] for r in cur.fetchall()}
            if existing != set(current):
                raise ValueError(f"Existing {code} snapshot for {as_of} differs from IWB; refusing silent overwrite")
            return 0, 0

        cur.execute("SELECT symbol FROM market_indices.index_constituents WHERE index_id = %s AND end_date IS NULL FOR UPDATE", (index_id,))
        previous = {r["symbol"] for r in cur.fetchall()}
        # Historical EODHD snapshots have no intervals. A multi-year gap must not
        # be filled by implying their membership persisted until the first IWB file.
        if previous and len(current) < 0.6 * len(previous):
            raise ValueError(f"Suspicious IWB universe shrink: {len(previous)} -> {len(current)}")

        cur.execute(
            """INSERT INTO market_indices.sources(source_type, title, url)
               VALUES ('ETF', %s, %s) ON CONFLICT (url) DO UPDATE SET title = EXCLUDED.title
               RETURNING source_id""", (SOURCE_TITLE, HOLDINGS_URL))
        source_id = cur.fetchone()["source_id"]
        for h in holdings:
            cur.execute(
                """INSERT INTO market_indices.seed_snapshots
                   (index_id, as_of_date, symbol, company_name, sector, weight, source_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (index_id, as_of, h.symbol, h.name, h.sector, h.weight, source_id))

        removed = previous - set(current)
        added = set(current) - previous
        for symbol in removed:
            cur.execute(
                """UPDATE market_indices.index_constituents SET end_date = %s
                   WHERE index_id = %s AND symbol = %s AND end_date IS NULL""",
                (as_of, index_id, symbol))
        for symbol in added:
            h = current[symbol]
            cur.execute(
                """INSERT INTO market_indices.index_constituents
                   (index_id, symbol, company_name, sector, start_date, last_verified_asof)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (index_id, symbol, h.name, h.sector, as_of, as_of))
        for symbol in previous & set(current):
            h = current[symbol]
            cur.execute(
                """UPDATE market_indices.index_constituents
                   SET company_name = %s, sector = %s, last_verified_asof = %s
                   WHERE index_id = %s AND symbol = %s AND end_date IS NULL""",
                (h.name, h.sector, as_of, index_id, symbol))
    return len(added), len(removed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default=INDEX_CODE)
    parser.add_argument("--asof", type=date.fromisoformat, default=date.today(),
                        help="Run date; reject an IWB file more than 7 days old")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    as_of, holdings, skipped = fetch_holdings()
    if not as_of <= args.asof or args.asof - as_of > timedelta(days=7):
        raise ValueError(f"IWB holdings date {as_of} is not recent for run date {args.asof}")
    added, removed = update(as_of, holdings, args.code)
    logging.info("%s IWB snapshot %s: %d equities, added=%d, removed=%d, skipped=%s",
                 args.code, as_of, len(holdings), added, removed, skipped)


if __name__ == "__main__":
    main()
