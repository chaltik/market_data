"""Build point-in-time approximate SP500 membership from historical snapshots."""

import argparse
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date
from io import BytesIO, StringIO
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from dateutil import parser as dateparser
from psycopg2.extras import execute_values
try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ModuleNotFoundError:  # Keeps pure parser tests runnable before dependency install.
    def retry(*args, **kwargs):
        return lambda function: function

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return None

from db_config import config as conn_params
from db_utils import get_connection

HISTORY_URL = (
    "https://raw.githubusercontent.com/thuningxu/sp500nq100/main/"
    "sp500_components_history.csv"
)
SPY_HOLDINGS_XLSX = (
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/"
    "holdings-daily-us-en-spy.xlsx"
)
HISTORY_SOURCE_TITLE = "thuningxu/sp500nq100 S&P 500 component history"
SPY_SOURCE_TITLE = "State Street SPDR SPY daily holdings"
INDEX_CODE = "SP500"
MIN_UNIVERSE_ROWS = 450
MAX_UNIVERSE_ROWS = 550
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,7}$")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "sp500-history-builder/1.0 (+local market-data pipeline)"})


@dataclass(frozen=True)
class Source:
    source_type: str
    title: str
    url: str
    published_date: Optional[date]
    raw_text: Optional[str]


@dataclass(frozen=True)
class ChangeEvent:
    action: str
    company_name: Optional[str]
    symbol: Optional[str]
    effective_date: date
    effective_session: str
    announcement_date: Optional[date]
    source_url: str
    source_title: Optional[str]
    source_type: str
    notes: Optional[str] = None


def upsert_source(cur, source: Source) -> int:
    checksum = hashlib.sha256((source.raw_text or "").encode("utf-8")).hexdigest() if source.raw_text else None
    cur.execute(
        """
        INSERT INTO market_indices.sources (source_type, title, url, published_date, raw_text, checksum)
        VALUES (%(source_type)s, %(title)s, %(url)s, %(published_date)s, %(raw_text)s, %(checksum)s)
        ON CONFLICT (url) DO UPDATE SET
            source_type = EXCLUDED.source_type,
            title = COALESCE(EXCLUDED.title, market_indices.sources.title),
            published_date = COALESCE(EXCLUDED.published_date, market_indices.sources.published_date),
            raw_text = COALESCE(EXCLUDED.raw_text, market_indices.sources.raw_text),
            checksum = COALESCE(EXCLUDED.checksum, market_indices.sources.checksum)
        RETURNING source_id
        """,
        {"source_type": source.source_type, "title": source.title, "url": source.url,
         "published_date": source.published_date, "raw_text": source.raw_text, "checksum": checksum},
    )
    return cur.fetchone()["source_id"]


def insert_change_event(cur, index_id: int, event: ChangeEvent, source_id: int) -> int:
    value = f"{INDEX_CODE}|{event.action}|{event.symbol or ''}|{event.company_name or ''}|{event.effective_date.isoformat()}|{event.source_url}"
    event_hash = hashlib.sha256(value.encode("utf-8")).hexdigest()
    cur.execute(
        """
        INSERT INTO market_indices.index_changes_raw
            (index_id, action, company_name, symbol, effective_date, effective_session,
             announcement_date, source_id, notes, event_hash)
        VALUES (%(index_id)s, %(action)s, %(company_name)s, %(symbol)s, %(effective_date)s,
                %(effective_session)s, %(announcement_date)s, %(source_id)s, %(notes)s, %(event_hash)s)
        ON CONFLICT (event_hash) DO NOTHING
        RETURNING change_id
        """,
        {"index_id": index_id, "action": event.action, "company_name": event.company_name,
         "symbol": event.symbol, "effective_date": event.effective_date,
         "effective_session": event.effective_session, "announcement_date": event.announcement_date,
         "source_id": source_id, "notes": event.notes, "event_hash": event_hash},
    )
    row = cur.fetchone()
    return row["change_id"] if row else 0


@dataclass(frozen=True)
class Snapshot:
    as_of: date
    symbols: Tuple[str, ...]


@dataclass(frozen=True)
class Holding:
    symbol: str
    company_name: Optional[str]
    sector: Optional[str]
    weight: Optional[float]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _get(url: str) -> requests.Response:
    response = SESSION.get(url, timeout=30)
    response.raise_for_status()
    return response


def _normalize_symbols(value: object) -> Tuple[str, ...]:
    raw = "" if value is None else str(value)
    symbols, seen = [], set()
    for token in raw.split(","):
        symbol = token.strip().upper()
        if symbol and SYMBOL_RE.fullmatch(symbol) and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return tuple(symbols)


def parse_history_csv(csv_text: str) -> List[Snapshot]:
    """Parse and validate the historical ``date,tickers`` snapshot format."""
    frame = pd.read_csv(StringIO(csv_text), dtype=str)
    columns = {str(column).strip().lower(): column for column in frame.columns}
    if "date" not in columns or "tickers" not in columns:
        raise ValueError("SP500 history must contain date and tickers columns")

    snapshots: List[Snapshot] = []
    previous_date: Optional[date] = None
    for _, row in frame.iterrows():
        as_of = dateparser.parse(str(row[columns["date"]])).date()
        symbols = _normalize_symbols(row[columns["tickers"]])
        if previous_date is not None and as_of <= previous_date:
            raise ValueError("SP500 history dates must be strictly increasing")
        if not MIN_UNIVERSE_ROWS <= len(symbols) <= MAX_UNIVERSE_ROWS:
            raise ValueError(
                f"SP500 snapshot {as_of} has {len(symbols)} symbols; "
                f"expected {MIN_UNIVERSE_ROWS}-{MAX_UNIVERSE_ROWS}"
            )
        snapshots.append(Snapshot(as_of, symbols))
        previous_date = as_of
    if not snapshots:
        raise ValueError("SP500 history contained no snapshots")
    return snapshots


def fetch_history() -> List[Snapshot]:
    response = _get(HISTORY_URL)
    return parse_history_csv(response.content.decode("utf-8-sig"))


def _as_optional_text(value: object) -> Optional[str]:
    text = "" if value is None else str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "-"} else None


def parse_spy_holdings(content: bytes) -> List[Holding]:
    """Parse State Street's XLSX despite its introductory metadata rows."""
    frame = pd.read_excel(BytesIO(content), header=None, dtype=str)
    header_index = None
    for index, row in frame.iterrows():
        values = {str(value).strip().lower() for value in row.tolist() if pd.notna(value)}
        if "name" in values and "ticker" in values:
            header_index = index
            break
    if header_index is None:
        raise ValueError("SPY holdings workbook has no Name/Ticker header")

    headers = [str(value).strip() if pd.notna(value) else "" for value in frame.iloc[header_index]]
    data = frame.iloc[header_index + 1:].copy()
    data.columns = headers
    lower = {column.lower(): column for column in data.columns if column}
    ticker_col = lower.get("ticker")
    name_col = lower.get("name")
    sector_col = lower.get("sector")
    weight_col = lower.get("weight")
    if not ticker_col:
        raise ValueError("SPY holdings workbook has no ticker column")

    holdings: List[Holding] = []
    seen = set()
    for _, row in data.iterrows():
        symbol = _as_optional_text(row.get(ticker_col))
        if not symbol:
            continue
        symbol = symbol.upper()
        if not SYMBOL_RE.fullmatch(symbol) or symbol in seen or symbol == "SPY":
            continue
        seen.add(symbol)
        weight = None
        if weight_col:
            try:
                weight = float(str(row.get(weight_col)).replace(",", ""))
            except (TypeError, ValueError):
                pass
        holdings.append(Holding(
            symbol=symbol,
            company_name=_as_optional_text(row.get(name_col)) if name_col else None,
            sector=_as_optional_text(row.get(sector_col)) if sector_col else None,
            weight=weight,
        ))

    if not MIN_UNIVERSE_ROWS <= len(holdings) <= MAX_UNIVERSE_ROWS:
        raise ValueError(
            f"SPY holdings yielded {len(holdings)} symbols; "
            f"expected {MIN_UNIVERSE_ROWS}-{MAX_UNIVERSE_ROWS}"
        )
    return holdings


def fetch_spy_holdings() -> List[Holding]:
    return parse_spy_holdings(_get(SPY_HOLDINGS_XLSX).content)


def build_transitions(snapshots: Sequence[Snapshot]) -> List[Tuple[str, str, date]]:
    """Return ``(action, symbol, effective_date)`` transitions."""
    transitions: List[Tuple[str, str, date]] = [
        ("ADD", symbol, snapshots[0].as_of) for symbol in snapshots[0].symbols
    ]
    previous = set(snapshots[0].symbols)
    for snapshot in snapshots[1:]:
        current = set(snapshot.symbols)
        transitions.extend(("REMOVE", symbol, snapshot.as_of) for symbol in sorted(previous - current))
        transitions.extend(("ADD", symbol, snapshot.as_of) for symbol in sorted(current - previous))
        previous = current
    return transitions


def compute_intervals(
    transitions: Iterable[Tuple[str, str, date, Optional[int]]],
) -> List[Tuple[str, date, Optional[date], Optional[int], Optional[int]]]:
    """Convert transitions to half-open intervals, preserving ticker changes."""
    active: Dict[str, Tuple[date, Optional[int]]] = {}
    intervals = []
    for action, symbol, effective_date, change_id in transitions:
        if action == "REMOVE":
            previous = active.pop(symbol, None)
            if previous is not None:
                intervals.append((symbol, previous[0], effective_date, previous[1], change_id))
        elif action == "ADD" and symbol not in active:
            active[symbol] = (effective_date, change_id)
    intervals.extend(
        (symbol, start, None, change_id, None)
        for symbol, (start, change_id) in active.items()
    )
    return sorted(intervals, key=lambda item: (item[0], item[1], item[2] or date.max))


def _event_hash(symbol: str, action: str, effective_date: date) -> str:
    value = f"{INDEX_CODE}|{action}|{symbol}||{effective_date.isoformat()}|{HISTORY_URL}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ensure_index(cur) -> int:
    cur.execute(
        """
        INSERT INTO market_indices.indices(code, name, provider, methodology_url, notes)
        VALUES (%(code)s, %(name)s, %(provider)s, %(url)s, %(notes)s)
        ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name,
            provider = EXCLUDED.provider, methodology_url = EXCLUDED.methodology_url,
            notes = EXCLUDED.notes
        RETURNING index_id
        """,
        {
            "code": INDEX_CODE,
            "name": "Approximate S&P 500 large-cap universe",
            "provider": "State Street / public historical reconstruction",
            "url": "https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy",
            "notes": "Approximately 500 names; SPY holdings maintain the current universe.",
        },
    )
    return cur.fetchone()["index_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-spy", action="store_true", help="Do not fetch a current SPY metadata snapshot")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    snapshots = fetch_history()
    holdings = [] if args.skip_spy else fetch_spy_holdings()
    logging.info("Loaded %d historical snapshots; latest is %s", len(snapshots), snapshots[-1].as_of)
    logging.info("Loaded %d current SPY holdings", len(holdings))

    params = conn_params()
    with get_connection(params) as (conn, cur):
        index_id = _ensure_index(cur)

    transitions = build_transitions(snapshots)
    with get_connection(params) as (conn, cur):
        history_source_id = upsert_source(cur, Source("MANUAL", HISTORY_SOURCE_TITLE, HISTORY_URL, None, None))
        rows = [
            (index_id, snapshot.as_of, symbol, None, None, history_source_id)
            for snapshot in snapshots for symbol in snapshot.symbols
        ]
        execute_values(
            cur,
            """
            INSERT INTO market_indices.seed_snapshots
                (index_id, as_of_date, symbol, company_name, sector, source_id)
            VALUES %s ON CONFLICT (index_id, as_of_date, symbol) DO NOTHING
            """,
            rows,
            page_size=5000,
        )
        logging.info("Historical seed rows considered: %d", len(rows))

        if holdings:
            spy_source_id = upsert_source(cur, Source("ETF", SPY_SOURCE_TITLE, SPY_HOLDINGS_XLSX, date.today(), None))
            current_rows = [
                (index_id, date.today(), h.symbol, h.company_name, h.sector, h.weight, spy_source_id)
                for h in holdings
            ]
            execute_values(
                cur,
                """
                INSERT INTO market_indices.seed_snapshots
                    (index_id, as_of_date, symbol, company_name, sector, weight, source_id)
                VALUES %s
                ON CONFLICT (index_id, as_of_date, symbol) DO UPDATE SET
                    company_name = EXCLUDED.company_name, sector = EXCLUDED.sector,
                    weight = EXCLUDED.weight, source_id = EXCLUDED.source_id
                """,
                current_rows,
                page_size=1000,
            )

        event_ids: Dict[Tuple[str, str, date], int] = {}
        for action, symbol, effective_date in transitions:
            event = ChangeEvent(
                action=action, company_name=None, symbol=symbol,
                effective_date=effective_date, effective_session="UNKNOWN",
                announcement_date=None, source_url=HISTORY_URL,
                source_title=HISTORY_SOURCE_TITLE, source_type="MANUAL",
                notes="Transition derived from point-in-time full snapshot",
            )
            insert_change_event(cur, index_id, event, history_source_id)
            cur.execute(
                "SELECT change_id FROM market_indices.index_changes_raw WHERE event_hash = %(hash)s",
                {"hash": _event_hash(symbol, action, effective_date)},
            )
            event_ids[(action, symbol, effective_date)] = cur.fetchone()["change_id"]

        indexed_transitions = [
            (action, symbol, effective_date, event_ids[(action, symbol, effective_date)])
            for action, symbol, effective_date in transitions
        ]
        intervals = compute_intervals(indexed_transitions)
        cur.execute("DELETE FROM market_indices.index_constituents WHERE index_id = %(id)s", {"id": index_id})
        execute_values(
            cur,
            """
            INSERT INTO market_indices.index_constituents
                (index_id, symbol, start_date, end_date, added_change_id, removed_change_id)
            VALUES %s
            """,
            [(index_id, symbol, start, end, add_id, remove_id)
             for symbol, start, end, add_id, remove_id in intervals],
            page_size=5000,
        )
        logging.info("Rebuilt %d constituent intervals", len(intervals))


if __name__ == "__main__":
    main()
